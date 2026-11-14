"""SigLIP2 vision tower, weight loading and the SigLIP2 autoencoder on tiny architectures."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn
from safetensors.torch import save_file

from hae import encoders
from hae.config import load_config, validate_config
from hae.ema import ModelEMA
from hae.models import configure_trainable, get_last_layer
from hae.models.generator import CnnPatchEmbed, ConvLayer, RMSNorm2d
from hae.siglip2 import (Siglip2HcpeAutoencoder, Siglip2VisionArch, Siglip2VisionTower,
                                  build_autoencoder, build_teacher, load_vision_weights,
                                  validate_model_config)
from hae.siglip2.vision import PatchifyEmbed, map_hf_vision_state_dict
from hae.training import TrainingEngine, normalize_imagenet

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs/siglip2"
TINY_ARCH = {"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
             "num_attention_heads": 4, "image_size": 64, "patch_size": 16}
LARGE_ARCH = {"hidden_size": 1024, "intermediate_size": 4096, "num_hidden_layers": 24,
              "num_attention_heads": 16, "image_size": 256, "patch_size": 16}


def tiny_decoder():
    return {"in_channels": 3, "latent_channels": 32, "width_list": [8, 16, 32, 32, 32],
            "depth_list": [1] * 5, "block_type": ["ResBlock"] * 3 + ["EViT_GLU"] * 2,
            "norm": "trms2d", "act": "silu", "in_shortcut": "duplicating",
            "upsample_block_type": "InterpolateConv", "upsample_match_channel": True,
            "upsample_shortcut": "duplicating", "out_norm": "trms2d", "out_act": "relu"}


def model_config(weights_path="missing/vision_model.safetensors", **encoder):
    settings = {"type": "siglip2", "weights_path": weights_path, "arch": dict(TINY_ARCH)}
    settings.update(encoder)
    return {"encoder": settings, "decoder": tiny_decoder(), "dino": {"frozen_backbone_eval": True}}


def hf_style_state(tower):
    """Invert the tower's key mapping to fabricate a Hugging Face SiglipModel checkpoint."""
    renamed = {}
    for name, value in tower.state_dict().items():
        if name.startswith("patch_embed.proj."):
            key = "vision_model.embeddings.patch_embedding." + name[len("patch_embed.proj."):]
        elif name == "pos_embed.weight":
            key = "vision_model.embeddings.position_embedding.weight"
        elif name.startswith("blocks."):
            key = "vision_model.encoder.layers." + name[len("blocks."):]
        elif name.startswith("norm."):
            key = "vision_model.post_layernorm." + name[len("norm."):]
        else:
            raise AssertionError(f"Unmapped tower tensor: {name}")
        renamed[key] = value.detach().clone()
    return renamed


def write_fake_checkpoint(directory, tower):
    state = hf_style_state(tower)
    state["vision_model.head.probe"] = torch.zeros(1, 1, TINY_ARCH["hidden_size"])
    state["text_model.embeddings.token_embedding.weight"] = torch.zeros(4, TINY_ARCH["hidden_size"])
    state["logit_scale"] = torch.zeros(1)
    path = Path(directory) / "vision_model.safetensors"
    save_file(state, str(path))
    return path


def engine_config(scope="cnn_decoder"):
    return {"stage": 1, "seed": 123, "model": model_config(),
            "training": {"scope": scope, "learning_rate": 0.01, "betas": [0.9, 0.999],
                         "weight_decay": 0.0, "precision": "fp32", "max_grad_norm": 1000.0,
                         "micro_batch_size": 2, "global_batch_size": 4, "max_updates": 4},
            "losses": {"l1": 1.0, "lpips": 0.0, "alignment": 0.5},
            "gan": {"enabled": False}, "ema": {"decay": 0.9, "warmup": True}}


class Siglip2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def test_arch_from_hf_config(self):
        arch = Siglip2VisionArch.from_hf_config({"model_type": "siglip", "vision_config": {
            "hidden_size": 1024, "image_size": 256, "intermediate_size": 4096,
            "model_type": "siglip_vision_model", "num_attention_heads": 16, "num_hidden_layers": 24}})
        self.assertEqual((arch.patch_size, arch.grid_size, arch.head_dim), (16, 16, 64))
        with self.assertRaises(ValueError):
            Siglip2VisionArch.from_hf_config({"vision_config": {"hidden_act": "gelu"}})
        with self.assertRaises(ValueError):
            Siglip2VisionArch.from_hf_config({"vision_config": {"attention_dropout": 0.1}})
        with self.assertRaises(ValueError):
            Siglip2VisionArch(hidden_size=30, num_attention_heads=4)

    def test_hcpe_is_the_paper_table7_module(self):
        """HCPE (Hyperspherical Autoencoder, Table 7): 7x7 s2 64 / 3x3 s2 128 / 3x3 s2 256 / 3x3 s2 D."""
        hcpe = CnnPatchEmbed(embed_dim=1024)
        layers = list(hcpe.cnn)
        self.assertEqual(len(layers), 4)
        expected = [(7, 64, True), (3, 128, True), (3, 256, True), (3, 1024, False)]
        for layer, (kernel, channels, has_norm_act) in zip(layers, expected):
            self.assertIsInstance(layer, ConvLayer)
            self.assertEqual(layer.conv.kernel_size, (kernel, kernel))
            self.assertEqual(layer.conv.stride, (2, 2))
            self.assertEqual(layer.conv.out_channels, channels)
            self.assertEqual(isinstance(layer.norm, RMSNorm2d), has_norm_act)
            self.assertEqual(isinstance(layer.act, nn.SiLU), has_norm_act)
        tokens = hcpe(torch.randn(1, 3, 256, 256))
        self.assertEqual(tuple(tokens.shape), (1, 16, 16, 1024))

    def test_hf_key_mapping_roundtrip_and_strictness(self):
        arch = Siglip2VisionArch(**TINY_ARCH)
        source = Siglip2VisionTower(arch)
        with tempfile.TemporaryDirectory() as directory:
            path = write_fake_checkpoint(directory, source)
            target = Siglip2VisionTower(arch)
            report = load_vision_weights(target, path)
            for name, value in source.state_dict().items():
                torch.testing.assert_close(target.state_dict()[name], value, rtol=0, atol=0)
            self.assertEqual(report["tensor_count"], len(source.state_dict()))
            with self.assertRaises(FileNotFoundError):
                load_vision_weights(Siglip2VisionTower(arch), Path(directory) / "absent.safetensors")
            target.patch_embed = CnnPatchEmbed(embed_dim=arch.hidden_size)
            with self.assertRaises(TypeError):
                load_vision_weights(target, path)
        with self.assertRaises(KeyError):
            map_hf_vision_state_dict({"vision_model.encoder.extra.weight": torch.zeros(1)})
        with self.assertRaises(KeyError):
            map_hf_vision_state_dict({"unrelated.weight": torch.zeros(1)})
        mapped = map_hf_vision_state_dict({**hf_style_state(source), "vision_model.head.probe": torch.zeros(1)})
        self.assertNotIn("head.probe", mapped)

    def test_forward_features_api_and_renormalization(self):
        arch = Siglip2VisionArch(**TINY_ARCH)
        siglip_path = Siglip2VisionTower(arch, input_normalization="siglip").eval()
        imagenet_path = Siglip2VisionTower(arch, input_normalization="imagenet").eval()
        imagenet_path.load_state_dict(siglip_path.state_dict())
        images = torch.rand(2, 3, 64, 64)
        with torch.no_grad():
            reference = siglip_path.forward_features(images * 2 - 1)
            converted = imagenet_path.forward_features(normalize_imagenet(images))
        self.assertEqual(tuple(reference["x_norm_patchtokens"].shape), (2, 16, 32))
        torch.testing.assert_close(converted["x_norm_patchtokens"], reference["x_norm_patchtokens"],
                                   rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(converted["x_prenorm_patchtokens"], reference["x_prenorm_patchtokens"],
                                   rtol=1e-5, atol=1e-5)
        self.assertNotIn("renorm_scale", imagenet_path.state_dict())
        partial = Siglip2VisionTower(arch, feature_layer=1, post_layernorm=False, input_normalization="siglip").eval()
        partial.load_state_dict(siglip_path.state_dict())
        with torch.no_grad():
            manual = siglip_path.blocks[0](siglip_path.embed(images * 2 - 1))
            torch.testing.assert_close(partial.forward_features(images * 2 - 1)["x_norm_patchtokens"], manual)
        with self.assertRaises(ValueError):
            Siglip2VisionTower(arch, feature_layer=3)
        with self.assertRaises(ValueError):
            siglip_path.forward_features(torch.rand(1, 3, 128, 128))
        flexible = Siglip2VisionTower(arch, input_normalization="siglip", allow_pos_interpolation=True)
        self.assertEqual(tuple(flexible(torch.rand(1, 3, 128, 128)).shape), (1, 64, 32))

    def test_hcpe_injection_scopes_and_gradient_flow(self):
        model = build_autoencoder(model_config(), load_pretrained=False)
        self.assertIsInstance(model, Siglip2HcpeAutoencoder)
        self.assertIsInstance(model.hcpe, CnnPatchEmbed)
        self.assertIs(model.siglip2, model.dino_model)
        self.assertTrue(all(name.startswith(("dino_model.", "decoder.")) for name in model.state_dict()))
        for scope in ("cnn_decoder", "decoder", "head"):
            model.zero_grad(set_to_none=True)
            configure_trainable(model, scope)
            model.train()
            self.assertFalse(model.dino_model.blocks[0].training)
            actual = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
            expected = {name for name, _ in model.named_parameters()
                        if (scope == "cnn_decoder" and name.startswith(("dino_model.patch_embed.", "decoder.")))
                        or (scope == "decoder" and name.startswith("decoder."))
                        or (scope == "head" and name.startswith("decoder.project_out."))}
            self.assertEqual(actual, expected)
            reconstruction, tokens = model(normalize_imagenet(torch.rand(2, 3, 64, 64)))
            self.assertEqual(tuple(reconstruction.shape), (2, 3, 64, 64))
            self.assertEqual(tuple(tokens.shape), (2, 16, 32))
            reconstruction.square().mean().backward()
            self.assertEqual(tokens.requires_grad, scope == "cnn_decoder")
            if scope == "cnn_decoder":
                grad = model.hcpe.cnn[0].conv.weight.grad
                self.assertIsNotNone(grad)
                self.assertGreater(grad.abs().sum().item(), 0)
            for name, parameter in model.named_parameters():
                if name not in expected:
                    self.assertIsNone(parameter.grad, name)
            self.assertTrue(get_last_layer(model).requires_grad)
            ModelEMA(model, decay=0.9)

    def test_pretrained_loading_teacher_and_config_json(self):
        arch = Siglip2VisionArch(**TINY_ARCH)
        pretrained = Siglip2VisionTower(arch)
        with tempfile.TemporaryDirectory() as directory:
            path = write_fake_checkpoint(directory, pretrained)
            config = model_config(weights_path=str(path))
            teacher = build_teacher(config)
            self.assertIsInstance(teacher.patch_embed, PatchifyEmbed)
            self.assertFalse(teacher.training)
            self.assertFalse(any(p.requires_grad for p in teacher.parameters()))
            for name, value in pretrained.state_dict().items():
                torch.testing.assert_close(teacher.state_dict()[name], value, rtol=0, atol=0)
            student = build_autoencoder(config, load_pretrained=True)
            self.assertIsInstance(student.hcpe, CnnPatchEmbed)
            for name, value in pretrained.state_dict().items():
                if not name.startswith("patch_embed."):
                    torch.testing.assert_close(student.dino_model.state_dict()[name], value, rtol=0, atol=0)
            with torch.no_grad():
                target = teacher.forward_features(normalize_imagenet(torch.rand(1, 3, 64, 64)))
            self.assertEqual(tuple(target["x_norm_patchtokens"].shape), (1, 16, 32))
            (Path(directory) / "config.json").write_text(json.dumps(
                {"model_type": "siglip", "vision_config": {**TINY_ARCH, "model_type": "siglip_vision_model"}}))
            from_json = model_config(weights_path=str(path), arch=None)
            self.assertEqual(validate_model_config(from_json, image_size=64), arch)
            build_autoencoder(from_json, load_pretrained=True)
        with self.assertRaises(FileNotFoundError):
            build_autoencoder(model_config(), load_pretrained=True)
        with self.assertRaises(FileNotFoundError):
            build_teacher(model_config())

    def test_dispatcher_routes_by_encoder_type(self):
        dinov3 = {"dino": {"model_name": "dinov3_vitl16"}, "decoder": tiny_decoder()}
        with mock.patch("hae.models.build_autoencoder") as factory:
            encoders.build_autoencoder(dinov3, load_pretrained=False)
            factory.assert_called_once_with(dinov3, False)
        with mock.patch("hae.models.build_teacher") as factory:
            encoders.build_teacher(dinov3)
            factory.assert_called_once_with(dinov3)
        with mock.patch("hae.siglip2.factory.build_autoencoder") as factory:
            encoders.build_autoencoder(model_config(), load_pretrained=False)
            factory.assert_called_once()
        with mock.patch("hae.siglip2.factory.build_teacher") as factory:
            encoders.build_teacher(model_config())
            factory.assert_called_once()
        with self.assertRaises(ValueError):
            encoders.encoder_type({"encoder": {"type": "clip"}})
        self.assertEqual(encoders.encoder_type(dinov3), "dinov3")

    def test_one_stage1_update_trains_hcpe_and_decoder_only(self):
        arch = Siglip2VisionArch(**TINY_ARCH)
        pretrained = Siglip2VisionTower(arch)
        with tempfile.TemporaryDirectory() as directory:
            path = write_fake_checkpoint(directory, pretrained)
            config = engine_config()
            config["model"]["encoder"]["weights_path"] = str(path)
            model = build_autoencoder(config["model"], load_pretrained=True)
            teacher = build_teacher(config["model"])
            engine = TrainingEngine(model, config, "cpu", teacher=teacher)
            before = {name: value.detach().clone() for name, value in model.named_parameters()}
            images = [torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64)]
            stats = engine.train_update(iter(images), accumulation_steps=2)
            self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in stats.values()))
            self.assertGreater(stats["alignment"], 0.0)
            self.assertGreater(stats["l1"], 0.0)
            self.assertEqual(stats["lpips"], 0.0)
            self.assertEqual(engine.progress["stage_update"], 1)
            for name, value in model.named_parameters():
                changed = not torch.equal(before[name], value.detach())
                trainable = name.startswith(("dino_model.patch_embed.", "decoder."))
                self.assertEqual(changed, trainable, name)
            for name, value in teacher.state_dict().items():
                torch.testing.assert_close(value, pretrained.state_dict()[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
