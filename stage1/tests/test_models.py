"""Structural and CPU numerical checks of the autoencoder; no weights or downloads."""
import ast
import copy
import hashlib
import io
import json
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from hae.models import (build_autoencoder, build_teacher,
                                 configure_trainable, get_last_layer)
from hae.models.factory import Autoencoder
from hae.models.generator import DecoderConfig, RMSNorm2d
from hae.models.efficientvit_compat import EfficientViTBlock

ROOT = Path(__file__).resolve().parents[1]


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 32
        self.patch_embed = nn.Conv2d(3, 32, 16, 16)
        self.transformer = nn.Linear(32, 32)

    def forward_features(self, x):
        x = self.patch_embed(x)
        if isinstance(self.patch_embed, nn.Conv2d):
            x = x.permute(0, 2, 3, 1)
        x = self.transformer(x.flatten(1, 2))
        return {"x_norm_patchtokens": x}


def tiny_config():
    return DecoderConfig(in_channels=3, latent_channels=32,
                         width_list=[8, 16, 32, 32, 32], depth_list=[1] * 5,
                         block_type=["ResBlock"] * 3 + ["EViT_GLU"] * 2,
                         upsample_block_type="InterpolateConv")


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_trainable_scopes_and_encoder_gradient_flow(self):
        model = Autoencoder(TinyBackbone(), tiny_config())
        for scope in ("cnn_decoder", "decoder", "head"):
            model.zero_grad(set_to_none=True)
            configure_trainable(model, scope)
            model.train()
            actual = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
            expected = {name for name, _ in model.named_parameters()
                        if (scope == "cnn_decoder" and (name.startswith("dino_model.patch_embed.")
                                                        or name.startswith("decoder.")))
                        or (scope == "decoder" and name.startswith("decoder."))
                        or (scope == "head" and name.startswith("decoder.project_out."))}
            self.assertEqual(actual, expected)
            self.assertFalse(model.dino_model.transformer.training)
            reconstruction, tokens = model(torch.randn(2, 3, 32, 32))
            reconstruction.square().mean().backward()
            self.assertEqual(tokens.requires_grad, scope == "cnn_decoder")
            if scope == "cnn_decoder":
                grad = model.dino_model.patch_embed.cnn[0].conv.weight.grad
                self.assertIsNotNone(grad)
                self.assertGreater(grad.abs().sum().item(), 0)
            for name, parameter in model.named_parameters():
                if name not in expected:
                    self.assertIsNone(parameter.grad, name)
            self.assertTrue(get_last_layer(model).requires_grad)

    @unittest.skipUnless((ROOT / "vendor/dinov3/hubconf.py").is_file(), "needs facebookresearch/dinov3 in vendor/dinov3")
    def test_offline_factory_and_teacher(self):
        config = {"dino": {"repo_path": str(ROOT / "vendor/dinov3"), "model_name": "dinov3_vitl16"},
                  "decoder": vars(tiny_config())}
        constructor = mock.Mock(return_value=TinyBackbone())
        with mock.patch("hae.models.factory._backbone_constructor", return_value=constructor):
            build_autoencoder(config, load_pretrained=False)
            self.assertEqual(constructor.call_args.kwargs["pretrained"], False)
        with mock.patch("hae.models.factory._backbone_constructor") as constructor:
            with self.assertRaises(FileNotFoundError):
                build_autoencoder(config, load_pretrained=True)
            constructor.assert_not_called()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "teacher.pt"
            backbone = TinyBackbone()
            torch.save(backbone.state_dict(), path)
            config["dino"]["weights_path"] = str(path)
            with mock.patch("hae.models.factory._backbone_constructor",
                            return_value=mock.Mock(return_value=copy.deepcopy(backbone))):
                teacher = build_teacher(config)
            self.assertFalse(teacher.training)
            self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))
            self.assertIsInstance(teacher.patch_embed, nn.Conv2d)

    def test_efficientvit_norm_instance_behavior(self):
        block = EfficientViTBlock(32, norm=RMSNorm2d(32), act_func="silu",
                                  local_module="GLUMBConv", scales=())
        self.assertIsNone(block.context_module.main.proj.norm)
        self.assertIsNone(block.local_module.main.point_conv.norm)
        self.assertNotIn("context_module.main.proj.norm.weight", block.state_dict())

    def test_backbone_train_mode_is_explicit(self):
        model = Autoencoder(TinyBackbone(), tiny_config(),
                                              frozen_backbone_eval=False)
        model.train()
        self.assertTrue(model.dino_model.training)
        configure_trainable(model, "decoder")
        self.assertTrue(model.dino_model.training)
        self.assertTrue(model.encoder_frozen)
        model.eval()
        self.assertFalse(model.dino_model.training)

    @unittest.skipUnless((ROOT / "vendor/dinov3/hubconf.py").is_file(), "needs facebookresearch/dinov3 in vendor/dinov3")
    def test_full_vitl_architecture_without_allocating_weights(self):
        import torch._dynamo
        decoder = dict(in_channels=3, latent_channels=1024,
                       width_list=[256, 512, 512, 1024, 1024],
                       depth_list=[5, 5, 3, 3, 3],
                       block_type=["ResBlock"] * 3 + ["EViT_GLU"] * 2,
                       norm="trms2d", act="silu", in_shortcut="duplicating",
                       upsample_block_type="InterpolateConv", upsample_match_channel=True,
                       upsample_shortcut="duplicating", out_norm="trms2d", out_act="relu")
        with mock.patch("torch.hub.load_state_dict_from_url", side_effect=AssertionError("Unexpected download")):
            with torch.device("meta"):
                model = build_autoencoder({"dino": {"repo_path": str(ROOT / "vendor/dinov3"),
                                                     "model_name": "dinov3_vitl16"},
                                           "decoder": decoder}, load_pretrained=False)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 477108611)
        self.assertEqual(sum(p.numel() for p in model.decoder.parameters()), 172004099)
        self.assertEqual(len(model.decoder.state_dict()), 107)
        configure_trainable(model, "head")
        self.assertEqual({name for name, p in model.named_parameters() if p.requires_grad},
                         {"decoder.project_out.op_list.0.weight",
                          "decoder.project_out.op_list.2.conv.weight",
                          "decoder.project_out.op_list.2.conv.bias"})


if __name__ == "__main__":
    unittest.main()
