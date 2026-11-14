"""CPU regression checks for real update logic, using a small compatible model.

These are control-flow/gradient tests, not quality claims for DINO checkpoints.
The GAN checks use the actual five-head spectral-normalized DinoDisc with a
small frozen feature backbone; no external weights or downloads are needed.
"""
from copy import deepcopy
import math
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

from hae.checkpoints import read_training_checkpoint, restore_rng_state, seed_all
from hae.distributed import Context
from hae.models import DinoDisc
from hae.train import BatchStream, _checkpoint
from hae.training import TrainingEngine, frozen_discriminator


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 4, 1)
        self.transformer = nn.Conv2d(4, 4, 1)

    def forward(self, x):
        return self.transformer(self.patch_embed(x)).tanh()


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Conv2d(4, 4, 1)
        self.project_out = nn.Sequential(nn.Conv2d(4, 3, 1), nn.Sigmoid())

    def forward(self, x):
        return self.project_out(self.body(x).tanh())


class TinyAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dino_model = TinyEncoder()
        self.decoder = TinyDecoder()

    def forward(self, normalized):
        latent = self.dino_model(normalized)
        return self.decoder(latent), latent.flatten(2).transpose(1, 2)


class TinyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 4, 1)

    def forward_features(self, normalized):
        return {"x_norm_patchtokens": self.projection(normalized).flatten(2).transpose(1, 2)}


class TinyDinoBackbone(nn.Module):
    """Five differentiable feature maps with DinoDisc's frozen-backbone API."""
    embed_dim = 8
    key_depths = (0, 1, 2, 3)

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, self.embed_dim, 1)
        self.blocks = nn.ModuleList([nn.Conv1d(self.embed_dim, self.embed_dim, 1)
                                    for _ in self.key_depths])

    def forward(self, x, grad_ckpt=False):
        features = self.patch_embed(x).flatten(2).tanh()
        outputs = [features]
        for block in self.blocks:
            features = block(features).tanh()
            outputs.append(features)
        return outputs


def tiny_discriminator():
    return DinoDisc(backbone=TinyDinoBackbone(), ks=3,
                    key_depths=TinyDinoBackbone.key_depths)


def config_for(stage=3, warmup=0):
    return {
        "stage": stage, "seed": 123,
        "model": {"dino": {"frozen_backbone_eval": True}},
        "training": {"scope": {1: "cnn_decoder", 2: "decoder", 3: "decoder", 4: "head"}[stage],
                     "learning_rate": 0.01, "betas": [0.5, 0.9], "weight_decay": 0.0,
                     "precision": "fp32", "max_grad_norm": 1000.0,
                     "micro_batch_size": 2, "global_batch_size": 4, "max_updates": 4},
        "losses": {"l1": 1.0, "lpips": 0.0, "alignment": 0.5 if stage == 1 else 0.0},
        "gan": {"enabled": stage == 2, "learning_rate": 0.002, "betas": [0.5, 0.9],
                "warmup_updates": warmup, "weight": 0.5, "adaptive": True, "adaptive_max": 10.0},
        "ema": {"decay": 0.9},
    }


def make_engine(stage=3, model=None, discriminator=None, warmup=0):
    if stage == 2 and discriminator is None:
        discriminator = tiny_discriminator()
    return TrainingEngine(model or TinyAutoencoder(), config_for(stage, warmup), "cpu",
                          teacher=TinyTeacher() if stage == 1 else None,
                          discriminator=discriminator)


def make_stream(data, epoch=0, batch_in_epoch=0, batch_size=2):
    sampler = DistributedSampler(data, num_replicas=1, rank=0, seed=123, shuffle=True, drop_last=True)
    loader = DataLoader(data, sampler=sampler, batch_size=batch_size, drop_last=True,
                        num_workers=0, generator=torch.Generator().manual_seed(456))
    return BatchStream(loader, sampler, epoch=epoch, batch_in_epoch=batch_in_epoch)


def tensor_snapshot(module):
    return {name: value.detach().clone() for name, value in module.named_parameters()}


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        seed_all(2026)

    def assert_parameters_close(self, first, second, atol=1e-7):
        a, b = dict(first.named_parameters()), dict(second.named_parameters())
        self.assertEqual(set(a), set(b))
        for name in a:
            with self.subTest(parameter=name):
                torch.testing.assert_close(a[name], b[name], rtol=1e-6, atol=atol)

    def test_accumulation_equals_full_batch_and_does_not_discard_first_microbatch(self):
        initial = TinyAutoencoder()
        full = make_engine(model=deepcopy(initial))
        accumulated = make_engine(model=deepcopy(initial))
        last_only = make_engine(model=deepcopy(initial))
        for _ in range(3):
            early = torch.rand(2, 3, 8, 8) * 0.25
            late = 0.75 + torch.rand(2, 3, 8, 8) * 0.25
            full.train_update([torch.cat((early, late))], accumulation_steps=1)
            accumulated.train_update([early, late], accumulation_steps=2)
            last_only.train_update([late], accumulation_steps=1)
        self.assert_parameters_close(full.model, accumulated.model)
        self.assertGreater((accumulated.model.decoder.project_out[0].bias
                            - last_only.model.decoder.project_out[0].bias).abs().max().item(), 1e-4)
        self.assertEqual(accumulated.progress["generator_updates"], 3)
        self.assertEqual(accumulated.progress["microbatches"], 6)
        self.assertEqual(accumulated.progress["samples_seen"], 12)
        for optimizer_state in accumulated.optimizer_g.state.values():
            self.assertEqual(optimizer_state["step"].item(), 3)

    def test_all_four_stages_update_only_intended_parameters(self):
        images = [torch.rand(2, 3, 8, 8), torch.rand(2, 3, 8, 8)]
        for stage in (1, 2, 3, 4):
            with self.subTest(stage=stage):
                engine = make_engine(stage=stage)
                before = tensor_snapshot(engine.model)
                teacher_before = tensor_snapshot(engine.teacher) if engine.teacher else {}
                engine.train_update(images, accumulation_steps=2)
                changed = []
                for name, parameter in engine.model.named_parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                        if not torch.equal(parameter, before[name]):
                            changed.append(name)
                    else:
                        self.assertIsNone(parameter.grad, name)
                        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
                self.assertTrue(changed)
                self.assertFalse(any("transformer" in name for name in changed))
                if stage == 1:
                    self.assertTrue(any("patch_embed" in name for name in changed))
                    self.assertGreater(engine.model.dino_model.patch_embed.weight.grad.norm().item(), 0)
                    for name, parameter in engine.teacher.named_parameters():
                        self.assertIsNone(parameter.grad)
                        torch.testing.assert_close(parameter, teacher_before[name], rtol=0, atol=0)
                if stage == 4:
                    self.assertTrue(all(name.startswith("decoder.project_out.") for name in changed))

    def test_gan_warmup_uses_optimizer_updates_and_real_adaptive_gradients(self):
        engine = make_engine(stage=2, warmup=2)
        active = []
        for _ in range(3):
            stats = engine.train_update([torch.rand(2, 3, 8, 8) for _ in range(3)], 3)
            active.append(stats["gan_active"])
            self.assertTrue(all(math.isfinite(value) for value in stats.values()))
        self.assertEqual(active, [0.0, 0.0, 1.0])
        self.assertEqual(engine.progress["stage_update"], 3)
        self.assertEqual(engine.progress["discriminator_updates"], 3)
        self.assertEqual(engine.progress["microbatches"], 9)
        self.assertGreater(stats["gan_gradient_norm"], 0)
        self.assertGreater(stats["rec_gradient_norm"], 0)
        self.assertGreater(stats["weighted_gan_gradient_norm"], 0)
        self.assertTrue(0 <= stats["adaptive_weight"] <= 10)
        self.assertTrue(all(parameter.requires_grad for parameter in engine.discriminator.heads.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in engine.discriminator.backbone.parameters()))
        self.assertFalse(engine.discriminator.backbone.training)

    def test_dinodisc_optimizer_only_updates_heads_and_preserves_backbone(self):
        engine = make_engine(stage=2)
        discriminator = engine.discriminator
        backbone_before = {name: value.clone() for name, value in discriminator.backbone.state_dict().items()}
        heads_before = tensor_snapshot(discriminator.heads)
        optimized_ids = {id(parameter) for group in engine.optimizer_d.param_groups
                         for parameter in group["params"]}
        self.assertEqual(optimized_ids, {id(parameter) for parameter in discriminator.heads.parameters()})
        for _ in range(2):
            stats = engine.train_update([torch.rand(2, 3, 8, 8) for _ in range(2)], 2)
            self.assertFalse(discriminator.backbone.training)
            self.assertTrue(discriminator.heads.training)
            self.assertGreater(stats["gan_gradient_norm"], 0)
            for name, value in discriminator.backbone.state_dict().items():
                torch.testing.assert_close(value, backbone_before[name], rtol=0, atol=0)
            for parameter in discriminator.backbone.parameters():
                self.assertFalse(parameter.requires_grad)
                self.assertIsNone(parameter.grad)
        self.assertTrue(any(not torch.equal(parameter, heads_before[name])
                            for name, parameter in discriminator.heads.named_parameters()))
        for state in engine.optimizer_d.state.values():
            self.assertEqual(state["step"].item(), 2)

    def test_frozen_dinodisc_preserves_input_gradient_modes_and_spectral_buffers(self):
        discriminator = tiny_discriminator().train()
        discriminator(torch.rand(2, 3, 8, 8) * 2 - 1)
        buffers = {name: value.clone() for name, value in discriminator.named_buffers()}
        frozen_before = tensor_snapshot(discriminator.backbone)
        images = torch.rand(2, 3, 8, 8, requires_grad=True)
        with frozen_discriminator(discriminator):
            self.assertFalse(discriminator.training)
            self.assertFalse(discriminator.backbone.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in discriminator.parameters()))
            discriminator(images * 2 - 1).mean().backward()
        self.assertGreater(images.grad.abs().sum().item(), 0)
        self.assertTrue(discriminator.training)
        self.assertTrue(discriminator.heads.training)
        self.assertFalse(discriminator.backbone.training)
        for parameter in discriminator.parameters():
            self.assertIsNone(parameter.grad)
        for name, value in discriminator.named_buffers():
            torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)
        for name, value in discriminator.backbone.named_parameters():
            self.assertFalse(value.requires_grad)
            torch.testing.assert_close(value, frozen_before[name], rtol=0, atol=0)
        self.assertTrue(all(parameter.requires_grad for parameter in discriminator.heads.parameters()))

    def test_generator_backward_does_not_add_discriminator_gradients(self):
        model = TinyAutoencoder()
        discriminator = tiny_discriminator()
        active = make_engine(stage=2, model=deepcopy(model), discriminator=deepcopy(discriminator), warmup=0)
        warmup = make_engine(stage=2, model=deepcopy(model), discriminator=deepcopy(discriminator), warmup=1)
        batches = [torch.rand(2, 3, 8, 8), torch.rand(2, 3, 8, 8)]
        active_stats = active.train_update(batches, 2)
        warmup_stats = warmup.train_update(batches, 2)
        self.assertEqual(active_stats["gan_active"], 1)
        self.assertEqual(warmup_stats["gan_active"], 0)
        self.assert_parameters_close(active.discriminator, warmup.discriminator, atol=0)
        for name, parameter in active.discriminator.named_parameters():
            other = dict(warmup.discriminator.named_parameters())[name]
            if parameter.requires_grad:
                torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)
            else:
                self.assertIsNone(parameter.grad)
                self.assertIsNone(other.grad)
        active_buffers = dict(active.discriminator.named_buffers())
        for name, value in warmup.discriminator.named_buffers():
            torch.testing.assert_close(active_buffers[name], value, rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(
            active.model.decoder.parameters(), warmup.model.decoder.parameters())))

    def test_incomplete_and_nonfinite_windows_never_step_optimizer(self):
        for failure in ("incomplete", "too_many", "nan"):
            with self.subTest(failure=failure):
                engine = make_engine(stage=2)
                before_g, before_d = tensor_snapshot(engine.model), tensor_snapshot(engine.discriminator)
                valid = torch.rand(2, 3, 8, 8)
                if failure == "incomplete":
                    batches, count, error = [valid], 2, ValueError
                elif failure == "too_many":
                    batches, count, error = [valid, valid], 1, ValueError
                else:
                    invalid = valid.clone()
                    invalid[0, 0, 0, 0] = float("nan")
                    batches, count, error = [valid, invalid], 2, FloatingPointError
                with self.assertRaises(error):
                    engine.train_update(batches, count)
                for module, before in ((engine.model, before_g), (engine.discriminator, before_d)):
                    for name, parameter in module.named_parameters():
                        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
                self.assertEqual(engine.progress["stage_update"], 0)
                self.assertEqual(engine.progress["generator_updates"], 0)
                self.assertEqual(engine.progress["discriminator_updates"], 0)
                self.assertEqual(len(engine.optimizer_g.state), 0)
                self.assertEqual(len(engine.optimizer_d.state), 0)

    def test_batch_stream_resume_preserves_order_across_epoch_boundary(self):
        data = torch.arange(17)
        for skip in (0, 4, 5, 8):
            with self.subTest(skip=skip):
                original = make_stream(data, batch_size=3)
                for _ in range(skip):
                    next(original)
                state = original.state_dict()
                resumed = make_stream(data, **state, batch_size=3)
                for _ in range(13):
                    torch.testing.assert_close(next(original), next(resumed), rtol=0, atol=0)
                    self.assertEqual(original.state_dict(), resumed.state_dict())

    def test_full_engine_checkpoint_replays_next_gan_update_and_ema(self):
        data = torch.rand(11, 3, 8, 8)
        engine = make_engine(stage=2, warmup=1)
        stream = make_stream(data)
        engine.train_update([next(stream), next(stream)], 2)

        def next_batches(current_stream):
            shift = random.random() * 0.01 + float(np.random.rand()) * 0.01
            return [(next(current_stream) + torch.rand(()) * 0.01 + shift).clamp(0, 1) for _ in range(2)]

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _checkpoint(engine, engine.config, stream, Context(), {"test": True}, directory)
            expected_stats = engine.train_update(next_batches(stream), 2)
            expected_ema = engine.ema.state_dict()
            loaded = read_training_checkpoint(directory / "checkpoint_last.pt")
            resumed_model = TinyAutoencoder()
            resumed_model.load_state_dict(loaded["model"], strict=True)
            resumed = make_engine(stage=2, model=resumed_model, warmup=1)
            resumed.optimizer_g.load_state_dict(loaded["optimizer_g"])
            resumed.discriminator.load_state_dict(loaded["discriminator"], strict=True)
            resumed.optimizer_d.load_state_dict(loaded["optimizer_d"])
            resumed.ema.load_state_dict(loaded["ema"])
            for key in resumed.progress:
                resumed.progress[key] = loaded["progress"][key]
            resumed_stream = make_stream(data, loaded["progress"]["epoch"], loaded["progress"]["batch_in_epoch"])
            restore_rng_state(loaded["rng_states"][0])
            actual_stats = resumed.train_update(next_batches(resumed_stream), 2)
        self.assert_parameters_close(engine.model, resumed.model, atol=0)
        self.assert_parameters_close(engine.discriminator, resumed.discriminator, atol=0)
        for name, value in engine.discriminator.state_dict().items():
            torch.testing.assert_close(value, resumed.discriminator.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(engine.progress, resumed.progress)
        self.assertEqual(stream.state_dict(), resumed_stream.state_dict())
        self.assertEqual(expected_stats, actual_stats)
        for name, value in expected_ema["shadow"].items():
            torch.testing.assert_close(value, resumed.ema.state_dict()["shadow"][name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
