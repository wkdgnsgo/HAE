import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image
import torch
from torch import nn

from hae.checkpoints import (atomic_save, capture_rng_state, load_model_weights,
                                      read_training_checkpoint, restore_rng_state, seed_all)
from hae.data import ImageDataset, dataset_fingerprint
from hae.ema import ModelEMA


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 3)
        self.head = nn.Linear(3, 2)
        self.encoder.requires_grad_(False)
        self.register_buffer("counter", torch.tensor(0))
        self.register_buffer("derived", torch.tensor(1), persistent=False)

    def forward(self, x):
        return self.head(self.encoder(x))


def package(model, ema):
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=0.01)
    return {"format_version": 1, "model": model.state_dict(),
            "ema": ema.state_dict(), "optimizer_g": optimizer.state_dict(),
            "optimizer_d": None, "discriminator": None,
            "progress": {"update_step": 7, "epoch": 2, "batch_in_epoch": 3},
            "config": {"stage": 3}, "rng_states": [capture_rng_state()]}


class StateTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        seed_all(1337)

    def test_ema_initialized_after_loading_preserves_frozen_weights(self):
        source, target = TinyModel(), TinyModel()
        with torch.no_grad():
            source.encoder.weight.fill_(8)
            source.head.weight.fill_(2)
        path = self.root / "old_ema.pt"
        atomic_save(source.state_dict(), path)
        report = load_model_weights(path, target, weights="ema")
        ema = ModelEMA(target, decay=0.5)
        with torch.no_grad():
            target.head.weight.fill_(4)
            target.counter.fill_(5)
        ema.update(target)
        exported = ema.full_state_dict(target)
        self.assertTrue(torch.equal(exported["encoder.weight"], torch.full_like(target.encoder.weight, 8)))
        self.assertTrue(torch.equal(exported["head.weight"], torch.full_like(target.head.weight, 3)))
        self.assertEqual(exported["counter"].item(), 5)
        self.assertEqual(report["selected_weights"], "ema")
        self.assertNotIn("encoder.weight", ema.shadow)
        self.assertNotIn("derived", exported)
        self.assertNotIn("derived", ema.buffers)

    def test_ema_context_restores_after_exception_and_rejects_stage_change(self):
        model = TinyModel()
        ema = ModelEMA(model, decay=0.5)
        original = model.head.weight.detach().clone()
        with torch.no_grad():
            model.head.weight.add_(4)
        updated = model.head.weight.detach().clone()
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with ema.average_parameters(model):
                torch.testing.assert_close(model.head.weight, original)
                raise RuntimeError("intentional")
        torch.testing.assert_close(model.head.weight, updated)
        model.encoder.requires_grad_(True)
        with self.assertRaises(ValueError):
            ema.update(model)

    def test_ema_resume_validates_schema_and_restores_device_dtype(self):
        model = TinyModel().double()
        ema = ModelEMA(model, 0.5)
        state = ema.state_dict()
        other = ModelEMA(model, 0.9)
        other.load_state_dict(state)
        self.assertEqual(other.decay, 0.5)
        self.assertEqual(other.shadow["head.weight"].dtype, torch.float64)
        invalid = {**state, "shadow": {}}
        with self.assertRaises(ValueError):
            other.load_state_dict(invalid)
        with self.assertRaises(ValueError):
            other.load_state_dict(state["shadow"])

    def test_missing_mismatched_and_incomplete_weights_never_silently_load(self):
        model = TinyModel()
        with self.assertRaises(FileNotFoundError):
            load_model_weights(self.root / "missing.pt", model)
        original = model.head.weight.detach().clone()
        bad = dict(model.state_dict())
        bad["head.weight"] = torch.zeros(5, 3)
        path = self.root / "bad.pt"
        atomic_save(bad, path)
        with self.assertRaises(ValueError):
            load_model_weights(path, model)
        torch.testing.assert_close(model.head.weight, original)
        bad = dict(model.state_dict())
        del bad["encoder.weight"]
        atomic_save(bad, path)
        with self.assertRaises(ValueError):
            load_model_weights(path, model)

    def test_package_requires_unambiguous_raw_or_ema_selection(self):
        model = TinyModel()
        ema = ModelEMA(model, 0.5)
        with torch.no_grad():
            model.head.weight.add_(4)
        path = self.root / "training.pt"
        atomic_save(package(model, ema), path)
        target = TinyModel()
        with self.assertRaisesRegex(ValueError, "Both raw and EMA"):
            load_model_weights(path, target)
        load_model_weights(path, target, weights="raw")
        torch.testing.assert_close(target.head.weight, model.head.weight)
        load_model_weights(path, target, weights="ema")
        torch.testing.assert_close(target.head.weight, ema.shadow["head.weight"])
        payload = package(model, ema)
        payload["ema"] = None
        atomic_save(payload, path)
        with self.assertRaisesRegex(ValueError, "missing"):
            load_model_weights(path, target, weights="ema")

    def test_safe_full_resume_replays_rng_and_optimizer_update(self):
        model = TinyModel()
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=0.01)
        x = torch.randn(4, 3)
        model(x).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        ema = ModelEMA(model)
        state = package(model, ema)
        state["optimizer_g"] = optimizer.state_dict()
        path = self.root / "resume.pt"
        atomic_save(state, path)
        expected_random = (random.random(), float(np.random.rand()), torch.rand(3))
        next_x = torch.randn(4, 3)
        model(next_x).square().mean().backward()
        optimizer.step()
        expected_weight = model.head.weight.detach().clone()

        loaded = read_training_checkpoint(path)
        resumed = TinyModel()
        resumed.load_state_dict(loaded["model"])
        resumed_optimizer = torch.optim.AdamW(resumed.head.parameters(), lr=0.01)
        resumed_optimizer.load_state_dict(loaded["optimizer_g"])
        restore_rng_state(loaded["rng_states"][0])
        self.assertEqual(random.random(), expected_random[0])
        self.assertEqual(float(np.random.rand()), expected_random[1])
        torch.testing.assert_close(torch.rand(3), expected_random[2], rtol=0, atol=0)
        resumed(torch.randn(4, 3)).square().mean().backward()
        resumed_optimizer.step()
        torch.testing.assert_close(resumed.head.weight, expected_weight, rtol=0, atol=0)
        self.assertEqual(loaded["progress"]["update_step"], 7)
        self.assertEqual(loaded["progress"]["batch_in_epoch"], 3)

    def test_atomic_failed_write_preserves_previous_checkpoint(self):
        path = self.root / "state.pt"
        atomic_save({"version": 1}, path)
        with mock.patch("torch.save", side_effect=OSError("full disk")):
            with self.assertRaises(OSError):
                atomic_save({"version": 2}, path)
        self.assertEqual(torch.load(path, weights_only=True)["version"], 1)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["state.pt"])

    def test_exact_resume_rejects_model_only_checkpoint(self):
        path = self.root / "weights.pt"
        atomic_save(TinyModel().state_dict(), path)
        with self.assertRaises(ValueError):
            read_training_checkpoint(path)

    def test_data_is_sorted_rgb_square_and_repeatable(self):
        (self.root / "class_a").mkdir()
        Image.new("L", (13, 7), 128).save(self.root / "class_a" / "b.png")
        Image.new("RGB", (7, 13), (255, 0, 0)).save(self.root / "a.png")
        dataset = ImageDataset(self.root, image_size=8)
        self.assertEqual([p.name for p in dataset.paths], ["a.png", "b.png"])
        self.assertEqual(dataset[1].shape, (3, 8, 8))
        self.assertEqual(dataset[1].dtype, torch.float32)
        torch.testing.assert_close(dataset[1], torch.full((3, 8, 8), 128 / 255))
        self.assertEqual(dataset_fingerprint(self.root), dataset_fingerprint(self.root))
        old = dataset_fingerprint(self.root, hash_contents=True)
        Image.new("L", (13, 7), 64).save(self.root / "class_a" / "b.png")
        self.assertNotEqual(old["sha256"], dataset_fingerprint(self.root, hash_contents=True)["sha256"])


if __name__ == "__main__":
    unittest.main()
