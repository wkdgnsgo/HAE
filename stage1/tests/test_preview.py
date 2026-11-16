"""Fixed EMA preview integration checks with the actual TrainingEngine, offline."""
from copy import deepcopy
import json
import random

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from hae.checkpoints import capture_rng_state, restore_rng_state, seed_all
from hae.preview import FixedPreview
from hae.training import TrainingEngine


class PreviewModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dino_model = nn.Conv2d(3, 3, 1)
        self.decoder = nn.Module()
        self.decoder.body = nn.Conv2d(3, 3, 1)
        self.decoder.project_out = nn.Conv2d(3, 3, 1)
        self.register_buffer("counter", torch.tensor(0))
        self.fail = False

    def forward(self, normalized):
        assert normalized.dtype == torch.float32
        assert not torch.is_autocast_enabled(normalized.device.type)
        random.random(), np.random.random(), torch.rand(())
        if self.fail:
            self.counter.add_(123)
            raise RuntimeError("deliberate preview failure")
        latent = self.dino_model(normalized)
        result = self.decoder.project_out(self.decoder.body(latent))
        return result, latent.flatten(2).transpose(1, 2)


def configuration(root, count=10):
    return {"stage": 4, "seed": 3,
            "data": {"val_root": str(root), "image_size": 4},
            "logging": {"wandb": {"num_images": count}},
            "training": {"scope": "head", "learning_rate": 0.001,
                         "betas": [0.9, 0.999], "weight_decay": 0.,
                         "precision": "fp32", "max_grad_norm": 1.},
            "gan": {"enabled": False}, "losses": {"l1": 1., "lpips": 0., "alignment": 0.},
            "ema": {"decay": .9}}


def make_images(root, count=10):
    root.mkdir()
    for index in reversed(range(count)):
        Image.fromarray(np.full((4, 4, 3), index * 10, dtype=np.uint8)).save(root / f"{index:02}.png")


def make_engine(config):
    model = PreviewModel()
    with torch.no_grad():
        model.decoder.project_out.weight.zero_()
        model.decoder.project_out.bias.fill_(.8)
    engine = TrainingEngine(model, config, "cpu")
    engine.ema.shadow["decoder.project_out.bias"].fill_(.25)
    return engine


def assert_nested_equal(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_nested_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            assert_nested_equal(left, right)
    else:
        assert first == second


def test_fixed_ema_preview_layout_and_gt_psnr(tmp_path):
    root = tmp_path / "val"
    make_images(root)
    config = configuration(root)
    preview = FixedPreview(config, tmp_path / "run")
    result = preview.render(make_engine(config))
    pixels = np.asarray(result["images"])
    assert pixels.shape == (26, 32, 3)
    assert result["weights"] == "ema" and result["image_count"] == 10
    for index in range(10):
        y, x = 2 + index // 5 * 6, 2 + index % 5 * 6
        np.testing.assert_array_equal(pixels[y:y + 4, x:x + 4], index * 10)
        np.testing.assert_array_equal(pixels[y + 12:y + 16, x:x + 4], 63)
    targets = np.arange(10, dtype=np.float64) * 10 / 255
    expected = np.mean(-10 * np.log10((targets - .25) ** 2))
    assert result["psnr_db"] == pytest.approx(expected, abs=1e-5)
    assert np.count_nonzero(pixels[0]) == 0


def test_manifest_and_subset_stable_across_stages_and_resumes(tmp_path):
    root = tmp_path / "val"
    make_images(root, 4)
    config = configuration(root, 3)
    first = FixedPreview(config, tmp_path / "stage1")
    second_config = deepcopy(config)
    second_config["stage"] = 4
    second = FixedPreview(second_config, tmp_path / "stage4")
    resumed = FixedPreview(config, tmp_path / "stage1")
    assert first.paths == second.paths == resumed.paths == ["00.png", "01.png", "02.png"]
    assert first.manifest == second.manifest == resumed.manifest
    assert json.loads((tmp_path / "stage1/preview_manifest.json").read_text()) == first.manifest
    Image.fromarray(np.full((4, 4, 3), 123, dtype=np.uint8)).save(root / "00.png")
    with pytest.raises(ValueError, match="manifest mismatch"):
        FixedPreview(config, tmp_path / "stage1")


@pytest.mark.parametrize("count", [0, -1, True, 2.5])
def test_preview_count_must_be_positive_integer(tmp_path, count):
    with pytest.raises(ValueError, match="positive integer"):
        FixedPreview(configuration(tmp_path / "missing", count), tmp_path / "run")


def test_missing_validation_never_falls_back_to_training_images(tmp_path):
    root = tmp_path / "train"
    make_images(root, 1)
    config = configuration(tmp_path / "missing")
    config["data"]["train_root"] = str(root)
    with pytest.raises(FileNotFoundError):
        FixedPreview(config, tmp_path / "run")
    assert not (tmp_path / "run/preview_manifest.json").exists()


@pytest.mark.parametrize("failure", [False, True])
def test_preview_restores_modes_weights_gradients_optimizers_rng_even_on_error(tmp_path, failure):
    seed_all(45)
    root = tmp_path / "val"
    make_images(root, 2)
    config = configuration(root)
    engine = make_engine(config)
    engine.train_update([torch.rand(2, 3, 4, 4)], 1)
    engine.model.train()
    engine.model.dino_model.eval()
    engine.model.decoder.body.eval()
    engine.model.fail = failure
    before_model = deepcopy(engine.model.state_dict())
    before_ema = deepcopy(engine.ema.state_dict())
    before_optimizer = deepcopy(engine.optimizer_g.state_dict())
    before_gradients = {name: None if p.grad is None else p.grad.clone()
                        for name, p in engine.model.named_parameters()}
    before_modes = [module.training for module in engine.model.modules()]
    before_requires_grad = [p.requires_grad for p in engine.model.parameters()]
    before_progress = dict(engine.progress)
    before_rng = capture_rng_state()
    preview = FixedPreview(config, tmp_path / "run")
    if failure:
        with pytest.raises(RuntimeError, match="deliberate preview failure"):
            preview.render(engine)
    else:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = preview.render(engine)
        assert result["image_count"] == 2
    assert_nested_equal(engine.model.state_dict(), before_model)
    assert_nested_equal(engine.ema.state_dict(), before_ema)
    assert_nested_equal(engine.optimizer_g.state_dict(), before_optimizer)
    assert_nested_equal(capture_rng_state(), before_rng)
    assert [module.training for module in engine.model.modules()] == before_modes
    assert [p.requires_grad for p in engine.model.parameters()] == before_requires_grad
    assert engine.progress == before_progress
    for name, parameter in engine.model.named_parameters():
        assert_nested_equal(parameter.grad, before_gradients[name])
    assert engine.ema._averaging is False


def test_preview_does_not_change_the_next_training_update(tmp_path):
    root = tmp_path / "val"
    make_images(root, 2)
    config = configuration(root)
    first = make_engine(config)
    first.train_update([torch.rand(2, 3, 4, 4)], 1)
    second = deepcopy(first)
    before_rng = capture_rng_state()
    batch = torch.rand(2, 3, 4, 4)
    update_rng = capture_rng_state()
    first.train_update([batch], 1)
    restore_rng_state(update_rng)
    FixedPreview(config, tmp_path / "run").render(second)
    second.train_update([batch], 1)
    assert_nested_equal(first.model.state_dict(), second.model.state_dict())
    assert_nested_equal(first.optimizer_g.state_dict(), second.optimizer_g.state_dict())
    assert_nested_equal(first.ema.state_dict(), second.ema.state_dict())
    restore_rng_state(before_rng)
