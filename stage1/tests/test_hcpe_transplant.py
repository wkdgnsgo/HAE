"""HCPE transplant: Stage 0 initial weights (model.encoder.hcpe_init) and the HCPE learning-rate multiplier."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile

import pytest
import torch

from hae.config import assert_resume_compatible, load_config, validate_config
from hae.models.generator import CnnPatchEmbed
from hae.siglip2 import Siglip2VisionArch, Siglip2VisionTower, build_autoencoder
from hae.training import TrainingEngine
from test_siglip2 import LARGE_ARCH, TINY_ARCH, engine_config, model_config, write_fake_checkpoint

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs/siglip2"
ARCH_OVERRIDE = ["model.encoder.arch=" + json.dumps(LARGE_ARCH)]


@pytest.fixture
def workspace():
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as directory:
        weights = write_fake_checkpoint(directory, Siglip2VisionTower(Siglip2VisionArch(**TINY_ARCH)))
        hcpe = CnnPatchEmbed(embed_dim=TINY_ARCH["hidden_size"])
        hcpe.reset_parameters()
        init = Path(directory) / "hcpe_init.pt"
        torch.save(hcpe.state_dict(), init)
        yield str(weights), str(init), hcpe.state_dict()


def test_hcpe_init_replaces_only_the_hcpe(workspace):
    weights, init, saved = workspace
    torch.manual_seed(42)
    plain = build_autoencoder(model_config(weights_path=weights), load_pretrained=True)
    torch.manual_seed(42)
    started = build_autoencoder(model_config(weights_path=weights, hcpe_init=init), load_pretrained=True)
    assert isinstance(started.dino_model.patch_embed, CnnPatchEmbed)
    for name, value in saved.items():
        assert torch.equal(started.hcpe.state_dict()[name], value), name
    assert not all(torch.equal(a, b) for a, b in zip(plain.hcpe.state_dict().values(), started.hcpe.state_dict().values()))
    for (name, a), (_, b) in zip(plain.decoder.state_dict().items(), started.decoder.state_dict().items()):
        assert torch.equal(a, b), name
    build_autoencoder(model_config(weights_path=weights, hcpe_init="missing/hcpe.pt"), load_pretrained=False)
    with pytest.raises(FileNotFoundError, match="pretrain_hcpe_patchify"):
        build_autoencoder(model_config(weights_path=weights, hcpe_init="missing/hcpe.pt"), load_pretrained=True)


def make_engine(weights, scale):
    config = engine_config()
    config["model"]["encoder"]["weights_path"] = weights
    config["losses"]["alignment"] = 0.0
    if scale is not None:
        config["training"]["hcpe_lr_scale"] = scale
    torch.manual_seed(7)
    model = build_autoencoder(config["model"], load_pretrained=True)
    return TrainingEngine(model, config, "cpu", teacher=None), config


def batches(seed=3):
    generator = torch.Generator().manual_seed(seed)
    return [torch.rand(2, 3, 64, 64, generator=generator) for _ in range(2)]


def test_without_the_key_the_optimizer_is_unchanged(workspace):
    weights, _, _ = workspace
    engine, _ = make_engine(weights, None)
    assert len(engine.optimizer_g.param_groups) == 1 and "lr_scale" not in engine.optimizer_g.param_groups[0]
    assert len(engine.optimizer_g.param_groups[0]["params"]) == len(engine.parameters)


def test_scale_one_trains_exactly_like_no_scale(workspace):
    weights, _, _ = workspace
    reference, _ = make_engine(weights, None)
    scaled, _ = make_engine(weights, 1.0)
    reference.train_update(iter(batches()), 2)
    scaled.train_update(iter(batches()), 2)
    for (name, a), (_, b) in zip(reference.model.state_dict().items(), scaled.model.state_dict().items()):
        assert torch.equal(a, b), name


def test_hcpe_moves_by_the_scaled_rate(workspace):
    weights, _, _ = workspace
    engine, config = make_engine(weights, 0.1)
    groups = engine.optimizer_g.param_groups
    assert [group["lr_scale"] for group in groups] == [1.0, 0.1]
    hcpe_ids = {id(p) for name, p in engine.model.named_parameters() if name.startswith("dino_model.patch_embed.")}
    assert {id(p) for p in groups[1]["params"]} == hcpe_ids
    assert all(id(p) not in hcpe_ids for p in groups[0]["params"])
    assert len(groups[0]["params"]) + len(groups[1]["params"]) == len(engine.parameters)
    full, _ = make_engine(weights, 1.0)
    before = {name: value.detach().clone() for name, value in engine.model.state_dict().items()}
    engine.train_update(iter(batches()), 2)
    full.train_update(iter(batches()), 2)
    base = config["training"]["learning_rate"]
    assert groups[0]["lr"] == pytest.approx(base) and groups[1]["lr"] == pytest.approx(0.1 * base)
    after, after_full = engine.model.state_dict(), full.model.state_dict()
    for name in before:
        step, step_full = after[name] - before[name], after_full[name] - before[name]
        if name.startswith("dino_model.patch_embed."):
            torch.testing.assert_close(step, 0.1 * step_full, rtol=1e-4, atol=1e-9)
            assert step.abs().max() > 0
        elif name.startswith("decoder."):
            assert torch.equal(after[name], after_full[name]), name


def test_continue_from_may_change_the_scale_and_the_config_value_wins(workspace):
    from hae.config import CONTINUE_CHANGEABLE
    config = load_config(CONFIGS / "stage1_alignment.yaml", ARCH_OVERRIDE)
    lowered = deepcopy(config)
    lowered["training"]["hcpe_lr_scale"] = 0.3
    with pytest.raises(ValueError, match="Resume config differs"):
        assert_resume_compatible(lowered, config)
    assert_resume_compatible(lowered, config, allow=CONTINUE_CHANGEABLE)
    weights, _, _ = workspace
    saved_engine, _ = make_engine(weights, 1.0)
    saved_engine.train_update(iter(batches()), 2)
    engine, _ = make_engine(weights, 0.3)
    engine.optimizer_g.load_state_dict(deepcopy(saved_engine.optimizer_g.state_dict()))
    assert [group["lr_scale"] for group in engine.optimizer_g.param_groups] == [1.0, 1.0]
    engine.restore_lr_scales()
    assert [group["lr_scale"] for group in engine.optimizer_g.param_groups] == [1.0, 0.3]
    single, _ = make_engine(weights, None)
    with pytest.raises(ValueError, match="parameter groups|parameter-group layout"):
        single.optimizer_g.load_state_dict(deepcopy(saved_engine.optimizer_g.state_dict()))
        single.restore_lr_scales()


def test_optimizer_state_with_two_groups_survives_a_reload(workspace):
    weights, _, _ = workspace
    engine, _ = make_engine(weights, 0.1)
    engine.train_update(iter(batches()), 2)
    state = deepcopy(engine.optimizer_g.state_dict())
    assert [group["lr_scale"] for group in state["param_groups"]] == [1.0, 0.1]
    fresh, _ = make_engine(weights, 0.1)
    fresh.model.load_state_dict(engine.model.state_dict())
    fresh.optimizer_g.load_state_dict(state)
    engine.train_update(iter(batches(5)), 2)
    fresh.train_update(iter(batches(5)), 2)
    for (name, a), (_, b) in zip(engine.model.state_dict().items(), fresh.model.state_dict().items()):
        assert torch.equal(a, b), name
