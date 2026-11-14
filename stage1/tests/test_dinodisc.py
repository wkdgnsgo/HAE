"""DinoDisc structure, weight loading and gradient checks on tiny backbones."""
import copy
import io
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import pytest
import torch

from hae.models import DinoDisc, build_discriminator
from hae.models.dinodisc import FrozenDINONoDrop, load_pretrained_backbone

ROOT = Path(__file__).resolve().parents[1]
TINY = dict(depth=12, key_depths=(2, 5, 8, 11), norm_eps=1e-6,
            patch_size=56, in_chans=3, embed_dim=8, num_heads=2, mlp_ratio=2)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_disc():
    return DinoDisc(backbone=FrozenDINONoDrop(**TINY))


def factory_config(path="missing.pt"):
    return {"discriminator": {"type": "dinodisc", "backbone": "dino_vits8",
                              "recipe": "S_8", "weights_path": str(path), "ks": 3}}


def test_full_state_portable_without_external_weights(tmp_path):
    torch.manual_seed(1)
    model = tiny_disc().train()
    model(torch.randn(2, 3, 15, 19)).square().mean().backward()
    model.eval()
    path = tmp_path / "discriminator.pt"
    torch.save(model.state_dict(), path)
    saved = torch.load(path, weights_only=True)
    assert "backbone.blocks.11.attn.qkv.weight" in saved
    assert "backbone.x_scale" in saved
    assert "heads.0.0.0.weight_u" in saved
    torch.manual_seed(999)
    restored = tiny_disc().eval()
    restored.load_state_dict(saved, strict=True)
    image = torch.randn(1, 3, 25, 27)
    torch.testing.assert_close(model(image), restored(image), rtol=0, atol=0)


def test_registered_backbone_moves_with_parent_and_stays_eval():
    model = tiny_disc()
    model.to(dtype=torch.float64).train()
    assert model.backbone.patch_embed.proj.weight.dtype == torch.float64
    assert model.backbone.x_scale.dtype == torch.float64
    assert not model.backbone.training
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(head.training for head in model.heads)
    model.backbone.train()
    assert not model.backbone.training


def test_pretrained_plain_state_strict_and_no_network(tmp_path):
    original = FrozenDINONoDrop(**TINY)
    state = {k: v for k, v in original.state_dict().items() if k not in {"x_scale", "x_shift"}}
    path = tmp_path / "dino_vits8.pt"
    torch.save(state, path)
    restored = FrozenDINONoDrop(**TINY)
    with mock.patch("torch.hub.load", side_effect=AssertionError("Network forbidden")):
        load_pretrained_backbone(restored, path)
    for key, value in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    for failure in ("missing", "unexpected", "shape", "wrapper"):
        bad = dict(state)
        if failure == "missing":
            del bad["blocks.7.attn.qkv.weight"]
        elif failure == "unexpected":
            bad["head.weight"] = torch.zeros(1)
        elif failure == "shape":
            bad["blocks.7.attn.qkv.weight"] = torch.zeros(1)
        else:
            bad = {"teacher": state}
        torch.save(bad, path)
        before = copy.deepcopy(restored.state_dict())
        with pytest.raises(ValueError, match="mismatch|plain tensor"):
            load_pretrained_backbone(restored, path)
        for key, value in before.items():
            torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)


def test_factory_missing_fails_before_model_or_network_construction():
    with mock.patch("hae.models.dinodisc.DinoDisc") as constructor, \
         mock.patch("torch.hub.load", side_effect=AssertionError("Network forbidden")):
        with pytest.raises(FileNotFoundError, match="DINOv1"):
            build_discriminator(factory_config())
        constructor.assert_not_called()


@pytest.mark.parametrize("field,value", [("type", "patchgan"), ("backbone", "dino_vitb8"),
                                          ("recipe", "S_16"), ("ks", 1)])
def test_factory_rejects_other_recipes(field, value):
    config = factory_config()
    config["discriminator"][field] = value
    with pytest.raises(ValueError):
        build_discriminator(config, load_pretrained=False)


def test_factory_pretrained_and_explicit_full_resume_only(tmp_path):
    path = tmp_path / "pretrained.pt"
    source = tiny_disc()
    torch.save(source.backbone.state_dict(), path)
    with mock.patch("hae.models.dinodisc.DinoDisc", side_effect=lambda **kw: tiny_disc()), \
         mock.patch("torch.hub.load", side_effect=AssertionError("Network forbidden")):
        built = build_discriminator(factory_config(path))
        for key, value in source.backbone.state_dict().items():
            torch.testing.assert_close(built.backbone.state_dict()[key], value, rtol=0, atol=0)
        resumed = build_discriminator(factory_config(), load_pretrained=False)
        with pytest.raises(RuntimeError, match="backbone"):
            resumed.load_state_dict({k: v for k, v in source.state_dict().items() if k.startswith("heads.")}, strict=True)
        resumed.load_state_dict(source.state_dict(), strict=True)


def test_disabled_crop_and_gradient_checkpoint_fail_explicitly():
    with pytest.raises(ValueError, match="crop_prob"):
        FrozenDINONoDrop(**TINY, crop_prob=0.5)
    model = tiny_disc()
    with pytest.raises(RuntimeError, match="checkpointing"):
        model(torch.randn(1, 3, 10, 10, requires_grad=True), grad_ckpt=True)


def test_s8_production_parameter_counts_and_feature_shapes_meta():
    with torch.device("meta"):
        model = build_discriminator(factory_config(), load_pretrained=False)
        output = model(torch.empty(1, 3, 256, 256))
    assert output.shape == (1, 5 * 28 * 28)
    assert sum(p.numel() for p in model.backbone.parameters()) == 21_670_272
    assert sum(p.numel() for p in model.heads.parameters()) == 2_962_565
    assert len(model.heads) == 5
