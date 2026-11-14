"""losses.alignment_type: cosine (default) or feature MSE for the Stage 1 alignment term."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from hae.config import assert_resume_compatible, load_config, validate_config
from hae.training import TrainingEngine

from test_training import TinyAutoencoder, TinyTeacher, config_for, tensor_snapshot

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("cos_l0p5", "mse_l0p5", "mse_l5", "mse_l50")


def engine_for(alignment_type=None, weight=0.5, seed=0):
    torch.manual_seed(seed)
    model, teacher = TinyAutoencoder(), TinyTeacher()
    config = config_for(stage=1)
    config["losses"]["alignment"] = weight
    if alignment_type is not None:
        config["losses"]["alignment_type"] = alignment_type
    return TrainingEngine(model, config, "cpu", teacher=teacher), teacher


def batches(seed=1):
    generator = torch.Generator().manual_seed(seed)
    return [torch.rand(2, 3, 4, 4, generator=generator) for _ in range(2)]


def test_cosine_training_is_unchanged_by_naming_the_type():
    default, _ = engine_for(None)
    named, _ = engine_for("cos")
    stats_a = default.train_update(iter(batches()), 2)
    stats_b = named.train_update(iter(batches()), 2)
    for (name, a), (_, b) in zip(tensor_snapshot(default.model).items(), tensor_snapshot(named.model).items()):
        assert torch.equal(a, b), name
    assert stats_a["alignment"] == stats_b["alignment"] and stats_a["loss_g"] == stats_b["loss_g"]
    extra = {"alignment_mse", "feature_norm", "teacher_norm"}
    assert not extra & set(stats_a)
    assert extra <= set(stats_b)
    assert set(stats_b) - extra == set(stats_a)


def test_mse_alignment_enters_the_loss_and_cosine_is_still_reported():
    weight = 3.0
    engine, _ = engine_for("mse", weight=weight)
    stats = engine.train_update(iter(batches()), 2)
    assert stats["loss_g"] == pytest.approx(stats["l1"] + weight * stats["alignment_mse"], rel=1e-6)
    assert stats["alignment"] != pytest.approx(stats["alignment_mse"], rel=1e-3)
    cosine, _ = engine_for("cos", weight=weight)
    cosine.train_update(iter(batches()), 2)
    differences = [not torch.equal(a, b) for a, b in zip(tensor_snapshot(engine.model).values(),
                                                         tensor_snapshot(cosine.model).values())]
    assert any(differences)


def test_logged_values_match_their_definitions():
    engine, teacher = engine_for("mse")
    images = batches()[0]
    from hae.training import normalize_imagenet
    with torch.no_grad():
        _, features = engine.model(normalize_imagenet(images))
        target = teacher.forward_features(normalize_imagenet(images))["x_norm_patchtokens"]
    stats = engine.train_update(iter([images]), 1)
    assert stats["alignment_mse"] == pytest.approx(F.mse_loss(features, target).item(), rel=1e-6)
    assert stats["alignment"] == pytest.approx((1 - F.cosine_similarity(features, target, dim=-1)).mean().item(), rel=1e-6)
    assert stats["feature_norm"] == pytest.approx(features.norm(dim=-1).mean().item(), rel=1e-6)
    assert stats["teacher_norm"] == pytest.approx(target.norm(dim=-1).mean().item(), rel=1e-6)


def test_mse_of_equal_norm_features_is_the_scaled_cosine_distance():
    """The scale relation used to choose the MSE weights: MSE = (2 r^2 / D) (1 - cos) at equal norms r."""
    generator = torch.Generator().manual_seed(0)
    radius, dim = 9.9, 1024
    a = F.normalize(torch.randn(64, dim, generator=generator), dim=-1) * radius
    b = F.normalize(a + 0.5 * torch.randn(64, dim, generator=generator), dim=-1) * radius
    cosine_distance = (1 - F.cosine_similarity(a, b, dim=-1)).mean()
    assert F.mse_loss(a, b).item() == pytest.approx((2 * radius ** 2 / dim) * cosine_distance.item(), rel=1e-4)
    assert 2 * radius ** 2 / dim == pytest.approx(0.19, abs=0.005)
