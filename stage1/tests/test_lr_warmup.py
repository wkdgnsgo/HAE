"""Generator learning-rate warmup (training.lr_warmup_updates)."""
from types import SimpleNamespace

import pytest

from hae.config import assert_resume_compatible, load_config, validate_config
from hae.training import TrainingEngine


def lr_at(step, base=1e-4, warmup=4):
    fake = SimpleNamespace(config={"training": {"learning_rate": base, "lr_warmup_updates": warmup}})
    return TrainingEngine.generator_lr(fake, step)


def test_linear_warmup_then_constant():
    assert [lr_at(s) for s in range(6)] == pytest.approx([2.5e-5, 5e-5, 7.5e-5, 1e-4, 1e-4, 1e-4])
    assert lr_at(0, warmup=0) == pytest.approx(1e-4)
    fake = SimpleNamespace(config={"training": {"learning_rate": 3e-5}})
    assert TrainingEngine.generator_lr(fake, 0) == pytest.approx(3e-5)


def test_config_validation_and_resume_default():
    config = load_config("tests/configs/decoder_only.yaml")
    assert config["training"]["lr_warmup_updates"] == 500
    validate_config(config, world_size=4)
    with pytest.raises(ValueError, match="lr_warmup_updates"):
        validate_config(load_config("tests/configs/decoder_only.yaml", ["training.lr_warmup_updates=-1"]), world_size=4)
    current = load_config("configs/stage1_alignment.yaml")
    saved = {**current, "training": {k: v for k, v in current["training"].items() if k != "lr_warmup_updates"}}
    assert_resume_compatible(current, saved)
