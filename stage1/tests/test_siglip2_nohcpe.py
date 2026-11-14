"""SigLIP2 without HCPE (model.encoder.hcpe_mode: none): the frozen original encoder, decoder-only training."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile

import pytest
import torch

from hae.config import assert_resume_compatible, load_config, validate_config
from hae.models.generator import CnnPatchEmbed
from hae.siglip2 import Siglip2VisionArch, Siglip2VisionTower, build_autoencoder, build_teacher
from hae.siglip2.vision import PatchifyEmbed
from hae.training import TrainingEngine, normalize_imagenet
from test_siglip2 import LARGE_ARCH, TINY_ARCH, engine_config, model_config, write_fake_checkpoint

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs/siglip2"
ARCH_OVERRIDE = ["model.encoder.arch=" + json.dumps(LARGE_ARCH)]


@pytest.fixture
def pretrained_path():
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as directory:
        yield write_fake_checkpoint(directory, Siglip2VisionTower(Siglip2VisionArch(**TINY_ARCH)))


def nohcpe_engine_config(pretrained_path):
    config = engine_config(scope="decoder")
    config["model"]["encoder"].update(weights_path=str(pretrained_path), hcpe_mode="none")
    config["losses"]["alignment"] = 0.0
    return config


def test_encoder_is_the_original_tower(pretrained_path):
    config = model_config(weights_path=str(pretrained_path), hcpe_mode="none")
    model = build_autoencoder(config, load_pretrained=True).eval()
    teacher = build_teacher(config)
    assert isinstance(model.dino_model.patch_embed, PatchifyEmbed)
    assert model.hcpe is None
    assert not any(isinstance(module, CnnPatchEmbed) for module in model.modules())
    images = normalize_imagenet(torch.rand(2, 3, 64, 64))
    with torch.no_grad():
        tokens = model.encode(images)
        target = teacher.forward_features(images)["x_norm_patchtokens"]
    torch.testing.assert_close(tokens, target, rtol=0, atol=0)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable and all(name.startswith("decoder.") for name in trainable)
    assert model.encoder_frozen and model.trainable_scope == "decoder"
    assert not model.encode(images).requires_grad


def test_decoder_initialisation_equals_the_hcpe_runs(pretrained_path):
    def decoder_state(mode):
        torch.manual_seed(42)
        model = build_autoencoder(model_config(weights_path=str(pretrained_path), hcpe_mode=mode), load_pretrained=True)
        return {name: value.clone() for name, value in model.decoder.state_dict().items()}

    reference = decoder_state("replace")
    for mode in ("none", "replace"):
        state = decoder_state(mode)
        assert state.keys() == reference.keys()
        for name in reference:
            assert torch.equal(state[name], reference[name]), (mode, name)


def test_one_update_moves_only_the_decoder(pretrained_path):
    config = nohcpe_engine_config(pretrained_path)
    model = build_autoencoder(config["model"], load_pretrained=True)
    engine = TrainingEngine(model, config, "cpu", teacher=None)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    stats = engine.train_update(iter([torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64)]), accumulation_steps=2)
    assert stats["alignment"] == 0.0
    assert stats["loss_g"] == pytest.approx(stats["l1"], rel=1e-6)
    after = model.state_dict()
    changed = {name for name in before if not torch.equal(before[name], after[name])}
    assert changed and all(name.startswith("decoder.") for name in changed)
    assert all(parameter.grad is None for name, parameter in model.named_parameters()
               if name.startswith("dino_model."))
