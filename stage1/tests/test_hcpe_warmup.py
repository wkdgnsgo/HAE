"""training.hcpe_warmup: the HCPE is fitted to the original patch embedding before Stage 1."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from hae import distributed, train
from hae.checkpoints import read_training_checkpoint
from hae.config import assert_resume_compatible, load_config, validate_config
from hae.data import ImageDataset
from hae.encoders import build_autoencoder, build_teacher
from hae.hcpe_warmup import learning_rate_at, patch_embed_input, run_hcpe_warmup, warmup_enabled
from hae.siglip2.vision import Siglip2VisionArch, Siglip2VisionTower
from hae.training import normalize_imagenet
from test_siglip2 import TINY_ARCH, tiny_decoder, write_fake_checkpoint

CONFIGS = Path(__file__).resolve().parents[1] / 'configs'
ARCH = {**TINY_ARCH, 'image_size': 256}


@pytest.fixture
def workspace(tmp_path):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    images = tmp_path / 'images'
    images.mkdir()
    generator = np.random.default_rng(0)
    for index in range(8):
        Image.fromarray(generator.integers(0, 256, (16, 16, 3), dtype=np.uint8)).save(images / f'{index}.png')
    weights = write_fake_checkpoint(tmp_path, Siglip2VisionTower(Siglip2VisionArch(**ARCH)))
    config = load_config(CONFIGS / 'siglip2/stage1_alignment.yaml')
    config['model']['encoder'].update(weights_path=str(weights), arch=dict(ARCH))
    config['model']['decoder'] = tiny_decoder()
    config['data'].update(train_root=str(images), val_root=str(images), num_workers=0)
    config['losses']['lpips'] = 0
    config['validation']['enabled'] = False
    config['logging']['wandb']['enabled'] = False
    config['training'].update(precision='fp32', micro_batch_size=2, global_batch_size=4, save_every=2, log_every=1,
                              output_dir=str(tmp_path / 'run'),
                              phases=[{'epochs': 1, 'learning_rate': 1e-4, 'betas': [0.9, 0.999]}],
                              hcpe_warmup={'epochs': 40, 'batch_size': 4, 'learning_rate': 1e-3,
                                           'lr_warmup_updates': 5})
    try:
        yield tmp_path, config
    finally:
        torch.set_num_threads(threads)


def relative_mse(model, teacher, dataset):
    images = normalize_imagenet(torch.stack([dataset[index] for index in range(len(dataset))]))
    source = patch_embed_input(teacher, images)
    with torch.no_grad():
        target = teacher.patch_embed(source)
        return float((model.dino_model.patch_embed(source) - target).pow(2).mean() / target.pow(2).mean())


def test_warmup_fits_the_patch_embedding_and_touches_only_the_hcpe(workspace):
    tmp_path, config = workspace
    out_dir = tmp_path / 'warmup'
    out_dir.mkdir()
    dataset = ImageDataset(config['data']['train_root'], 256)
    torch.manual_seed(1)
    model, teacher = build_autoencoder(config['model']), build_teacher(config['model'])
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    teacher_before = {name: value.detach().clone() for name, value in teacher.state_dict().items()}
    flags = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    start = relative_mse(model, teacher, dataset)
    report = run_hcpe_warmup(model, teacher, config, dataset, distributed.Context(), out_dir, log_every=20)
    assert report['status'] == 'trained' and report['updates'] == 80
    assert relative_mse(model, teacher, dataset) < 0.25 * start
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name]) != name.startswith('dino_model.patch_embed.'), name
    for name, value in teacher.state_dict().items():
        assert torch.equal(value, teacher_before[name])
    assert flags == {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    rows = [json.loads(line) for line in (out_dir / 'hcpe_warmup.jsonl').read_text().splitlines()]
    assert [row['hcpe_warmup_update'] for row in rows] == [20, 40, 60, 80]
    assert rows[-1]['relative_mse'] < rows[0]['relative_mse'] and rows[-1]['learning_rate'] < 1e-5

    torch.manual_seed(2)
    fresh = build_autoencoder(config['model'])
    again = run_hcpe_warmup(fresh, teacher, config, dataset, distributed.Context(), out_dir)
    assert again['status'] == 'loaded'
    for name, value in fresh.dino_model.patch_embed.state_dict().items():
        assert torch.equal(value, model.dino_model.patch_embed.state_dict()[name])
    shorter = deepcopy(config)
    shorter['training']['hcpe_warmup']['epochs'] = 1
    assert run_hcpe_warmup(fresh, teacher, shorter, dataset, distributed.Context(), out_dir)['status'] == 'trained'


def test_learning_rate_schedule():
    assert learning_rate_at(0, 100, 1e-3, 10) == pytest.approx(1e-4)
    assert learning_rate_at(9, 100, 1e-3, 10) == pytest.approx(1e-3)
    assert learning_rate_at(10, 100, 1e-3, 10) == pytest.approx(1e-3)
    assert learning_rate_at(55, 100, 1e-3, 10) == pytest.approx(5e-4)
    assert learning_rate_at(99, 100, 1e-3, 10) < 1e-6
    assert learning_rate_at(0, 100, 1e-3, 0) == pytest.approx(1e-3)


def test_config_rules(workspace):
    _, config = workspace
    assert warmup_enabled(config) and validate_config(deepcopy(config), 2) == 1
    assert not warmup_enabled(load_config(CONFIGS / 'stage1_alignment.yaml'))
    assert not warmup_enabled(load_config(CONFIGS / 'siglip2/stage2_gan.yaml'))
    assert load_config(CONFIGS / 'stage1_alignment.yaml', ['training.hcpe_warmup.epochs=2'])['training'][
        'hcpe_warmup']['epochs'] == 2
    for key, value, message in (('epochs', -1, 'epochs'), ('epochs', 1.5, 'epochs'), ('batch_size', 0, 'batch_size'),
                                ('learning_rate', 0, 'learning_rate'), ('lr_warmup_updates', -1, 'lr_warmup_updates'),
                                ('steps', 10, 'keys')):
        broken = deepcopy(config)
        broken['training']['hcpe_warmup'][key] = value
        with pytest.raises(ValueError, match=message):
            validate_config(broken, 1)
    with pytest.raises(ValueError, match='multiple of the GPU count'):
        validate_config(deepcopy(config), 3)
    both = deepcopy(config)
    both['model']['encoder']['hcpe_init'] = 'weights/hcpe.pt'
    with pytest.raises(ValueError, match='not both'):
        validate_config(both, 1)
    frozen = load_config(CONFIGS / 'siglip2/stage1_no_hcpe.yaml',
                         ['model.encoder.arch=' + json.dumps(ARCH), 'training.hcpe_warmup.epochs=2'])
    frozen['model']['decoder'] = tiny_decoder()
    with pytest.raises(ValueError, match='trainable HCPE'):
        validate_config(frozen, 1)
    without = deepcopy(config)
    del without['training']['hcpe_warmup']
    validate_config(deepcopy(without), 1)
    assert_resume_compatible(config, without)


def test_train_run_warms_up_once_then_trains(workspace, monkeypatch):
    tmp_path, config = workspace
    config['training']['hcpe_warmup'].update(epochs=3)
    calls = []
    original = train.run_hcpe_warmup
    monkeypatch.setattr(train, 'run_hcpe_warmup', lambda *a, **k: calls.append(1) or original(*a, **k))
    train.run(config, device='cpu')
    out_dir = tmp_path / 'run'
    assert calls == [1] and json.loads((out_dir / 'hcpe_warmup.json').read_text())['updates'] == 6
    saved = read_training_checkpoint(out_dir / 'checkpoint_last.pt')
    assert saved['progress']['stage_update'] == 2
    rows = [json.loads(line) for line in (out_dir / 'metrics.jsonl').read_text().splitlines()]
    assert rows[0]['stage_update'] == 1 and rows[0]['alignment'] > 0
    extended = deepcopy(config)
    extended['training']['phases'] = [{'epochs': 2, 'learning_rate': 1e-4, 'betas': [0.9, 0.999]}]
    train.run(extended, device='cpu', resume=str(out_dir / 'checkpoint_last.pt'))
    assert calls == [1]
    assert read_training_checkpoint(out_dir / 'checkpoint_last.pt')['progress']['stage_update'] == 4
