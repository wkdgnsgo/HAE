"""Exercise the actual CLI training orchestration without external tracking."""
from copy import deepcopy
import json
import warnings

import numpy as np
from PIL import Image
import pytest
import torch

from hae import train
from hae.checkpoints import read_training_checkpoint
from hae.config import load_config, validate_config, assert_resume_compatible
from test_training import TinyAutoencoder


def test_training_events_checkpoint_identity_and_resume(tmp_path, monkeypatch):
    torch_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    instances = []

    class Tracker:
        def __init__(self, config, output_dir, *, rank, resume, checkpoint_state):
            self.enabled = rank == 0 and config['logging']['wandb']['enabled']
            self.image_every = 2
            self.identity = checkpoint_state or {'id': 'test-run'}
            self.resume = resume
            self.records = []
            instances.append(self)

        def log(self, record, preview=None):
            self.records.append((deepcopy(record), preview))

        def state_dict(self):
            return self.identity if self.enabled else None

        def finish(self, exit_code=0):
            self.exit_code = exit_code

    class Preview:
        def __init__(self, config, output_dir):
            pass

        def render(self, engine):
            return {'psnr_db': 26.0, 'image_count': 1, 'weights': 'ema'}

    monkeypatch.setattr(train, 'WandbTracker', Tracker)
    monkeypatch.setattr(train, 'FixedPreview', Preview)
    monkeypatch.setattr(train, 'build_autoencoder', lambda *a, **k: TinyAutoencoder())
    root = tmp_path / 'images'
    root.mkdir()
    for index in range(4):
        Image.fromarray(np.full((8, 8, 3), 32 + index * 48, dtype=np.uint8)).save(root / f'{index}.png')
    initial = tmp_path / 'initial.pt'
    torch.save(TinyAutoencoder().state_dict(), initial)
    config = load_config('tests/configs/decoder_only.yaml')
    config['losses']['lpips'] = 0
    config['data'].update(train_root=str(root), val_root=str(root), num_workers=0)
    config['training'].update(precision='fp32', micro_batch_size=1, global_batch_size=2,
                              max_updates=2, save_every=2, log_every=10,
                              output_dir=str(tmp_path / 'run'), init_from=str(initial), init_weights='raw')
    try:
        train.run(config, device='cpu')
        saved = read_training_checkpoint(tmp_path / 'run/checkpoint_last.pt')
        assert saved['tracking'] == {'id': 'test-run'}
        assert saved['progress']['generator_updates'] == 2
        assert saved['progress']['microbatches'] == 4
        assert [record['stage_update'] for record, _ in instances[0].records] == [1, 2]
        assert all(preview['weights'] == 'ema' for _, preview in instances[0].records)
        assert instances[0].exit_code == 0
        config['training']['max_updates'] = 3
        train.run(config, device='cpu', resume=str(tmp_path / 'run/checkpoint_last.pt'))
        assert instances[1].resume
        assert instances[1].identity == {'id': 'test-run'}
        assert [record['stage_update'] for record, _ in instances[1].records] == [3]
        records = [json.loads(line) for line in (tmp_path / 'run/metrics.jsonl').read_text().splitlines()]
        assert [row['stage_update'] for row in records] == [1, 2, 3]
        assert records[-1]['microbatches'] == 6
        assert records[-1]['preview_psnr_db'] == 26
        config['logging']['wandb']['enabled'] = False
        config['training']['max_updates'] = 4
        train.run(config, device='cpu', resume=str(tmp_path / 'run/checkpoint_last.pt'))
        saved = read_training_checkpoint(tmp_path / 'run/checkpoint_last.pt')
        assert saved['tracking'] == {'id': 'test-run'}
    finally:
        torch.set_num_threads(torch_threads)


@pytest.mark.parametrize('key,value', [('mode', 'invalid'), ('image_every', -1),
                                     ('image_every', 1.5), ('num_images', 0), ('project', ''),
                                     ('enabled', 'yes')])
def test_invalid_tracking_config_is_rejected(key, value):
    config = load_config('configs/stage1_alignment.yaml')
    config['logging']['wandb'][key] = value
    with pytest.raises(ValueError, match='logging.wandb'):
        validate_config(config, world_size=4)


def test_old_checkpoint_config_can_enable_tracking_on_resume():
    current = load_config('tests/configs/decoder_only.yaml')
    old = deepcopy(current)
    old.pop('logging')
    assert_resume_compatible(current, old)


def test_preview_diagnostic_cannot_interrupt_training(tmp_path):
    not_a_directory = tmp_path / 'file'
    not_a_directory.write_text('occupied')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        train._preview_failed(not_a_directory, RuntimeError('preview failed'), step=10)
