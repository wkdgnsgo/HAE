"""--continue-from: keep weights, optimizer, EMA and counters; change only the lr schedule and Adam betas."""
from copy import deepcopy
import json
import shutil

import numpy as np
from PIL import Image
import pytest
import torch

from hae import train
from hae.checkpoints import read_training_checkpoint
from hae.config import (CONTINUE_CHANGEABLE, CONTINUE_CHANGEABLE_GAN, CONTINUE_RELOCATABLE,
                                 assert_resume_compatible, load_config)
from test_training import TinyAutoencoder


@pytest.fixture
def tiny_run(tmp_path, monkeypatch):
    trackers = []

    class Tracker:
        def __init__(self, config, output_dir, *, rank, resume, checkpoint_state):
            self.enabled, self.image_every, self.resume = False, 0, resume
            self.identity = checkpoint_state
            trackers.append(self)

        def log(self, record, preview=None):
            pass

        def state_dict(self):
            return None

        def finish(self, exit_code=0):
            self.exit_code = exit_code

    monkeypatch.setattr(train, 'WandbTracker', Tracker)
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
    config['training'].update(precision='fp32', micro_batch_size=1, global_batch_size=2, learning_rate=1e-4,
                              lr_warmup_updates=0, max_updates=2, save_every=2, log_every=1,
                              output_dir=str(tmp_path / 'first'), init_from=str(initial), init_weights='raw')
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        train.run(config, device='cpu')
        yield tmp_path, config, trackers
    finally:
        torch.set_num_threads(threads)


def test_continue_keeps_optimizer_and_counters_with_new_lr(tiny_run):
    tmp_path, config, trackers = tiny_run
    source = tmp_path / 'first/checkpoint_last.pt'
    first = read_training_checkpoint(source)
    changed = deepcopy(config)
    changed['training'].update(learning_rate=1e-3, betas=[0.9, 0.9], max_updates=4, output_dir=str(tmp_path / 'second'))
    train.run(changed, device='cpu', continue_from=str(source))
    rows = [json.loads(line) for line in (tmp_path / 'second/metrics.jsonl').read_text().splitlines()]
    assert [row['stage_update'] for row in rows] == [3, 4]
    assert all(row['lr_g'] == pytest.approx(1e-3) for row in rows)
    saved = read_training_checkpoint(tmp_path / 'second/checkpoint_last.pt')
    steps = {float(state['step']) for state in saved['optimizer_g']['state'].values()}
    assert steps == {4.0}
    assert [tuple(group['betas']) for group in saved['optimizer_g']['param_groups']] == [(0.9, 0.9)]
    report = json.loads((tmp_path / 'second/initialization.json').read_text())
    assert report['saved_betas'] == list(config['training']['betas']) and report['betas'] == [0.9, 0.9]
    expected_epoch = first['progress']['epoch'] + (1 if first['progress']['batch_in_epoch'] else 0)
    assert report['mode'] == 'continue_from' and report['optimizer_step'] == 2.0
    assert report['data_cursor'] == {'epoch': expected_epoch, 'batch_in_epoch': 0}
    assert report['saved_learning_rate'] == pytest.approx(1e-4) and report['learning_rate'] == pytest.approx(1e-3)
    assert trackers[-1].resume is False and trackers[-1].identity is None


def test_continue_rejects_other_changes_and_existing_output(tiny_run):
    tmp_path, config, _ = tiny_run
    source = str(tmp_path / 'first/checkpoint_last.pt')
    other = deepcopy(config)
    other['training'].update(global_batch_size=4, output_dir=str(tmp_path / 'batch'))
    with pytest.raises(ValueError, match='Resume config differs'):
        train.run(other, device='cpu', continue_from=source)
    same_dir = deepcopy(config)
    same_dir['training']['learning_rate'] = 1e-3
    with pytest.raises(FileExistsError):
        train.run(same_dir, device='cpu', continue_from=source)
    with pytest.raises(ValueError, match='mutually exclusive'):
        train.run(same_dir, device='cpu', continue_from=source, resume=source)


def test_only_the_lr_schedule_and_betas_are_changeable():
    current = load_config('tests/configs/decoder_only.yaml')
    lr = deepcopy(current)
    lr['training'].update(learning_rate=1e-4, lr_warmup_updates=500, betas=[0.9, 0.9])
    with pytest.raises(ValueError):
        assert_resume_compatible(lr, current)
    assert_resume_compatible(lr, current, allow=CONTINUE_CHANGEABLE)
    batch = deepcopy(lr)
    batch['training']['global_batch_size'] = 256
    with pytest.raises(ValueError):
        assert_resume_compatible(batch, current, allow=CONTINUE_CHANGEABLE)


def test_continue_accepts_a_relocated_dataset_but_not_a_changed_one(tiny_run):
    """A checkpoint moved to another server: the folders sit elsewhere, the training manifest is the same."""
    tmp_path, config, _ = tiny_run
    source = str(tmp_path / 'first/checkpoint_last.pt')
    moved = tmp_path / 'other_server/images'
    shutil.copytree(tmp_path / 'images', moved)
    relocated = deepcopy(config)
    relocated['data'].update(train_root=str(moved), val_root=str(moved))
    with pytest.raises(ValueError, match='Resume config differs'):
        train.run(relocated, device='cpu', resume=source)
    relocated['training'].update(max_updates=4, output_dir=str(tmp_path / 'relocated'))
    train.run(relocated, device='cpu', continue_from=source)
    rows = [json.loads(line) for line in (tmp_path / 'relocated/metrics.jsonl').read_text().splitlines()]
    assert [row['stage_update'] for row in rows] == [3, 4]
    changed = tmp_path / 'changed/images'
    shutil.copytree(tmp_path / 'images', changed)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(changed / 'extra.png')
    other = deepcopy(relocated)
    other['data'].update(train_root=str(changed), val_root=str(changed))
    other['training']['output_dir'] = str(tmp_path / 'changed_run')
    with pytest.raises(ValueError, match='Training dataset changed'):
        train.run(other, device='cpu', continue_from=source)


def test_relocation_is_only_for_continue_from():
    current = load_config('configs/stage1_alignment.yaml')
    moved = deepcopy(current)
    moved['data'].update(train_root='/another/server/train', val_root='/another/server/val')
    with pytest.raises(ValueError):
        assert_resume_compatible(moved, current)
    with pytest.raises(ValueError):
        assert_resume_compatible(moved, current, allow=CONTINUE_CHANGEABLE)
    assert_resume_compatible(moved, current, allow=CONTINUE_CHANGEABLE, relocatable=CONTINUE_RELOCATABLE)
    assert train.same_training_data({'root': '/a', 'image_count': 4, 'sha256': 'x'},
                                    {'root': '/b', 'image_count': 4, 'sha256': 'x'}, relocated=True)
    assert not train.same_training_data({'root': '/a', 'image_count': 4, 'sha256': 'x'},
                                        {'root': '/b', 'image_count': 4, 'sha256': 'x'})
    assert not train.same_training_data({'root': '/a', 'image_count': 4, 'sha256': 'x'},
                                        {'root': '/b', 'image_count': 4, 'sha256': 'y'}, relocated=True)


def test_continue_may_lift_the_adaptive_cap_and_nothing_else_of_the_gan():
    saved = load_config('configs/stage2_gan.yaml')
    lifted = load_config('configs/stage2_gan.yaml', ['gan.adaptive_max=100000.0'])
    assert lifted['gan']['adaptive_max'] == 100000.0 and saved['gan']['adaptive_max'] == 10.0
    with pytest.raises(ValueError):
        assert_resume_compatible(lifted, saved)
    with pytest.raises(ValueError):
        assert_resume_compatible(lifted, saved, allow=CONTINUE_CHANGEABLE)
    assert_resume_compatible(lifted, saved, allow=CONTINUE_CHANGEABLE, gan_allow=CONTINUE_CHANGEABLE_GAN)
    weight = deepcopy(lifted)
    weight['gan']['weight'] = 0.75
    with pytest.raises(ValueError):
        assert_resume_compatible(weight, saved, allow=CONTINUE_CHANGEABLE, gan_allow=CONTINUE_CHANGEABLE_GAN)
