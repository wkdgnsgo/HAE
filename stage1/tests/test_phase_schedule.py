"""training.phases: lr / Adam-betas phases inside one run, with in-run validation and plateau decisions."""
from copy import deepcopy
import json
import shutil

import numpy as np
from PIL import Image
import pytest
import torch

from hae import train
from hae.checkpoints import read_training_checkpoint
from hae.config import assert_resume_compatible, load_config, validate_config
from hae.schedule import PhaseSchedule, lr_betas_at, normalize_phases
from test_training import TinyAutoencoder

RULE = {'patience': 2, 'min_psnr_gain_db': 0.05, 'min_lpips_gain': 0.003, 'max_alignment': 0.1}
PHASES = [{'epochs': 2, 'learning_rate': 1e-3, 'betas': [0.9, 0.999]},
          {'max_epochs': 5, 'learning_rate': 1e-2, 'betas': [0.9, 0.9], 'until_plateau': RULE},
          {'epochs': 1, 'learning_rate': 1e-4, 'betas': [0.9, 0.9]}]


def val(psnr, lpips):
    return {'val_psnr_db': psnr, 'val_lpips': lpips}


def test_phase_bounds_lr_and_betas():
    schedule = PhaseSchedule(PHASES, 5)
    assert schedule.bounds() == [(0, 10), (10, 35), (35, 40)] and schedule.total_updates() == 40
    assert schedule.at(9) == (1e-3, (0.9, 0.999))
    assert schedule.at(10) == (1e-2, (0.9, 0.9)) and schedule.at(34) == (1e-2, (0.9, 0.9))
    assert schedule.at(35) == (1e-4, (0.9, 0.9)) and schedule.at(39) == (1e-4, (0.9, 0.9))


def test_plateau_needs_patience_consecutive_flat_epochs():
    schedule = PhaseSchedule(PHASES, 5)
    assert schedule.observe(10, val(20.0, 0.300)) is None
    assert schedule.observe(15, val(20.5, 0.290)) is None
    assert schedule.observe(20, val(20.52, 0.289)) is None
    reason = schedule.observe(25, val(20.54, 0.288))
    assert 'plateau' in reason
    assert schedule.bounds() == [(0, 10), (10, 25), (25, 30)]
    assert schedule.at(24) == (1e-2, (0.9, 0.9)) and schedule.at(25) == (1e-4, (0.9, 0.9))
    assert schedule.observe(30, val(20.6, 0.287)) is None


def test_lpips_gain_or_alignment_decide_too():
    schedule = PhaseSchedule(PHASES, 5)
    for update, psnr, lpips in ((10, 20.0, 0.30), (15, 20.01, 0.295), (20, 20.02, 0.290)):
        assert schedule.observe(update, val(psnr, lpips)) is None
    assert schedule.total_updates() == 40
    assert 'alignment' in schedule.observe(25, val(20.5, 0.2), alignment=0.2)
    assert schedule.bounds()[1] == (10, 25)
    capped = PhaseSchedule(PHASES, 5)
    assert capped.observe(10, val(20.0, 0.3), alignment=0.5) is None


def test_state_round_trip_and_epoch_length_check():
    schedule = PhaseSchedule(PHASES, 5)
    for update, psnr in ((10, 20.0), (15, 20.01), (20, 20.02)):
        schedule.observe(update, val(psnr, 0.3))
    state = json.loads(json.dumps(schedule.state_dict()))
    restored = PhaseSchedule(PHASES, 5, state)
    assert restored.bounds() == schedule.bounds() == [(0, 10), (10, 20), (20, 25)]
    assert restored.evaluated_at(15) and not restored.evaluated_at(16)
    with pytest.raises(ValueError, match='updates per epoch'):
        PhaseSchedule(PHASES, 6, state)


def test_phase_validation():
    assert normalize_phases([{'max_epochs': 2, 'learning_rate': 1e-4, 'betas': [0.9, 0.9],
                              'until_plateau': True}])[0]['until_plateau'] == {
        'patience': 2, 'min_psnr_gain_db': 0.05, 'min_lpips_gain': 0.003, 'max_alignment': None}
    for bad in ([], [{'epochs': 0, 'learning_rate': 1e-4, 'betas': [0.9, 0.9]}],
                [{'epochs': 1, 'learning_rate': '1e-4', 'betas': [0.9, 0.9]}],
                [{'epochs': 1, 'learning_rate': 1e-4, 'betas': [0.9, 1.0]}],
                [{'epochs': 1, 'learning_rate': 1e-4, 'betas': [0.9, 0.9], 'until_plateau': True}],
                [{'max_epochs': 1, 'learning_rate': 1e-4, 'betas': [0.9, 0.9]}],
                [{'epochs': 1, 'learning_rate': 1e-4, 'betas': [0.9, 0.9], 'warmup': 5}]):
        with pytest.raises(ValueError):
            normalize_phases(bad)


def test_lr_betas_without_phases_match_the_engine_formula():
    training = {'learning_rate': 1e-4, 'lr_warmup_updates': 4, 'betas': [0.9, 0.999]}
    assert lr_betas_at(training, 0, 10) == (pytest.approx(2.5e-5), (0.9, 0.999))
    assert lr_betas_at(training, 7, 10)[0] == pytest.approx(1e-4)


@pytest.fixture
def tiny_run(tmp_path, monkeypatch):
    class Tracker:
        def __init__(self, config, output_dir, *, rank, resume, checkpoint_state):
            self.enabled, self.image_every = False, 0

        def log(self, record, preview=None):
            pass

        def state_dict(self):
            return None

        def finish(self, exit_code=0):
            pass

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
                              betas=[0.9, 0.999], lr_warmup_updates=0, max_updates=2, save_every=2, log_every=1,
                              output_dir=str(tmp_path / 'first'), init_from=str(initial), init_weights='raw')
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        train.run(config, device='cpu')
        yield tmp_path, config
    finally:
        torch.set_num_threads(threads)


def phased(config, *, first_lr=1e-4):
    out = deepcopy(config)
    out['training'].update(max_updates=None, phases=[
        {'epochs': 1, 'learning_rate': first_lr, 'betas': [0.9, 0.999]},
        {'max_epochs': 3, 'learning_rate': 1e-3, 'betas': [0.9, 0.9],
         'until_plateau': {'patience': 1, 'min_psnr_gain_db': 100.0, 'min_lpips_gain': 0.0}},
        {'epochs': 1, 'learning_rate': 1e-5, 'betas': [0.9, 0.9]}])
    out['validation'] = {'enabled': True, 'images': 4, 'batch_size': 2, 'lpips': False}
    return out


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_exact_resume_switches_phases_inside_the_run(tiny_run):
    tmp_path, config = tiny_run
    run_dir = tmp_path / 'first'
    shutil.copytree(run_dir, tmp_path / 'snapshot')
    train.run(phased(config), device='cpu', resume=str(run_dir / 'checkpoint_last.pt'))
    records = {row['stage_update']: row for row in rows(run_dir / 'metrics.jsonl')}
    assert [records[u]['lr_g'] for u in (3, 4, 5, 6)] == pytest.approx([1e-3, 1e-3, 1e-5, 1e-5])
    assert [records[u]['phase'] for u in (3, 4, 5, 6)] == [1, 1, 2, 2]
    assert records[3]['val_update'] == 2 and records[4]['val_update'] == 4 and records[6]['val_update'] == 6
    validations = rows(run_dir / 'validation.jsonl')
    assert [v['stage_update'] for v in validations] == [2, 4, 6] and all(v['val_images'] == 4 for v in validations)
    assert 'plateau' in validations[1]['phase_end'] and 'val_ssim' in validations[0]
    saved = read_training_checkpoint(run_dir / 'checkpoint_last.pt')
    assert saved['progress']['stage_update'] == 6
    assert saved['schedule']['ended_early'] == {'1': 4}
    assert {float(state['step']) for state in saved['optimizer_g']['state'].values()} == {6.0}
    assert [tuple(group['betas']) for group in saved['optimizer_g']['param_groups']] == [(0.9, 0.9)]
    plan = json.loads((run_dir / 'schedule.json').read_text())
    assert [(p['start_update'], p['end_update'], p['status']) for p in plan['phases']] == [
        (0, 2, 'done'), (2, 4, 'done'), (4, 6, 'done')]
    copy = phased(config)
    copy['training']['output_dir'] = str(tmp_path / 'copy')
    train.run(copy, device='cpu', resume=str(run_dir / 'checkpoint_0000004.pt'))
    assert [row['lr_g'] for row in rows(tmp_path / 'copy/metrics.jsonl')] == pytest.approx([1e-5, 1e-5])
    assert [v['stage_update'] for v in rows(tmp_path / 'copy/validation.jsonl')] == [6]
    plain = phased(config)
    plain['training']['phases'][1] = {'epochs': 1, 'learning_rate': 1e-3, 'betas': [0.9, 0.9]}
    plain['training']['output_dir'] = str(tmp_path / 'snapshot')
    plain.pop('validation')
    train.run(plain, device='cpu', resume=str(tmp_path / 'snapshot/checkpoint_last.pt'))
    other = read_training_checkpoint(tmp_path / 'snapshot/checkpoint_last.pt')
    for name, tensor in saved['model'].items():
        assert torch.equal(tensor, other['model'][name]), name


def test_exact_resume_rejects_a_different_past_schedule(tiny_run):
    tmp_path, config = tiny_run
    with pytest.raises(ValueError, match='Resume schedule differs at update 1'):
        train.run(phased(config, first_lr=2e-4), device='cpu', resume=str(tmp_path / 'first/checkpoint_last.pt'))
    other = deepcopy(config)
    other['training']['global_batch_size'] = 4
    other = phased(other)
    with pytest.raises(ValueError, match='Resume config differs'):
        train.run(other, device='cpu', resume=str(tmp_path / 'first/checkpoint_last.pt'))
