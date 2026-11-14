"""lr and Adam-betas phases inside one training run (training.phases).

Each phase sets the generator learning rate and Adam betas; the Adam moments carry over between phases.
A fixed phase lasts ``epochs`` epochs. An ``until_plateau`` phase lasts at most ``max_epochs`` and ends at
the first epoch boundary where the in-run validation (the ``validation`` section) shows ``patience``
consecutive epochs whose gains are all below both thresholds (PSNR and LPIPS), or where the train alignment
loss (1 - cos, mean of the recent updates) exceeds ``max_alignment``. Lengths are in epochs and boundaries in
optimizer updates (epochs x updates_per_epoch), so one phase list fits any global batch size.
"""
from __future__ import annotations

from copy import deepcopy
import math

PHASE_KEYS = frozenset({'epochs', 'max_epochs', 'learning_rate', 'betas', 'until_plateau'})
PLATEAU_DEFAULTS = {'patience': 2, 'min_psnr_gain_db': 0.05, 'min_lpips_gain': 0.003, 'max_alignment': None}


def _number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def normalize_phases(phases):
    """Validate training.phases and return copies with until_plateau expanded to a full rule (or None)."""
    if not isinstance(phases, list) or not phases:
        raise ValueError('training.phases must be a nonempty list')
    normalized = []
    for index, phase in enumerate(phases):
        where = f'training.phases[{index}]'
        if not isinstance(phase, dict) or set(phase) - PHASE_KEYS:
            raise ValueError(f'{where} must be a mapping with keys from {sorted(PHASE_KEYS)}')
        rule = phase.get('until_plateau')
        if rule is True:
            rule = {}
        if rule is not None and rule is not False:
            if not isinstance(rule, dict) or set(rule) - set(PLATEAU_DEFAULTS):
                raise ValueError(f'{where}.until_plateau must be true or a mapping with keys from '
                                 f'{sorted(PLATEAU_DEFAULTS)}')
            rule = {**PLATEAU_DEFAULTS, **rule}
            if isinstance(rule['patience'], bool) or not isinstance(rule['patience'], int) or rule['patience'] < 1:
                raise ValueError(f'{where}.until_plateau.patience must be a positive integer')
            for key in ('min_psnr_gain_db', 'min_lpips_gain'):
                if not _number(rule[key]) or rule[key] < 0:
                    raise ValueError(f'{where}.until_plateau.{key} must be a nonnegative number')
            if rule['max_alignment'] is not None and (not _number(rule['max_alignment']) or rule['max_alignment'] <= 0):
                raise ValueError(f'{where}.until_plateau.max_alignment must be positive or null')
        else:
            rule = None
        length_key, other_key = ('max_epochs', 'epochs') if rule else ('epochs', 'max_epochs')
        if other_key in phase:
            raise ValueError(f'{where}: use max_epochs with until_plateau and epochs otherwise')
        length = phase.get(length_key)
        if isinstance(length, bool) or not isinstance(length, int) or length < 1:
            raise ValueError(f'{where}.{length_key} must be a positive integer')
        if not _number(phase.get('learning_rate')) or phase['learning_rate'] <= 0:
            raise ValueError(f'{where}.learning_rate must be a positive number')
        betas = phase.get('betas')
        if (not isinstance(betas, (list, tuple)) or len(betas) != 2
                or not all(_number(beta) and 0 <= beta < 1 for beta in betas)):
            raise ValueError(f'{where}.betas must contain two values in [0,1)')
        normalized.append({'epochs': length, 'learning_rate': float(phase['learning_rate']),
                           'betas': (float(betas[0]), float(betas[1])), 'until_plateau': rule})
    return normalized


class PhaseSchedule:
    """Phase positions in optimizer updates, plateau decisions and their checkpoint state.

    ``update`` arguments of ``index_at``/``at`` are 0-based optimizer updates: update u moves stage_update
    from u to u + 1. ``observe`` takes the completed-update count at which the EMA weights were validated.
    """

    def __init__(self, phases, updates_per_epoch, state=None):
        if isinstance(updates_per_epoch, bool) or not isinstance(updates_per_epoch, int) or updates_per_epoch < 1:
            raise ValueError('An epoch must contain at least one optimizer update')
        self.phases = normalize_phases(phases)
        self.updates_per_epoch = updates_per_epoch
        self.ended_early = {}
        self.end_reasons = {}
        self.evaluations = []
        if state is not None:
            self.load_state_dict(state)

    def bounds(self):
        """[(start, end)] per phase; a running plateau phase counts at its maximum length."""
        out, start = [], 0
        for index, phase in enumerate(self.phases):
            end = self.ended_early.get(index, start + phase['epochs'] * self.updates_per_epoch)
            out.append((start, end))
            start = end
        return out

    def total_updates(self):
        return self.bounds()[-1][1]

    def index_at(self, update):
        for index, (_, end) in enumerate(self.bounds()):
            if update < end:
                return index
        return len(self.phases) - 1

    def at(self, update):
        """(learning rate, betas) of optimizer update ``update``."""
        phase = self.phases[self.index_at(update)]
        return phase['learning_rate'], phase['betas']

    def evaluated_at(self, update):
        return any(record['stage_update'] == update for record in self.evaluations)

    def observe(self, update, metrics, alignment=None):
        """Record a validation after ``update`` completed updates; end a plateau phase when its rule holds.

        Returns the reason when this observation ends the phase that would train the next update.
        """
        record = {'stage_update': int(update), **metrics}
        if alignment is not None:
            record['train_alignment'] = float(alignment)
        self.evaluations = [r for r in self.evaluations if r['stage_update'] != update] + [record]
        self.evaluations.sort(key=lambda r: r['stage_update'])
        index = self.index_at(update)
        rule = self.phases[index]['until_plateau']
        start, end = self.bounds()[index]
        if rule is None or not start < update < end:
            return None
        reason = None
        if rule['max_alignment'] is not None and alignment is not None and alignment > rule['max_alignment']:
            reason = f'train alignment {alignment:.4f} > {rule["max_alignment"]}'
        else:
            history = [r for r in self.evaluations if start <= r['stage_update'] <= update]
            if len(history) > rule['patience']:
                recent = history[-(rule['patience'] + 1):]
                gains = [(b['val_psnr_db'] - a['val_psnr_db'],
                          a['val_lpips'] - b['val_lpips'] if 'val_lpips' in a and 'val_lpips' in b else None)
                         for a, b in zip(recent, recent[1:])]
                if all(psnr < rule['min_psnr_gain_db'] and (lpips is None or lpips < rule['min_lpips_gain'])
                       for psnr, lpips in gains):
                    text = ', '.join(f'{psnr:+.3f} dB' + ('' if lpips is None else f' / LPIPS {-lpips:+.4f}')
                                     for psnr, lpips in gains)
                    reason = f'val plateau over {rule["patience"]} epoch(s): {text}'
        if reason:
            self.ended_early[index] = int(update)
            self.end_reasons[index] = reason
        return reason

    def describe(self, completed=None):
        """JSON-safe view for schedule.json: phase positions, status and the validation history."""
        phases = []
        for index, ((start, end), phase) in enumerate(zip(self.bounds(), self.phases)):
            status = None
            if completed is not None:
                status = 'done' if completed >= end else ('active' if completed >= start else 'pending')
            phases.append({'index': index, 'start_update': start, 'end_update': end,
                           'end_is_maximum': bool(phase['until_plateau']) and index not in self.ended_early,
                           'learning_rate': phase['learning_rate'], 'betas': list(phase['betas']),
                           'until_plateau': deepcopy(phase['until_plateau']), 'status': status,
                           'end_reason': self.end_reasons.get(index)})
        return {'updates_per_epoch': self.updates_per_epoch, 'completed_updates': completed,
                'total_updates': self.total_updates(), 'phases': phases,
                'evaluations': deepcopy(self.evaluations)}

    def state_dict(self):
        return {'format_version': 1, 'updates_per_epoch': self.updates_per_epoch,
                'ended_early': {str(k): v for k, v in self.ended_early.items()},
                'end_reasons': {str(k): v for k, v in self.end_reasons.items()},
                'evaluations': deepcopy(self.evaluations)}

    def load_state_dict(self, state):
        if not isinstance(state, dict) or state.get('format_version') != 1:
            raise ValueError('Unsupported phase schedule state')
        if state['updates_per_epoch'] != self.updates_per_epoch:
            raise ValueError(f"Saved schedule used {state['updates_per_epoch']} updates per epoch, "
                             f'this run has {self.updates_per_epoch}')
        ended = {int(k): int(v) for k, v in state['ended_early'].items()}
        if any(not 0 <= k < len(self.phases) or self.phases[k]['until_plateau'] is None for k in ended):
            raise ValueError('Saved schedule ended a phase that is not an until_plateau phase here')
        self.ended_early = ended
        self.end_reasons = {int(k): v for k, v in state.get('end_reasons', {}).items()}
        self.evaluations = deepcopy(state.get('evaluations', []))


def lr_betas_at(training, update, updates_per_epoch, state=None):
    """(learning rate, betas) of optimizer update ``update`` under a training config, with or without phases."""
    if training.get('phases'):
        return PhaseSchedule(training['phases'], updates_per_epoch, state).at(update)
    base = float(training['learning_rate'])
    warmup = int(training.get('lr_warmup_updates', 0) or 0)
    lr = base if warmup <= 0 else base * min(1.0, (update + 1) / warmup)
    return lr, tuple(float(beta) for beta in training['betas'])
