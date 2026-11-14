"""Optional, rank-zero W&B tracking that never changes training RNG or counters.

The SDK is deliberately imported only after the rank/enabled checks. An online
resume with a known identity uses ``resume='must'``; an old checkpoint without
an identity starts a new run. Offline W&B does not restore an SDK session: it
creates another local segment with the saved ID, to be synced explicitly.
"""
from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from importlib import import_module
import json
import netrc
import os
from pathlib import Path
import tempfile
import uuid
import warnings
from urllib.parse import urlparse

from .checkpoints import capture_rng_state, restore_rng_state


@contextmanager
def preserve_rng():
    """Keep Python, NumPy, torch CPU and this rank's CUDA RNG unchanged."""
    state = capture_rng_state()
    try:
        yield
    finally:
        restore_rng_state(state)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _warn(message):
    try:
        warnings.warn(message, RuntimeWarning, stacklevel=3)
    except Exception:
        pass


def _has_online_credentials():
    """Read login presence without asking the SDK to prompt or verify online."""
    if os.environ.get('WANDB_API_KEY'):
        return True
    token_file = os.environ.get('WANDB_IDENTITY_TOKEN_FILE')
    if token_file and Path(token_file).is_file():
        return True
    hostname = urlparse(os.environ.get('WANDB_BASE_URL', 'https://api.wandb.ai')).hostname
    configured_netrc = os.environ.get('NETRC')
    paths = (configured_netrc,) if configured_netrc else (
        str(Path.home() / '.netrc'), str(Path.home() / '_netrc'))
    for path in paths:
        if not path or not Path(path).is_file():
            continue
        try:
            credentials = netrc.netrc(path).authenticators(hostname)
            if credentials and credentials[-1]:
                return True
        except (OSError, netrc.NetrcParseError):
            continue
    return False


class WandbTracker:
    """Log completed optimizer updates, optionally with an EMA preview grid.

    The run identity comes from the checkpoint, then ``wandb_run.json`` on an exact resume. Authenticate
    with ``wandb login`` or the environment; never put an API key in a config.
    """

    def __init__(self, config, output_dir, *, rank=0, resume=False, checkpoint_state=None):
        options = config.get('logging', {}).get('wandb', {}) or {}
        self.enabled = False
        self.image_every = 500
        self.num_images = 10
        self.output_dir = Path(output_dir)
        self._run = None
        self._sdk = None
        self._identity = None
        self._failed = False
        self._finished = False
        self._last_step = -1
        if rank != 0 or not options.get('enabled', False) or options.get('mode') == 'disabled':
            return
        try:
            with preserve_rng():
                self.image_every = int(options.get('image_every', 500))
                self.num_images = int(options.get('num_images', 10))
                mode = options.get('mode', 'online')
                if mode not in ('online', 'offline'):
                    raise ValueError('W&B mode must be online, offline, or disabled')
                if self.image_every < 0 or self.num_images < 1:
                    raise ValueError('image_every must be nonnegative and num_images positive')
                identity = None
                source = 'new_run'
                if resume:
                    if checkpoint_state is not None:
                        identity = self._validated_identity(checkpoint_state)
                        source = 'checkpoint'
                    elif (self.output_dir / 'wandb_run.json').is_file():
                        identity = self._validated_identity(json.loads(
                            (self.output_dir / 'wandb_run.json').read_text()))
                        source = 'local_file'
                had_identity = identity is not None
                identity = identity or {
                    'format_version': 1,
                    'id': uuid.uuid4().hex[:8],
                    'project': options.get('project', 'hae-stage1'),
                    'entity': options.get('entity'),
                    'group': options.get('group', 'stage1'),
                }
                self._last_step = int(identity.get('last_logged_update', -1))
                self.output_dir.mkdir(parents=True, exist_ok=True)
                if mode == 'online' and not _has_online_credentials():
                    raise RuntimeError('No W&B credentials found. Run wandb login or set WANDB_API_KEY; '
                                       'use logging.wandb.mode=offline to log locally.')
                self._sdk = import_module('wandb')
                sdk_resume = 'must' if had_identity and mode == 'online' else 'never'
                self._run = self._sdk.init(
                    id=identity['id'], project=identity['project'], entity=identity.get('entity'),
                    group=identity.get('group'), name=options.get('name') or
                    f"stage{config.get('stage', '?')}-{self.output_dir.name}",
                    job_type=f"stage{config.get('stage', '?')}",
                    config=deepcopy(config), mode=mode, resume=sdk_resume,
                    dir=str(self.output_dir.resolve()), reinit='create_new', save_code=False,
                    settings=self._sdk.Settings(init_timeout=30, login_timeout=30, disable_git=True),
                )
                if self._run is None:
                    raise RuntimeError('wandb.init returned no run')
                self._identity = {
                    'format_version': 1,
                    'id': str(self._run.id),
                    'project': str(self._run.project or identity['project']),
                    'entity': self._run.entity or identity.get('entity'),
                    'group': self._run.group or identity.get('group'),
                    'mode': mode,
                    'last_logged_update': self._last_step,
                }
                self._run.define_metric('train/update')
                self._run.define_metric('train/*', step_metric='train/update')
                self._run.define_metric('preview/*', step_metric='train/update')
                self._run.define_metric('val/update')
                self._run.define_metric('val/*', step_metric='val/update')
                self._run.define_metric('Reconstructions_EMA', step_metric='train/update')
                _write_json(self.output_dir / 'wandb_run.json', self._identity)
                self.enabled = True
                self._status('active', identity_source=source, sdk_resume=sdk_resume,
                             offline_segment=mode == 'offline' and had_identity,
                             resumed_without_identity=bool(resume and not had_identity))
        except Exception as error:
            self._disable('init', error)

    @staticmethod
    def _validated_identity(value):
        if not isinstance(value, Mapping) or value.get('format_version') != 1:
            raise ValueError('Invalid W&B run identity schema')
        for field in ('id', 'project'):
            if not isinstance(value.get(field), str) or not value[field].strip():
                raise ValueError(f'Invalid W&B run identity {field}')
        for field in ('entity', 'group'):
            if value.get(field) is not None and not isinstance(value[field], str):
                raise ValueError(f'Invalid W&B run identity {field}')
        return dict(value)

    def _status(self, status, **extra):
        try:
            _write_json(self.output_dir / 'wandb_status.json', {
                'status': status, 'run': self.state_dict(), **extra,
            })
        except Exception as error:
            _warn(f'Could not write W&B status file: {type(error).__name__}')

    def _disable(self, operation, error):
        self.enabled = False
        self._failed = True
        message = str(error)
        secret = os.environ.get('WANDB_API_KEY')
        if secret:
            message = message.replace(secret, '[REDACTED]')
        _warn(f'W&B {operation} failed; tracking disabled, training continues: {message}')
        self._status('failed', operation=operation, error=message)
        if self._run is not None and not self._finished:
            try:
                with preserve_rng():
                    self._run.finish(exit_code=1)
            except Exception:
                pass
            self._finished = True

    def state_dict(self):
        """Return only JSON-safe run identity, never the SDK or credentials."""
        return deepcopy(self._identity)

    def log(self, record, preview=None):
        if not self.enabled or self._finished:
            return
        try:
            with preserve_rng():
                step = int(record['stage_update'])
                if step < 0 or step != record['stage_update']:
                    raise ValueError('stage_update must be a nonnegative integer')
                if step <= self._last_step:
                    return
                data = {(f'val/{name[4:]}' if name.startswith('val_') else f'train/{name}'): value
                        for name, value in record.items()}
                data['train/update'] = step
                if preview is not None:
                    if preview.get('weights') != 'ema':
                        raise ValueError('Reconstructions_EMA requires EMA preview weights')
                    count = int(preview['image_count'])
                    paths = [str(path) for path in preview.get('paths', ())]
                    caption = f'EMA, optimizer update {step}; GT first, reconstruction second; {count} images'
                    if paths:
                        caption += '\n' + '\n'.join(paths)
                    data['Reconstructions_EMA'] = self._sdk.Image(preview['images'], caption=caption)
                    data['preview/psnr_db'] = float(preview['psnr_db'])
                    data['preview/count'] = count
                self._run.log(data, step=step, commit=True)
                self._last_step = step
                self._identity['last_logged_update'] = step
                _write_json(self.output_dir / 'wandb_run.json', self._identity)
        except Exception as error:
            self._disable('log', error)

    def finish(self, exit_code=0):
        if self._run is None or self._finished:
            return
        try:
            with preserve_rng():
                self._run.finish(exit_code=exit_code)
            self._finished = True
            self.enabled = False
            self._status('finished', exit_code=int(exit_code))
        except Exception as error:
            self._disable('finish', error)
