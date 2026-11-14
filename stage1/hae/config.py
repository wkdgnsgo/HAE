"""Config loading with inheritance and overrides, validation and batch arithmetic."""
from copy import deepcopy
from pathlib import Path
import yaml


def _merge(base, update):
    out = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_config(path, overrides=(), _seen=None):
    path = Path(path).resolve()
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError(f"Circular config inheritance: {path}")
    seen.add(path)
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    parents = config.pop('_base_', None) or []
    base = {}
    for parent in [parents] if isinstance(parents, str) else parents:
        base = _merge(base, load_config(path.parent / parent, _seen=seen))
    config = _merge(base, config)
    for override in overrides or ():
        if '=' not in override:
            raise ValueError(f"Override must be key=value: {override}")
        key, value = override.split('=', 1)
        target = config
        parts = key.split('.')
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise ValueError(f"Unknown config field: {key}")
            target = target[part]
        if parts[-1] not in target:
            raise ValueError(f"Unknown config field: {key}")
        target[parts[-1]] = yaml.safe_load(value)
    return config


def accumulation_steps(config, world_size):
    micro = int(config['training']['micro_batch_size'])
    total = int(config['training']['global_batch_size'])
    if micro <= 0 or total <= 0 or world_size <= 0 or total % (micro * world_size):
        raise ValueError('global_batch_size must be a positive multiple of micro_batch_size x world_size')
    return total // (micro * world_size)


def validate_config(config, world_size=1):
    stage = config.get('stage')
    if stage not in (1, 2, 3, 4):
        raise ValueError('stage must be 1, 2, 3, or 4')
    expected_scope = {1: 'cnn_decoder', 2: 'decoder', 3: 'decoder', 4: 'head'}[stage]
    encoder_section = config['model'].get('encoder')
    no_hcpe = (isinstance(encoder_section, dict) and encoder_section.get('type') == 'siglip2'
               and encoder_section.get('hcpe_mode') == 'none')
    if no_hcpe and stage == 1:
        expected_scope = 'decoder'
    if config['training']['scope'] != expected_scope:
        raise ValueError(f'Stage {stage} requires scope={expected_scope}')
    if bool(config['gan']['enabled']) != (stage == 2):
        raise ValueError('GAN must be enabled only in stage 2')
    if config['losses']['alignment'] != 0 and stage != 1:
        raise ValueError('Feature alignment belongs to stage 1 only')
    if no_hcpe and config['losses']['alignment'] != 0:
        raise ValueError('hcpe_mode none keeps the original encoder: losses.alignment must be 0')
    if stage == 1 and not no_hcpe and config['losses']['alignment'] <= 0:
        raise ValueError('Stage 1 requires positive alignment loss')
    if config['losses'].get('alignment_type', 'cos') not in ALIGNMENT_TYPES:
        raise ValueError(f'losses.alignment_type must be one of {ALIGNMENT_TYPES}')
    hcpe_scale = config['training'].get('hcpe_lr_scale')
    if hcpe_scale is not None:
        if isinstance(hcpe_scale, bool) or not isinstance(hcpe_scale, (int, float)) or not 0 < hcpe_scale <= 1:
            raise ValueError('training.hcpe_lr_scale must be a number in (0, 1]')
        if config['training']['scope'] != 'cnn_decoder':
            raise ValueError('training.hcpe_lr_scale applies only while the HCPE trains (scope cnn_decoder)')
    from .hcpe_warmup import validate_hcpe_warmup_config
    validate_hcpe_warmup_config(config, world_size)
    if config['training']['precision'] not in ('fp32', 'bf16'):
        raise ValueError('Supported precision: fp32 or bf16')
    from .validation import validate_validation_config, validation_enabled
    validate_validation_config(config.get('validation'))
    phases = config['training'].get('phases')
    if phases is not None:
        from .schedule import normalize_phases
        normalized = normalize_phases(phases)
        if config['training'].get('max_updates') is not None:
            raise ValueError('training.max_updates must be null when training.phases sets the end')
        if config['training'].get('lr_warmup_updates', 0):
            raise ValueError('training.lr_warmup_updates must be 0 with training.phases')
        if any(phase['until_plateau'] for phase in normalized) and not validation_enabled(config):
            raise ValueError('until_plateau phases need the in-run validation (validation.enabled)')
    elif config['training'].get('max_updates') is None or int(config['training']['max_updates']) <= 0:
        raise ValueError('max_updates must be positive')
    if int(config['training']['save_every']) <= 0 or int(config['training']['log_every']) <= 0:
        raise ValueError('save_every and log_every must be positive')
    if config['training']['learning_rate'] <= 0:
        raise ValueError('learning_rate must be positive')
    lr_warmup = config['training'].get('lr_warmup_updates', 0)
    if isinstance(lr_warmup, bool) or not isinstance(lr_warmup, int) or lr_warmup < 0:
        raise ValueError('training.lr_warmup_updates must be a non-negative integer')
    if config['training']['max_grad_norm'] <= 0:
        raise ValueError('max_grad_norm must be positive')
    if len(config['training']['betas']) != 2 or not all(0 <= b < 1 for b in config['training']['betas']):
        raise ValueError('Adam betas must contain two values in [0,1)')
    if any(config['losses'][key] < 0 for key in ('l1', 'lpips', 'alignment')):
        raise ValueError('Loss weights cannot be negative')
    if config['losses']['l1'] + config['losses']['lpips'] <= 0:
        raise ValueError('At least one reconstruction loss is required')
    if config['data']['image_size'] != 256:
        raise ValueError('data.image_size must be 256')
    from .data import PREPROCESS_DESCRIPTIONS
    if config['data'].get('preprocess', 'squash_bilinear') not in PREPROCESS_DESCRIPTIONS:
        raise ValueError(f'data.preprocess must be one of {sorted(PREPROCESS_DESCRIPTIONS)}')
    if not 0 <= config['ema']['decay'] < 1:
        raise ValueError('EMA decay must be in [0,1)')
    if not isinstance(config['ema'].get('warmup', False), bool):
        raise ValueError('ema.warmup must be true or false')
    if config['gan']['warmup_updates'] < 0 or config['gan']['weight'] < 0:
        raise ValueError('GAN warmup/weight cannot be negative')
    if config['gan']['enabled'] and config['gan']['weight'] == 0:
        raise ValueError('Stage 2 requires a positive GAN weight')
    if config['gan']['learning_rate'] <= 0 or config['gan']['adaptive_max'] <= 0:
        raise ValueError('GAN learning rate and adaptive cap must be positive')
    if config['gan']['enabled']:
        discriminator = config['gan'].get('discriminator', {})
        if (discriminator.get('type') != 'dinodisc' or discriminator.get('backbone') != 'dino_vits8'
                or discriminator.get('recipe') != 'S_8'):
            raise ValueError('Stage 2 requires gan.discriminator: type dinodisc, backbone dino_vits8, recipe S_8')
        ks = discriminator.get('ks')
        if isinstance(ks, bool) or not isinstance(ks, int) or ks != 3:
            raise ValueError('gan.discriminator.ks must be 3')
        if not isinstance(discriminator.get('weights_path'), str) or not discriminator['weights_path'].strip():
            raise ValueError('gan.discriminator.weights_path must name local DINOv1 pretrained weights')
    tracking = config.get('logging', {}).get('wandb', {})
    if tracking.get('mode', 'online') not in ('online', 'offline', 'disabled'):
        raise ValueError('logging.wandb.mode must be online, offline, or disabled')
    if not isinstance(tracking.get('enabled', False), bool):
        raise ValueError('logging.wandb.enabled must be true or false')
    if tracking.get('enabled'):
        if not isinstance(tracking.get('project'), str) or not tracking['project'].strip():
            raise ValueError('logging.wandb.project must be a nonempty string')
        for key, minimum in (('image_every', 0), ('num_images', 1)):
            value = tracking.get(key, {'image_every': 500, 'num_images': 10}[key])
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f'logging.wandb.{key} must be an integer >= {minimum}')
    encoder = config['model'].get('encoder') or {}
    if not isinstance(encoder, dict) or encoder.get('type', 'dinov3') not in ('dinov3', 'siglip2'):
        raise ValueError("model.encoder.type must be 'dinov3' or 'siglip2'")
    if encoder.get('type', 'dinov3') == 'siglip2':
        from .siglip2.factory import validate_model_config
        validate_model_config(config['model'], image_size=int(config['data']['image_size']))
    return accumulation_steps(config, world_size)


ALIGNMENT_TYPES = ('cos', 'mse')
SCHEDULE_KEYS = ('learning_rate', 'lr_warmup_updates', 'betas', 'phases')
CONTINUE_CHANGEABLE = (*SCHEDULE_KEYS, 'hcpe_lr_scale')
CONTINUE_CHANGEABLE_GAN = ('adaptive_max',)
CONTINUE_RELOCATABLE = ('train_root', 'val_root')


def assert_resume_compatible(current, saved, allow=(), relocatable=(), gan_allow=()):
    """Refuse to continue a checkpoint under a different recipe.

    allow / gan_allow: training / gan keys that may differ (``--continue-from`` only).
    relocatable: data folder keys that may differ (``--continue-from`` only).
    With training.phases the lr and betas are compared by train.run: an exact resume must give the last
    completed update the lr and betas it was trained with.
    """
    a, b = deepcopy(current), deepcopy(saved)
    if a.get('training', {}).get('phases') or b.get('training', {}).get('phases'):
        allow = (*allow, *SCHEDULE_KEYS)
    for item in (a, b):
        for key in ('max_updates', 'save_every', 'log_every', 'output_dir', 'init_from', 'init_weights',
                    'hcpe_warmup', *allow):
            item.get('training', {}).pop(key, None)
        item.get('data', {}).pop('num_workers', None)
        for key in relocatable:
            item.get('data', {}).pop(key, None)
        item.pop('validation', None)
        item.setdefault('data', {}).setdefault('preprocess', 'squash_bilinear')
        item.setdefault('training', {}).setdefault('lr_warmup_updates', 0)
        item.setdefault('losses', {}).setdefault('alignment_type', 'cos')
        encoder = item.get('model', {}).get('encoder')
        if isinstance(encoder, dict) and encoder.get('type') == 'siglip2':
            encoder.setdefault('hcpe_mode', 'replace')
        item.pop('logging', None)
        for key in gan_allow:
            item.get('gan', {}).pop(key, None)
        if item.get('gan', {}).get('enabled') is False:
            item['gan'] = {'enabled': False}
    if a != b:
        raise ValueError('Resume config differs from saved training semantics. Use --init-from for a new stage/recipe.')
