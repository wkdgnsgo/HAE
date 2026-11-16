"""HCPE warmup: fit the HCPE to the encoder's original patch embedding before Stage 1.

Encoders whose frozen blocks amplify small input-embedding errors (SigLIP2, DINOv2) lose the teacher's
features when the HCPE starts from random weights. The warmup trains the HCPE alone to reproduce the output
of the original patch embedding (MSE between the two [B, H, W, C] embeddings). The transformer is not run,
so an epoch costs little more than reading the images.

``training.hcpe_warmup``::

    epochs: 2                # passes over data.train_root; 0 turns the warmup off
    batch_size: 256          # images per update, split evenly over the GPUs
    learning_rate: 1.0e-3    # AdamW, linear warmup then cosine decay to 0
    lr_warmup_updates: 100
"""
from __future__ import annotations

import json
import math
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, DistributedSampler

from . import distributed
from .checkpoints import atomic_save

WARMUP_DEFAULTS = {'epochs': 0, 'batch_size': 256, 'learning_rate': 1e-3, 'lr_warmup_updates': 100}
WEIGHTS_FILE, REPORT_FILE, LOG_FILE = 'hcpe_warmup.pt', 'hcpe_warmup.json', 'hcpe_warmup.jsonl'


def warmup_settings(config):
    return {**WARMUP_DEFAULTS, **(config['training'].get('hcpe_warmup') or {})}


def warmup_enabled(config):
    return warmup_settings(config)['epochs'] > 0


def validate_hcpe_warmup_config(config, world_size=1):
    section = config['training'].get('hcpe_warmup')
    if section is None:
        return
    if not isinstance(section, dict) or set(section) - set(WARMUP_DEFAULTS):
        raise ValueError(f'training.hcpe_warmup must be a mapping with keys from {sorted(WARMUP_DEFAULTS)}')
    settings = warmup_settings(config)
    for key, minimum in (('epochs', 0), ('batch_size', 1), ('lr_warmup_updates', 0)):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'training.hcpe_warmup.{key} must be an integer >= {minimum}')
    rate = settings['learning_rate']
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        raise ValueError('training.hcpe_warmup.learning_rate must be a positive number')
    if settings['epochs'] == 0:
        return
    if config['training']['scope'] != 'cnn_decoder':
        raise ValueError('training.hcpe_warmup needs a trainable HCPE (Stage 1, scope cnn_decoder)')
    if (config['model'].get('encoder') or {}).get('hcpe_init'):
        raise ValueError('Use training.hcpe_warmup or model.encoder.hcpe_init, not both')
    if settings['batch_size'] % world_size:
        raise ValueError('training.hcpe_warmup.batch_size must be a multiple of the GPU count')


def patch_embed_input(tower, normalized):
    """The tensor a tower passes to its patch embedding for ImageNet-normalized images."""
    return tower.to_siglip_input(normalized) if hasattr(tower, 'to_siglip_input') else normalized


def learning_rate_at(update, total, base, warmup):
    if update < warmup:
        return base * (update + 1) / warmup
    return base * 0.5 * (1 + math.cos(math.pi * (update - warmup) / max(total - warmup, 1)))


def _identity(config, dataset, world_size):
    return {'settings': warmup_settings(config), 'seed': config['seed'], 'train_images': len(dataset),
            'world_size': world_size}


def run_hcpe_warmup(model, teacher, config, dataset, context, out_dir, log_every=100):
    """Collective over all ranks: train ``model.dino_model.patch_embed`` in place and return the report.

    The result is saved in ``out_dir``; a restart with the same settings loads it instead of training again.
    """
    from .training import normalize_imagenet

    settings = warmup_settings(config)
    identity = _identity(config, dataset, context.world_size)
    hcpe, original = model.dino_model.patch_embed, teacher.patch_embed
    weights, report_file = out_dir / WEIGHTS_FILE, out_dir / REPORT_FILE
    if weights.is_file() and report_file.is_file():
        report = json.loads(report_file.read_text())
        if all(report.get(key) == value for key, value in identity.items()):
            hcpe.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
            return {**report, 'status': 'loaded'}
    per_rank = settings['batch_size'] // context.world_size
    sampler = DistributedSampler(dataset, num_replicas=context.world_size, rank=context.rank, shuffle=True,
                                 seed=config['seed'], drop_last=True)
    loader = DataLoader(dataset, batch_size=per_rank, sampler=sampler, num_workers=config['data']['num_workers'],
                        drop_last=True, pin_memory=context.device.type == 'cuda')
    total = settings['epochs'] * len(loader)
    if not total:
        raise ValueError('The training set is smaller than one HCPE warmup batch')
    parameters = list(hcpe.parameters())
    flags, mode = [p.requires_grad for p in parameters], hcpe.training
    for parameter in parameters:
        parameter.requires_grad_(True)
    hcpe.train()
    optimizer = torch.optim.AdamW(parameters, lr=settings['learning_rate'], betas=(0.9, 0.999), weight_decay=0.0)
    started, update, running, window, last = time.monotonic(), 0, 0.0, 0, None
    try:
        for epoch in range(settings['epochs']):
            sampler.set_epoch(epoch)
            for images in loader:
                source = patch_embed_input(teacher, normalize_imagenet(images.to(context.device, non_blocking=True)))
                with torch.no_grad():
                    target = original(source).float()
                output = hcpe(source).float()
                if output.shape != target.shape:
                    raise ValueError(f'HCPE output {tuple(output.shape)} != patch embedding {tuple(target.shape)}')
                loss = F.mse_loss(output, target)
                distributed.require_finite(loss, 'HCPE warmup loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                distributed.average_gradients(parameters)
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                rate = learning_rate_at(update, total, settings['learning_rate'], settings['lr_warmup_updates'])
                for group in optimizer.param_groups:
                    group['lr'] = rate
                optimizer.step()
                update += 1
                running += float(loss.detach() / target.pow(2).mean().clamp_min(1e-12))
                window += 1
                if update % log_every == 0 or update == total:
                    stats = distributed.mean_metrics({'relative_mse': running / window}, context.device)
                    last = {'hcpe_warmup_update': update, 'total_updates': total, 'epoch': epoch,
                            'relative_mse': stats['relative_mse'], 'learning_rate': rate,
                            'elapsed_seconds': time.monotonic() - started}
                    running, window = 0.0, 0
                    if context.rank == 0:
                        with (out_dir / LOG_FILE).open('a') as handle:
                            handle.write(json.dumps(last) + '\n')
                        print(json.dumps(last), flush=True)
    finally:
        for parameter, flag in zip(parameters, flags):
            parameter.requires_grad_(flag)
        hcpe.train(mode)
    distributed.broadcast_module(hcpe)
    report = {**identity, 'updates': total, 'final_relative_mse': last['relative_mse'],
              'elapsed_seconds': time.monotonic() - started, 'status': 'trained'}
    if context.rank == 0:
        atomic_save({name: value.detach().cpu() for name, value in hcpe.state_dict().items()}, weights)
        report_file.write_text(json.dumps(report, indent=2))
    distributed.barrier()
    return report
