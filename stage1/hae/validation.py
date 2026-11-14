"""In-run validation of the EMA weights on a fixed, sorted validation subset.

Same protocol as ``hae.evaluate --weights ema --limit N``: mean per-image PSNR, SSIM and LPIPS-VGG,
float32 without autocast and TF32 off on CUDA. Ranks split the images and sum their results, so every rank
must call ``evaluate`` at the same update. Model modes, weights and RNG streams are restored afterwards, so
validation cannot change the training trajectory.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch

from . import distributed
from .checkpoints import capture_rng_state, restore_rng_state
from .data import ImageDataset
from .metrics import create_lpips_model, psnr_per_image, reconstruct, ssim_per_image

VALIDATION_KEYS = frozenset({'enabled', 'images', 'batch_size', 'lpips', 'ssim'})


def validate_validation_config(section):
    if section is None:
        return
    if not isinstance(section, dict) or set(section) - VALIDATION_KEYS:
        raise ValueError(f'validation must be a mapping with keys from {sorted(VALIDATION_KEYS)}')
    for key in ('enabled', 'lpips', 'ssim'):
        if not isinstance(section.get(key, True), bool):
            raise ValueError(f'validation.{key} must be true or false')
    for key, default in (('images', 1000), ('batch_size', 16)):
        value = section.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'validation.{key} must be a positive integer')


def validation_enabled(config):
    section = config.get('validation')
    return bool(section) and section.get('enabled', True)


@contextmanager
def exact_float32(device):
    """TF32 off for the evaluation, as hae.evaluate does on CUDA; training precision restored after."""
    if device.type != 'cuda':
        yield
        return
    backends = (torch.backends.cuda.matmul, torch.backends.cudnn.conv, torch.backends.cudnn.rnn)
    saved = [backend.fp32_precision for backend in backends]
    try:
        for backend in backends:
            backend.fp32_precision = 'ieee'
        yield
    finally:
        for backend, value in zip(backends, saved):
            backend.fp32_precision = value


class InRunValidation:
    def __init__(self, config, rank, world_size, device, perceptual=None):
        section = config['validation']
        dataset = ImageDataset(config['data']['val_root'], config['data']['image_size'],
                               preprocess=config['data'].get('preprocess', 'squash_bilinear'))
        self.count = min(len(dataset), int(section.get('images', 1000)))
        if self.count < 1:
            raise ValueError(f"Validation root has no images: {config['data']['val_root']}")
        self.dataset = dataset
        self.indices = list(range(rank, self.count, world_size))
        self.batch_size = int(section.get('batch_size', 16))
        self.device = torch.device(device)
        self.ssim = section.get('ssim', True)
        self.lpips = None
        if section.get('lpips', True):
            self.lpips = perceptual if perceptual is not None else create_lpips_model(self.device)

    def evaluate(self, engine):
        """Collective over all ranks. Returns val_psnr_db, val_ssim, val_lpips and val_images."""
        model = engine.model
        flags = [(module, module.training) for module in model.modules()]
        rng_state = capture_rng_state()
        sums, error = None, None
        try:
            if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
                raise ValueError('In-run validation requires float32 master model parameters')
            model.eval()
            psnr = ssim = lpips = 0.0
            with engine.ema.average_parameters(model):
                with exact_float32(self.device), torch.inference_mode(), \
                        torch.autocast(self.device.type, enabled=False):
                    for start in range(0, len(self.indices), self.batch_size):
                        images = torch.stack([self.dataset[index] for index in
                                              self.indices[start:start + self.batch_size]]).to(self.device)
                        output = reconstruct(model, images)
                        psnr += float(psnr_per_image(output, images).double().sum())
                        if self.ssim:
                            ssim += float(ssim_per_image(output, images).double().sum())
                        if self.lpips is not None:
                            lpips += float(self.lpips(output * 2 - 1, images * 2 - 1).double().sum())
            sums = (psnr, ssim, lpips, len(self.indices))
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        finally:
            for module, training in flags:
                module.training = training
            restore_rng_state(rng_state)
        gathered = distributed.gather_objects((sums, error))
        errors = [message for _, message in gathered if message]
        if errors:
            raise RuntimeError(f'In-run validation failed: {errors}')
        psnr, ssim, lpips, count = (sum(values) for values in zip(*(item for item, _ in gathered)))
        if count != self.count:
            raise RuntimeError(f'In-run validation covered {count} images, expected {self.count}')
        result = {'val_psnr_db': psnr / count, 'val_images': int(count)}
        if self.ssim:
            result['val_ssim'] = ssim / count
        if self.lpips is not None:
            result['val_lpips'] = lpips / count
        return result
