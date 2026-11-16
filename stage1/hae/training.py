"""One optimizer update over a complete accumulation window."""
from contextlib import contextmanager, nullcontext
import torch
from torch import nn
from torch.nn import functional as F
from . import distributed
from .ema import ModelEMA
from .models import configure_trainable, get_last_layer


def normalize_imagenet(images):
    mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (images - mean) / std


@contextmanager
def frozen_discriminator(discriminator):
    training = discriminator.training
    flags = [p.requires_grad for p in discriminator.parameters()]
    discriminator.eval().requires_grad_(False)
    try:
        yield
    finally:
        discriminator.train(training)
        for p, flag in zip(discriminator.parameters(), flags):
            p.requires_grad_(flag)


def adaptive_weight(reconstruction_loss, adversarial_loss, last_layer, cap):
    rec = torch.autograd.grad(reconstruction_loss, last_layer, retain_graph=True)[0]
    gan = torch.autograd.grad(adversarial_loss, last_layer, retain_graph=True)[0]
    rec_norm, gan_norm = rec.float().norm(), gan.float().norm()
    ratio = torch.where(gan_norm > 1e-6, rec_norm / gan_norm.clamp_min(1e-6), gan_norm.new_zeros(()))
    weight = ratio.clamp(0, cap).detach()
    return weight, rec_norm.detach(), gan_norm.detach(), (ratio >= cap).float().detach()


class TrainingEngine:
    def __init__(self, model, config, device, teacher=None, discriminator=None, perceptual=None):
        self.model, self.config, self.device = model, config, torch.device(device)
        self.teacher, self.discriminator, self.perceptual = teacher, discriminator, perceptual
        configure_trainable(model, config['training']['scope'])
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        groups = self.parameters
        hcpe_scale = config['training'].get('hcpe_lr_scale')
        if hcpe_scale is not None:
            hcpe = [p for name, p in model.named_parameters()
                    if p.requires_grad and name.startswith('dino_model.patch_embed.')]
            hcpe_ids = {id(p) for p in hcpe}
            others = [p for p in self.parameters if id(p) not in hcpe_ids]
            if not hcpe or not others:
                raise ValueError('training.hcpe_lr_scale needs trainable HCPE and decoder parameters')
            groups = [{'params': others, 'lr_scale': 1.0}, {'params': hcpe, 'lr_scale': float(hcpe_scale)}]
        self.optimizer_g = torch.optim.AdamW(groups, lr=config['training']['learning_rate'],
            betas=tuple(config['training']['betas']), weight_decay=config['training']['weight_decay'])
        self._lr_scales = [group.get('lr_scale') for group in self.optimizer_g.param_groups]
        self.optimizer_d = None
        self.discriminator_parameters = []
        if config['gan']['enabled']:
            if discriminator is None:
                raise ValueError('Stage 2 requires a discriminator')
            self.discriminator_parameters = [p for p in discriminator.parameters() if p.requires_grad]
            if not self.discriminator_parameters:
                raise ValueError('The discriminator has no trainable heads')
            self.optimizer_d = torch.optim.AdamW(self.discriminator_parameters, lr=config['gan']['learning_rate'],
                betas=tuple(config['gan']['betas']), weight_decay=0)
        if config['losses']['alignment'] and teacher is None:
            raise ValueError('Alignment requires an original frozen DINO teacher')
        if config['losses']['lpips'] and perceptual is None:
            raise ValueError('LPIPS is enabled but its frozen network was not supplied')
        for network in (teacher, perceptual):
            if network is not None:
                network.eval().requires_grad_(False)
        self.ema = ModelEMA(model, decay=config['ema']['decay'], warmup=config['ema'].get('warmup', False))
        self.progress = {'stage_update': 0, 'generator_updates': 0, 'discriminator_updates': 0,
                         'microbatches': 0, 'samples_seen': 0}
        self.schedule = None

    def restore_lr_scales(self):
        """After optimizer_g.load_state_dict: the multipliers come from this run's config, not the checkpoint
        (they are equal on an exact resume; --continue-from may have changed training.hcpe_lr_scale)."""
        if len(self.optimizer_g.param_groups) != len(self._lr_scales):
            raise ValueError('The checkpoint optimizer has a different parameter-group layout (training.hcpe_lr_scale)')
        for group, scale in zip(self.optimizer_g.param_groups, self._lr_scales):
            if scale is None:
                group.pop('lr_scale', None)
            else:
                group['lr_scale'] = scale

    def generator_lr(self, step):
        """training.phases when set; else linear warmup over training.lr_warmup_updates (0 = constant lr)."""
        schedule = getattr(self, 'schedule', None)
        if schedule is not None:
            return schedule.at(step)[0]
        base = self.config['training']['learning_rate']
        warmup = int(self.config['training'].get('lr_warmup_updates', 0))
        return base if warmup <= 0 else base * min(1.0, (step + 1) / warmup)

    def autocast(self):
        if self.config['training']['precision'] == 'bf16':
            return torch.autocast(self.device.type, dtype=torch.bfloat16)
        return nullcontext()

    def train_update(self, microbatches, accumulation_steps, world_size=1):
        if accumulation_steps < 1:
            raise ValueError('accumulation_steps must be positive')
        cfg = self.config
        self.model.train()
        self.optimizer_g.zero_grad(set_to_none=True)
        if self.optimizer_d:
            self.optimizer_d.zero_grad(set_to_none=True)
            self.discriminator.train()
        adversarial_active = cfg['gan']['enabled'] and self.progress['stage_update'] >= cfg['gan']['warmup_updates']
        stats = {}
        count, local_images = 0, 0
        for images in microbatches:
            if count >= accumulation_steps:
                raise ValueError('More microbatches than the configured accumulation window')
            images = images.to(self.device, non_blocking=True)
            normalized = normalize_imagenet(images)
            with self.autocast():
                predicted, features = self.model(normalized)
                distributed.require_finite(predicted, 'reconstruction output')
                reconstructed = predicted.clamp(0, 1)
                l1 = F.l1_loss(reconstructed.float(), images.float())
                lpips = self.perceptual(reconstructed * 2 - 1, images * 2 - 1).mean() if self.perceptual is not None else l1.new_zeros(())
                reconstruction = cfg['losses']['l1'] * l1 + cfg['losses']['lpips'] * lpips
                alignment = l1.new_zeros(())
                alignment_loss = alignment
                monitor = {}
                if self.teacher is not None:
                    with torch.no_grad():
                        target = self.teacher.forward_features(normalized)['x_norm_patchtokens']
                    alignment = (1 - F.cosine_similarity(features.float(), target.float(), dim=-1)).mean()
                    alignment_loss = alignment
                    if 'alignment_type' in cfg['losses']:
                        alignment_mse = F.mse_loss(features.float(), target.float())
                        monitor = {'alignment_mse': alignment_mse,
                                   'feature_norm': features.detach().float().norm(dim=-1).mean(),
                                   'teacher_norm': target.float().norm(dim=-1).mean()}
                        if cfg['losses']['alignment_type'] == 'mse':
                            alignment_loss = alignment_mse
            values = {'l1': l1, 'lpips': lpips, 'alignment': alignment, **monitor}
            if self.optimizer_d:
                with self.autocast():
                    real_logits = self.discriminator(images * 2 - 1)
                    fake_logits = self.discriminator(reconstructed.detach() * 2 - 1)
                    distributed.require_finite(real_logits, 'real discriminator logits')
                    distributed.require_finite(fake_logits, 'fake discriminator logits')
                    loss_d = .5 * (F.relu(1 - real_logits.float()).mean() + F.relu(1 + fake_logits.float()).mean())
                distributed.require_finite(loss_d, 'discriminator loss')
                (loss_d / accumulation_steps).backward()
                values.update(loss_d=loss_d, logits_real=real_logits.float().mean(), logits_fake=fake_logits.float().mean())
            adversarial, weight = l1.new_zeros(()), l1.new_zeros(())
            if adversarial_active:
                with frozen_discriminator(self.discriminator), self.autocast():
                    adversarial = -self.discriminator(reconstructed * 2 - 1).float().mean()
                    if cfg['gan']['adaptive']:
                        weight, rn, gn, clipped = adaptive_weight(reconstruction, adversarial, get_last_layer(self.model), cfg['gan']['adaptive_max'])
                        values.update(rec_gradient_norm=rn, gan_gradient_norm=gn, adaptive_at_cap=clipped,
                                      weighted_gan_gradient_norm=cfg['gan']['weight'] * weight * gn)
                    else:
                        weight = l1.new_ones(())
                    total = reconstruction + cfg['losses']['alignment'] * alignment_loss + cfg['gan']['weight'] * weight * adversarial
                    distributed.require_finite(total, 'generator loss')
                    (total / accumulation_steps).backward()
            else:
                total = reconstruction + cfg['losses']['alignment'] * alignment_loss
                distributed.require_finite(total, 'generator loss')
                (total / accumulation_steps).backward()
            values.update(loss_g=total, adversarial=adversarial, adaptive_weight=weight)
            for name, value in values.items():
                stats[name] = stats.get(name, 0.0) + float(value.detach()) / accumulation_steps
            count += 1
            local_images += images.shape[0]
        if count != accumulation_steps:
            self.optimizer_g.zero_grad(set_to_none=True)
            if self.optimizer_d:
                self.optimizer_d.zero_grad(set_to_none=True)
            raise ValueError(f'Incomplete accumulation window: {count}/{accumulation_steps}')
        distributed.average_gradients(self.parameters)
        g_norm = nn.utils.clip_grad_norm_(self.parameters, cfg['training']['max_grad_norm'])
        distributed.require_finite(g_norm, 'generator gradient norm')
        if self.optimizer_d:
            distributed.average_gradients(self.discriminator_parameters)
            d_norm = nn.utils.clip_grad_norm_(self.discriminator_parameters, cfg['training']['max_grad_norm'])
            distributed.require_finite(d_norm, 'discriminator gradient norm')
            self.optimizer_d.step()
            self.progress['discriminator_updates'] += 1
            stats['d_gradient_norm'] = float(d_norm)
        lr = self.generator_lr(self.progress['stage_update'])
        betas = self.schedule.at(self.progress['stage_update'])[1] if self.schedule is not None else None
        for group in self.optimizer_g.param_groups:
            group['lr'] = lr * group.get('lr_scale', 1.0)
            if betas is not None:
                group['betas'] = betas
        self.optimizer_g.step()
        self.ema.update(self.model, step=self.progress['stage_update'])
        self.progress['stage_update'] += 1
        self.progress['generator_updates'] += 1
        self.progress['microbatches'] += count
        self.progress['samples_seen'] += local_images * world_size
        stats.update(g_gradient_norm=float(g_norm), gan_active=float(adversarial_active))
        return distributed.mean_metrics(stats, self.device)
