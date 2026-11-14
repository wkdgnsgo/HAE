"""Train one stage; torchrun supplies the multi-GPU rank information."""
import argparse
from collections import deque
from copy import deepcopy
import json
import math
from pathlib import Path
import time
import warnings

import torch
from torch.utils.data import DataLoader, DistributedSampler
from . import distributed
from .checkpoints import (atomic_save, capture_rng_state, load_model_weights,
                          read_training_checkpoint, restore_rng_state, seed_all)
from .config import (CONTINUE_CHANGEABLE, CONTINUE_CHANGEABLE_GAN, CONTINUE_RELOCATABLE, assert_resume_compatible,
                     load_config, validate_config)
from .data import ImageDataset, dataset_fingerprint
from .encoders import build_autoencoder, build_teacher
from .hcpe_warmup import run_hcpe_warmup, warmup_enabled
from .models import build_discriminator
from .schedule import PhaseSchedule, lr_betas_at
from .training import TrainingEngine
from .tracking import WandbTracker
from .preview import FixedPreview
from .validation import InRunValidation, validation_enabled

ALIGNMENT_WINDOW = 200


class BatchStream:
    """Deterministic next-batch cursor; accumulation can cross epoch boundaries."""
    def __init__(self, loader, sampler, epoch=0, batch_in_epoch=0):
        if not len(loader):
            raise ValueError('Dataset is smaller than one per-rank microbatch')
        if epoch < 0 or not 0 <= batch_in_epoch <= len(loader):
            raise ValueError('Invalid saved data cursor')
        self.loader, self.sampler = loader, sampler
        self.epoch, self.batch_in_epoch = epoch, batch_in_epoch
        self._reset()
        for _ in range(batch_in_epoch):
            next(self.iterator)

    def _reset(self):
        self.sampler.set_epoch(self.epoch)
        self.iterator = iter(self.loader)

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.epoch += 1
            self.batch_in_epoch = 0
            self._reset()
            batch = next(self.iterator)
        self.batch_in_epoch += 1
        return batch

    def state_dict(self):
        return {'epoch': self.epoch, 'batch_in_epoch': self.batch_in_epoch}


def build_perceptual(config, device):
    if not config['losses']['lpips']:
        return None
    weights = Path(torch.hub.get_dir()) / 'checkpoints' / 'vgg16-397923af.pth'
    if not weights.is_file():
        raise FileNotFoundError(f'LPIPS VGG weights missing: {weights}. Run scripts/prepare_perceptual.py explicitly first.')
    import lpips
    return lpips.LPIPS(net='vgg').to(device).eval().requires_grad_(False)


def _preview_failed(out_dir, error, step=None):
    """An optional preview diagnostic must not interrupt optimizer progress."""
    try:
        warnings.warn(f'W&B scalar tracking continues without validation previews: {error}')
    except Exception:
        pass
    try:
        (out_dir / 'preview_status.json').write_text(json.dumps({
            'status': 'DISABLED', 'at_update': step, 'error': str(error)}, indent=2))
    except Exception:
        pass


def _checkpoint(engine, config, stream, context, dataset_info, out_dir, tracking=None):
    rng_states = distributed.gather_objects(capture_rng_state())
    error = None
    if context.rank == 0:
        try:
            payload = {
                'format_version': 1,
                'model': engine.model.state_dict(), 'ema': engine.ema.state_dict(),
                'optimizer_g': engine.optimizer_g.state_dict(),
                'optimizer_d': engine.optimizer_d.state_dict() if engine.optimizer_d else None,
                'discriminator': engine.discriminator.state_dict() if engine.discriminator is not None else None,
                'progress': {**engine.progress, **stream.state_dict(), 'world_size': context.world_size},
                'config': deepcopy(config), 'rng_states': rng_states, 'dataset': dataset_info,
                'tracking': tracking,
                'schedule': engine.schedule.state_dict() if engine.schedule is not None else None,
            }
            step = engine.progress['stage_update']
            target = out_dir / f'checkpoint_{step:07d}.pt'
            atomic_save(payload, target)
            latest = out_dir / 'checkpoint_last.pt'
            temporary = out_dir / '.checkpoint_last.tmp'
            temporary.unlink(missing_ok=True)
            temporary.symlink_to(target.name)
            temporary.replace(latest)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
    errors = distributed.gather_objects(error)
    if any(errors):
        raise RuntimeError(f'Checkpoint save failed: {errors}')


def same_training_data(saved, current, *, relocated=False):
    """Exact resume: the same folder and manifest. ``relocated`` (--continue-from): the folder may differ,
    the manifest (image count, relative paths and sizes) may not.
    """
    if saved == current:
        return True
    if not relocated or not isinstance(saved, dict) or not isinstance(current, dict):
        return False
    return ({key: value for key, value in saved.items() if key != 'root'}
            == {key: value for key, value in current.items() if key != 'root'})


def run(config, *, device='auto', init_from=None, resume=None, continue_from=None):
    """resume: exact continuation. continue_from: same recipe with a new lr schedule, Adam betas and/or GPU
    count; restores weights, optimizer, EMA and update counters, starts the next epoch with fresh RNG streams."""
    context = distributed.initialize(device)
    tracker = None
    exit_code = 1
    try:
        accum = validate_config(config, context.world_size)
        if context.device.type == 'cuda' and config['training']['precision'] == 'bf16' and not torch.cuda.is_bf16_supported():
            raise ValueError('Selected CUDA GPU does not support BF16')
        if sum(bool(source) for source in (resume, init_from, continue_from)) > 1:
            raise ValueError('--resume, --init-from and --continue-from are mutually exclusive')
        init_from = None if (resume or continue_from) else (init_from or config['training'].get('init_from'))
        if config['stage'] > 1 and not (resume or init_from or continue_from):
            raise ValueError('Stages 2-4 require an existing initial checkpoint')
        if init_from and not Path(init_from).exists():
            raise FileNotFoundError(f'Initial checkpoint does not exist: {init_from}')
        source = resume or continue_from
        saved = read_training_checkpoint(source) if source else None
        if saved:
            assert_resume_compatible(config, saved['config'], allow=CONTINUE_CHANGEABLE if continue_from else (),
                                     relocatable=CONTINUE_RELOCATABLE if continue_from else (),
                                     gan_allow=CONTINUE_CHANGEABLE_GAN if continue_from else ())
            if resume and saved['progress']['world_size'] != context.world_size:
                raise ValueError('Exact resume requires the same world size; use --continue-from to change it')
            if resume and len(saved['rng_states']) != context.world_size:
                raise ValueError('Missing per-rank RNG states in checkpoint')
        dataset = ImageDataset(config['data']['train_root'], config['data']['image_size'],
                               preprocess=config['data'].get('preprocess', 'squash_bilinear'))
        dataset_info = dataset_fingerprint(config['data']['train_root'])
        if saved and not same_training_data(saved.get('dataset'), dataset_info, relocated=bool(continue_from)):
            raise ValueError('Training dataset changed since saved checkpoint')
        updates_per_epoch = len(dataset) // context.world_size // config['training']['micro_batch_size'] // accum
        schedule = None
        if config['training'].get('phases'):
            schedule = PhaseSchedule(config['training']['phases'], updates_per_epoch,
                                     state=saved.get('schedule') if resume else None)
        if resume and (schedule is not None or saved['config']['training'].get('phases')):
            last = saved['progress']['stage_update'] - 1
            if last >= 0:
                new = schedule.at(last) if schedule else lr_betas_at(config['training'], last, updates_per_epoch)
                old = lr_betas_at(saved['config']['training'], last, updates_per_epoch, saved.get('schedule'))
                if not math.isclose(new[0], old[0], rel_tol=1e-9) or tuple(new[1]) != tuple(old[1]):
                    raise ValueError(f'Resume schedule differs at update {last}: trained with lr {old[0]:g}, betas '
                                     f'{tuple(old[1])}; this config gives lr {new[0]:g}, betas {tuple(new[1])}. '
                                     'Use --continue-from for a new lr schedule.')
        out_dir = Path(config['training']['output_dir'])
        if not resume and (out_dir / 'checkpoint_last.pt').exists():
            raise FileExistsError(f'{out_dir} already contains a training run. Use --resume or a new output_dir.')
        if context.rank == 0:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / 'resolved_config.json').write_text(json.dumps(config, indent=2))
        distributed.barrier()
        seed_all(config['seed'])
        model = build_autoencoder(config['model'], load_pretrained=not bool(saved or init_from)).to(context.device)
        if saved:
            model.load_state_dict(saved['model'], strict=True)
        elif init_from:
            report = load_model_weights(init_from, model, weights=config['training']['init_weights'])
            if context.rank == 0:
                (out_dir / 'initialization.json').write_text(json.dumps(report, indent=2))
        discriminator = (build_discriminator(config['gan'], load_pretrained=not bool(saved)).to(context.device)
                         if config['gan']['enabled'] else None)
        teacher = build_teacher(config['model']).to(context.device) if config['losses']['alignment'] else None
        perceptual = build_perceptual(config, context.device)
        distributed.broadcast_module(model)
        if discriminator is not None:
            distributed.broadcast_module(discriminator)
        if teacher is not None:
            distributed.broadcast_module(teacher)
        if warmup_enabled(config) and not (saved or init_from):
            report = run_hcpe_warmup(model, teacher, config, dataset, context, out_dir)
            if context.rank == 0:
                print(json.dumps({'hcpe_warmup': report}), flush=True)
        engine = TrainingEngine(model, config, context.device, teacher=teacher, discriminator=discriminator, perceptual=perceptual)
        engine.schedule = schedule
        first_update = saved['progress']['stage_update'] if saved else 0
        first_lr, first_betas = (schedule.at(first_update) if schedule
                                 else (config['training']['learning_rate'], tuple(config['training']['betas'])))
        if saved:
            engine.optimizer_g.load_state_dict(saved['optimizer_g'])
            engine.restore_lr_scales()
            if continue_from:
                for group in engine.optimizer_g.param_groups:
                    group['betas'] = tuple(first_betas)
            engine.ema.load_state_dict(saved['ema'])
            if engine.optimizer_d:
                if saved['optimizer_d'] is None or saved['discriminator'] is None:
                    raise ValueError('GAN resume requires discriminator and optimizer states')
                engine.discriminator.load_state_dict(saved['discriminator'], strict=True)
                engine.optimizer_d.load_state_dict(saved['optimizer_d'])
            elif saved['discriminator'] is not None or saved['optimizer_d'] is not None:
                raise ValueError('Non-GAN phase cannot resume a GAN optimizer')
            for name in engine.progress:
                engine.progress[name] = saved['progress'][name]
        seed_all(config['seed'] + context.rank)
        sampler = DistributedSampler(dataset, num_replicas=context.world_size, rank=context.rank,
                                     shuffle=True, seed=config['seed'], drop_last=True)
        loader = DataLoader(dataset, batch_size=config['training']['micro_batch_size'], sampler=sampler,
                            num_workers=config['data']['num_workers'], drop_last=True,
                            pin_memory=context.device.type == 'cuda',
                            generator=torch.Generator().manual_seed(config['seed'] + context.rank))
        progress = saved['progress'] if saved else {}
        if continue_from:
            cursor = (progress['epoch'] + (1 if progress['batch_in_epoch'] else 0), 0)
        else:
            cursor = (progress.get('epoch', 0), progress.get('batch_in_epoch', 0))
        stream = BatchStream(loader, sampler, *cursor)
        tracking_state = saved.get('tracking') if resume else None
        if continue_from and context.rank == 0:
            state = saved['optimizer_g']['state']
            (out_dir / 'initialization.json').write_text(json.dumps({
                'mode': 'continue_from', 'path': str(Path(continue_from).resolve()),
                'saved_progress': saved['progress'],
                'optimizer_step': float(next(iter(state.values()))['step']) if state else 0.0,
                'saved_learning_rate': saved['config']['training']['learning_rate'],
                'learning_rate': first_lr,
                'saved_betas': list(saved['config']['training']['betas']),
                'betas': list(engine.optimizer_g.param_groups[0]['betas']),
                'lr_warmup_updates': config['training'].get('lr_warmup_updates', 0),
                'phases': deepcopy(config['training'].get('phases')),
                'world_size': context.world_size,
                'data_cursor': {'epoch': cursor[0], 'batch_in_epoch': cursor[1]}}, indent=2))
        if len(loader) // accum != updates_per_epoch:
            raise RuntimeError(f'Loader gives {len(loader) // accum} updates per epoch, expected {updates_per_epoch}')
        validator = None
        if validation_enabled(config):
            rng_state = capture_rng_state()
            validator = InRunValidation(config, context.rank, context.world_size, context.device,
                                        perceptual=perceptual)
            restore_rng_state(rng_state)
        if resume:
            restore_rng_state(saved['rng_states'][context.rank])
        del saved
        tracking_config = deepcopy(config)
        tracking_config['runtime'] = {
            'world_size': context.world_size, 'accumulation_steps': accum,
            'trainable_parameters': sum(p.numel() for p in engine.parameters),
            'starting_update': engine.progress['stage_update'],
        }
        tracker = WandbTracker(tracking_config, out_dir, rank=context.rank,
                               resume=bool(resume), checkpoint_state=tracking_state)
        preview = None
        if tracker.enabled and tracker.image_every > 0:
            try:
                preview = FixedPreview(config, out_dir)
            except Exception as exc:
                _preview_failed(out_dir, exc)
        distributed.barrier()
        start_update = engine.progress['stage_update']
        if context.rank == 0:
            print(json.dumps({'stage': config['stage'], 'world_size': context.world_size,
                'micro_batch_size': config['training']['micro_batch_size'], 'accumulation_steps': accum,
                'global_batch_size': config['training']['global_batch_size'],
                'trainable_parameters': sum(p.numel() for p in engine.parameters),
                'starting_update': start_update, 'updates_per_epoch': updates_per_epoch,
                'phases': schedule.describe(start_update)['phases'] if schedule else None,
                'validation_images': validator.count if validator else None}, indent=2), flush=True)
            if schedule:
                (out_dir / 'schedule.json').write_text(json.dumps(schedule.describe(start_update), indent=2))

        def planned_end():
            return schedule.total_updates() if schedule else config['training']['max_updates']

        alignment_window = deque(maxlen=ALIGNMENT_WINDOW)

        def validate(step):
            """Collective. Returns the fields for the next logged record and the phase-end reason (or None)."""
            metrics = validator.evaluate(engine)
            alignment = sum(alignment_window) / len(alignment_window) if alignment_window else None
            reason = schedule.observe(step, metrics, alignment=alignment) if schedule else None
            if context.rank == 0:
                entry = {'stage_update': step, **metrics, 'train_alignment': alignment, 'phase_end': reason}
                with (out_dir / 'validation.jsonl').open('a') as handle:
                    handle.write(json.dumps(entry) + '\n')
                if schedule:
                    (out_dir / 'schedule.json').write_text(json.dumps(schedule.describe(step), indent=2))
                print(json.dumps({'validation': entry}), flush=True)
            return {**metrics, 'val_update': step}, reason

        validation_fields = None
        if (validator is not None and 0 < start_update < planned_end() and start_update % updates_per_epoch == 0
                and not (schedule and schedule.evaluated_at(start_update))):
            validation_fields, _ = validate(start_update)
        started = time.monotonic()
        last_saved = engine.progress['stage_update'] if (resume or continue_from) else -1
        current_phase = schedule.index_at(start_update) if schedule else None
        while engine.progress['stage_update'] < planned_end():
            stats = engine.train_update((next(stream) for _ in range(accum)), accum, context.world_size)
            step = engine.progress['stage_update']
            alignment_window.append(stats.get('alignment', 0.0))
            phase = schedule.index_at(step - 1) if schedule else None
            if phase != current_phase and context.rank == 0:
                print(json.dumps({'phase_start': phase, 'first_update': step - 1,
                                  'learning_rate': engine.optimizer_g.param_groups[0]['lr'],
                                  'betas': list(engine.optimizer_g.param_groups[0]['betas'])}), flush=True)
            current_phase = phase
            if validator is not None and (step % updates_per_epoch == 0 or step == planned_end()):
                validation_fields, reason = validate(step)
                if reason and context.rank == 0:
                    print(json.dumps({'phase_end': phase, 'at_update': step, 'reason': reason,
                                      'total_updates': planned_end()}), flush=True)
            record = {**stats, **engine.progress, **stream.state_dict(),
                      'elapsed_seconds': time.monotonic() - started,
                      'lr_g': engine.optimizer_g.param_groups[0]['lr']}
            if schedule:
                record['phase'] = phase
                record['beta2_g'] = engine.optimizer_g.param_groups[0]['betas'][1]
            if len(engine.optimizer_g.param_groups) > 1:
                record['lr_hcpe'] = engine.optimizer_g.param_groups[1]['lr']
            if engine.optimizer_d:
                record['lr_d'] = engine.optimizer_d.param_groups[0]['lr']
            final_update = step == planned_end()
            scalar_due = (step == 1 or final_update or step % config['training']['log_every'] == 0
                          or validation_fields is not None)
            image_due = (tracker.enabled and preview is not None
                         and (step == 1 or final_update or step % tracker.image_every == 0))
            if validation_fields is not None:
                record.update(validation_fields)
                validation_fields = None
            if context.rank == 0 and (scalar_due or image_due):
                preview_result = None
                if image_due:
                    try:
                        preview_result = preview.render(engine)
                        record['preview_psnr_db'] = preview_result['psnr_db']
                        record['preview_image_count'] = preview_result['image_count']
                    except Exception as exc:
                        _preview_failed(out_dir, exc, step)
                        preview = None
                with (out_dir / 'metrics.jsonl').open('a') as handle:
                    handle.write(json.dumps(record) + '\n')
                print(json.dumps(record), flush=True)
                tracker.log(record, preview=preview_result)
            if step % config['training']['save_every'] == 0:
                _checkpoint(engine, config, stream, context, dataset_info, out_dir,
                            tracking=tracker.state_dict() or tracking_state)
                last_saved = step
        if last_saved != engine.progress['stage_update']:
            _checkpoint(engine, config, stream, context, dataset_info, out_dir,
                        tracking=tracker.state_dict() or tracking_state)
        distributed.barrier()
        exit_code = 0
    finally:
        if tracker is not None:
            tracker.finish(exit_code=exit_code)
        distributed.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--init-from')
    group.add_argument('--resume')
    group.add_argument('--continue-from', help='Training checkpoint to continue with a new lr schedule, Adam betas '
                       'or GPU count: restores weights, optimizer, EMA and counters; every other setting must match.')
    args = parser.parse_args()
    run(load_config(args.config, args.set), device=args.device, init_from=args.init_from, resume=args.resume,
        continue_from=args.continue_from)


if __name__ == '__main__':
    main()
