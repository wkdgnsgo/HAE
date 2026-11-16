"""Data parallelism with bucketed gradient averaging at optimizer boundaries.

Explicit reduction keeps GAN adaptive autograd.grad separate from DDP reducer hooks.
There is no gradient communication during microbatches. This is synchronous data
parallel training, with the same averaged gradients as the global loss objective.
"""
import os
from dataclasses import dataclass
from datetime import timedelta
import torch
import torch.distributed as dist


@dataclass
class Context:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    device: torch.device = torch.device('cpu')


def initialize(device='auto'):
    world = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local = int(os.environ.get('LOCAL_RANK', 0))
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    resolved = torch.device(device)
    if world > 1 and resolved.type == 'cuda' and resolved.index is not None:
        raise ValueError("Multi-rank training cannot use an explicit CUDA device index. "
                         "Use --device cuda or --device auto so each rank uses LOCAL_RANK.")
    if resolved.type == 'cuda':
        torch.cuda.set_device(local if resolved.index is None else resolved.index)
        resolved = torch.device('cuda', torch.cuda.current_device())
    if world > 1:
        dist.init_process_group('nccl' if resolved.type == 'cuda' else 'gloo', timeout=timedelta(minutes=30))
    return Context(rank, world, local, resolved)


def active():
    return dist.is_available() and dist.is_initialized()


@torch.no_grad()
def broadcast_module(module):
    if active():
        for tensor in list(module.parameters()) + list(module.buffers()):
            dist.broadcast(tensor, src=0)


@torch.no_grad()
def average_gradients(parameters, bucket_bytes=32 * 1024 * 1024):
    params = [p for p in parameters if p.requires_grad]
    if not params:
        raise ValueError('No trainable parameters')
    missing = torch.tensor(any(p.grad is None for p in params), device=params[0].device, dtype=torch.int)
    if active():
        dist.all_reduce(missing, op=dist.ReduceOp.MAX)
    if missing.item():
        raise RuntimeError('A trainable parameter received no gradient; freeze unused parameters explicitly')
    if not active():
        return
    bucket, size = [], 0

    def flush():
        if not bucket:
            return
        flat = torch.cat([p.grad.reshape(-1) for p in bucket])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(dist.get_world_size())
        offset = 0
        for p in bucket:
            n = p.numel()
            p.grad.copy_(flat[offset:offset+n].view_as(p))
            offset += n

    for p in params:
        bytes_ = p.numel() * p.element_size()
        if bucket and (size + bytes_ > bucket_bytes or p.dtype != bucket[0].dtype):
            flush()
            bucket, size = [], 0
        bucket.append(p)
        size += bytes_
    flush()


def require_finite(value, description):
    bad = (~torch.isfinite(value.detach()).all()).to(torch.int)
    if active():
        dist.all_reduce(bad, op=dist.ReduceOp.MAX)
    if bad.item():
        raise FloatingPointError(f'Non-finite {description}; optimizer was not updated')


def mean_metrics(metrics, device):
    if not active():
        return metrics
    keys = sorted(metrics)
    values = torch.tensor([metrics[k] for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(values)
    values /= dist.get_world_size()
    return dict(zip(keys, values.cpu().tolist()))


def gather_objects(value):
    if not active():
        return [value]
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return gathered


def barrier():
    if active():
        dist.barrier()


def close():
    if active():
        dist.destroy_process_group()
