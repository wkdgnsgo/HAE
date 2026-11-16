#!/usr/bin/env python
"""Initialise the HCPE from the encoder's own patch embedding (embedding-level distillation).

The HCPE is trained to reproduce the output of the encoder's original patchify (MSE between the two
[B, H, W, C] embeddings); no transformer forward is needed, so it takes minutes. Stage 1 then loads the
result through ``model.encoder.hcpe_init``. SigLIP2 needs this start: its frozen blocks amplify small
embedding errors, so a randomly initialised HCPE does not recover the teacher's features.

Training images are evenly spaced over the sorted file list of data.train_root; the final check uses the
sorted first --val-images of data.val_root.

  python scripts/pretrain_hcpe_patchify.py --config configs/siglip2/stage1_alignment.yaml \
      --out weights/hcpe_init/siglip2_l16_256_patchify.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hae.config import load_config
from hae.data import ImageDataset
from hae.encoders import build_teacher
from hae.models.generator import CnnPatchEmbed
from hae.training import normalize_imagenet


def cache_images(dataset, indices, device, workers: int) -> torch.Tensor:
    loader = DataLoader(Subset(dataset, [int(i) for i in indices]), batch_size=256, shuffle=False, num_workers=workers)
    return torch.cat([(batch * 255).round().to(torch.uint8) for batch in loader]).to(device)


def embed_input(tower, images_uint8: torch.Tensor):
    normalized = normalize_imagenet(images_uint8.float().div(255))
    return normalized, (tower.to_siglip_input(normalized) if hasattr(tower, "to_siglip_input") else normalized)


@torch.no_grad()
def check(tower, hcpe, images: torch.Tensor) -> dict:
    """Relative MSE of the embedding and the cosine of the final tokens when the HCPE replaces the patchify."""
    original = tower.patch_embed
    hcpe.eval()
    squared, power, cosines = 0.0, 0.0, []
    try:
        for start in range(0, len(images), 50):
            normalized, source = embed_input(tower, images[start:start + 50])
            wanted, output = original(source), hcpe(source)
            squared += (output - wanted).pow(2).sum().item()
            power += wanted.pow(2).sum().item()
            target = tower.forward_features(normalized)["x_norm_patchtokens"]
            tower.patch_embed = hcpe
            tokens = tower.forward_features(normalized)["x_norm_patchtokens"]
            tower.patch_embed = original
            cosines.append(F.cosine_similarity(tokens.float(), target.float(), dim=-1).mean(dim=1))
    finally:
        tower.patch_embed = original
        hcpe.train()
    return {"embed_relative_mse": squared / power, "token_cosine": torch.cat(cosines).mean().item()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="a stage config of the encoder (its model and data sections are used)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--images", type=int, default=200_000, help="training images cached on the GPU (39 GB per 200k)")
    parser.add_argument("--val-images", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--check-every", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    config = load_config(args.config, args.set)
    device = torch.device(args.device)
    torch.manual_seed(config["seed"])
    torch.set_float32_matmul_precision("high")
    started = time.monotonic()

    train_set = ImageDataset(config["data"]["train_root"], config["data"]["image_size"],
                             preprocess=config["data"].get("preprocess", "squash_bilinear"))
    val_set = ImageDataset(config["data"]["val_root"], config["data"]["image_size"],
                           preprocess=config["data"].get("preprocess", "squash_bilinear"))
    count = min(args.images, len(train_set))
    train_indices = np.linspace(0, len(train_set) - 1, count).round().astype(np.int64)
    val_images = cache_images(val_set, range(min(args.val_images, len(val_set))), device, args.workers)
    train_images = cache_images(train_set, train_indices, device, args.workers)
    print(json.dumps({"train_images": len(train_images), "val_images": len(val_images),
                      "load_seconds": round(time.monotonic() - started, 1)}), flush=True)

    tower = build_teacher(config["model"]).to(device)
    hcpe = CnnPatchEmbed(embed_dim=tower.embed_dim).to(device)
    hcpe.reset_parameters()
    optimizer = torch.optim.AdamW(hcpe.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (s + 1) / args.warmup if s < args.warmup else
                                                 0.5 * (1 + math.cos(math.pi * (s - args.warmup) / max(args.steps - args.warmup, 1))))
    generator = torch.Generator(device=device).manual_seed(config["seed"])
    history = [{"step": 0, **check(tower, hcpe, val_images)}]
    print(json.dumps(history[-1]), flush=True)
    running = 0.0
    for step in range(1, args.steps + 1):
        index = torch.randint(len(train_images), (args.batch,), generator=generator, device=device)
        _, source = embed_input(tower, train_images[index])
        with torch.no_grad():
            wanted = tower.patch_embed(source)
        output = hcpe(source)
        loss = F.mse_loss(output, wanted)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(hcpe.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        running += (loss.detach() / wanted.pow(2).mean()).item()
        if step % args.check_every == 0 or step == args.steps:
            window = args.check_every if step % args.check_every == 0 else step % args.check_every
            history.append({"step": step, **check(tower, hcpe, val_images), "train_relative_mse": running / window,
                            "seconds": round(time.monotonic() - started, 1)})
            running = 0.0
            print(json.dumps(history[-1]), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu() for name, value in hcpe.state_dict().items()}
    torch.save(state, out)
    encoder = config["model"].get("encoder") or {}
    report = {"what": "HCPE regressed onto the original patchify output (MSE), scripts/pretrain_hcpe_patchify.py",
              "config": str(Path(args.config).resolve()), "overrides": list(args.set),
              "encoder": encoder.get("type", "dinov3"),
              "encoder_weights": (encoder if encoder.get("type") == "siglip2" else config["model"].get("dino", {})).get("weights_path"),
              "train_root": config["data"]["train_root"], "train_images": int(count), "val_images": len(val_images),
              "steps": args.steps, "batch": args.batch, "lr": args.lr, "seed": config["seed"],
              "parameters": sum(v.numel() for v in state.values()),
              "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "history": history, "final": history[-1],
              "elapsed_seconds": round(time.monotonic() - started, 1), "torch": torch.__version__}
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
