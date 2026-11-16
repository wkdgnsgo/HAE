"""hae.evaluate split over GPUs: every shard evaluates one contiguous slice, merge writes the same summary.

  CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_sharded.py shard --config C --checkpoint K \
      --data-root data/imagenet/val --shard-index 0 --num-shards 4 --output OUT/shard0.npz
  python scripts/evaluate_sharded.py merge --config C --checkpoint K --data-root data/imagenet/val \
      --real-stats OUT/real_stats.npz --shards OUT/shard*.npz --output-dir OUT
"""
from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hae.checkpoints import load_model_weights
from hae.config import load_config
from hae.data import PREPROCESS_DESCRIPTIONS, ImageDataset
from hae.encoders import build_autoencoder
from hae.metrics import (FID_PROTOCOL, SSIM_PROTOCOL, FeatureMoments, create_fid_model, create_lpips_model,
                                  dataset_fingerprint, frechet_distance, inception_features, load_real_statistics,
                                  psnr_per_image, reconstruct, ssim_per_image, write_json)


def load_dataset(args):
    config = load_config(args.config, [])
    image_size = int(config["data"]["image_size"])
    root = Path(args.data_root).expanduser().resolve(strict=True)
    dataset = ImageDataset(root, image_size=image_size, preprocess=args.preprocess)
    return config, root, dataset, image_size


def run_shard(args):
    for backend in (torch.backends.cuda.matmul, torch.backends.cudnn.conv, torch.backends.cudnn.rnn):
        backend.fp32_precision = "ieee"
    device = torch.device("cuda:0")
    started = time.monotonic()
    config, _, dataset, _ = load_dataset(args)
    count = len(dataset)
    start, end = count * args.shard_index // args.num_shards, count * (args.shard_index + 1) // args.num_shards
    model = build_autoencoder(config["model"], load_pretrained=False)
    report = load_model_weights(Path(args.checkpoint), model, weights="ema", map_location="cpu")
    model = model.to(device).eval()
    lpips_model = create_lpips_model(device)
    fid_model = create_fid_model(device)
    moments = FeatureMoments()
    psnr, ssim, lpips = [], [], []
    loader = DataLoader(Subset(dataset, range(start, end)), batch_size=args.batch_size,
                        num_workers=args.num_workers, shuffle=False, pin_memory=True)
    with torch.inference_mode():
        for step, images in enumerate(loader):
            images = images.to(device)
            output = reconstruct(model, images)
            psnr += psnr_per_image(output, images).cpu().tolist()
            ssim += ssim_per_image(output, images).cpu().tolist()
            lpips += lpips_model(output * 2 - 1, images * 2 - 1).flatten().cpu().tolist()
            moments.update(inception_features(fid_model, output))
            if step % 50 == 0:
                print(f"shard {args.shard_index}: {len(psnr)}/{end - start} {time.monotonic() - started:.0f}s", flush=True)
    if len(psnr) != end - start or not np.isfinite(lpips).all():
        raise RuntimeError("Shard did not evaluate every image with finite metrics.")
    np.savez(args.output, start=start, end=end, psnr=np.array(psnr), ssim=np.array(ssim), lpips=np.array(lpips),
             count=moments.count, mean=moments.mean, m2=moments.m2, seconds=time.monotonic() - started,
             load_report=json.dumps(report))
    print(f"shard {args.shard_index}: done {end - start} images in {time.monotonic() - started:.0f}s", flush=True)


def run_merge(args):
    config, root, dataset, image_size = load_dataset(args)
    shards = sorted((dict(np.load(path)) for path in args.shards), key=lambda s: int(s["start"]))
    covered = 0
    for shard in shards:
        if int(shard["start"]) != covered or len(shard["psnr"]) != int(shard["end"]) - int(shard["start"]):
            raise ValueError("Shards do not tile the validation set exactly.")
        covered = int(shard["end"])
    count = len(dataset)
    if covered != count:
        raise ValueError(f"Shards cover {covered} images, expected {count}.")
    paths = [Path(path) for path in dataset.paths[:count]]
    protocol = {**FID_PROTOCOL, "image_size": image_size, "resize": PREPROCESS_DESCRIPTIONS[args.preprocess]}
    fingerprint = dataset_fingerprint(paths, root, protocol)
    real_mean, real_covariance = load_real_statistics(args.real_stats, fingerprint, protocol, count)
    total = FeatureMoments()
    for shard in shards:
        n, mean, m2 = int(shard["count"]), shard["mean"], shard["m2"]
        new = total.count + n
        delta = mean - total.mean
        total.m2 = total.m2 + m2 + np.outer(delta, delta) * (total.count * n / new)
        total.mean = total.mean + delta * (n / new)
        total.count = new
    mean, covariance = total.finalize()
    rfid = frechet_distance(real_mean, real_covariance, mean, covariance)
    psnr, ssim, lpips = (np.concatenate([s[k] for s in shards]) for k in ("psnr", "ssim", "lpips"))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_image.json").write_text(json.dumps(
        [{"index": i, "image": paths[i].relative_to(root).as_posix(), "psnr_db": float(psnr[i]),
          "lpips_vgg": float(lpips[i]), "ssim": float(ssim[i])} for i in range(count)]) + "\n")
    summary = {
        "status": "COMPLETE", "checkpoint": str(Path(args.checkpoint).resolve()), "weights_requested": "ema",
        "load_report": json.loads(str(shards[0]["load_report"])), "model_config": config["model"],
        "image_size": image_size, "data_root": str(root), "preprocess": args.preprocess,
        "dataset_sha256": fingerprint, "dataset_images_available": len(dataset), "evaluated_images": count,
        "is_subset": False, "mean_per_image_psnr_db": float(psnr.mean()), "lpips_vgg": float(lpips.mean()),
        "mean_per_image_ssim": float(ssim.mean()), "rfid": float(rfid), "rfid_status": "MEASURED",
        "psnr_protocol": "Mean per-image PSNR; clamped float reconstruction vs resized RGB [0,1], max_value=1.",
        "lpips_protocol": "lpips==0.1.4, VGG, version0.1, float RGB mapped to [-1,1]",
        "ssim_protocol": SSIM_PROTOCOL, "fid_protocol": protocol,
        "real_statistics_cache": str(Path(args.real_stats).resolve()),
        "device": f"cuda, {len(shards)} shards (scripts/evaluate_sharded.py)",
        "shard_seconds": [float(s["seconds"]) for s in shards],
        "precision": "float32; no autocast; TF32 off (fp32_precision=ieee)", "package_versions": {}}
    for package in ("torch", "torchvision", "numpy", "Pillow", "scipy", "pytorch-fid", "lpips"):
        try:
            summary["package_versions"][package] = version(package)
        except PackageNotFoundError:
            summary["package_versions"][package] = None
    write_json(out / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("evaluated_images", "mean_per_image_psnr_db", "mean_per_image_ssim",
                                              "lpips_vgg", "rfid", "shard_seconds")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    shard, merge = sub.add_parser("shard"), sub.add_parser("merge")
    for p in (shard, merge):
        p.add_argument("--config", required=True)
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--data-root", required=True)
        p.add_argument("--preprocess", choices=("squash_bilinear", "adm_center_crop"), default="squash_bilinear")
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--num-shards", type=int, required=True)
    shard.add_argument("--output", required=True)
    shard.add_argument("--batch-size", type=int, default=16)
    shard.add_argument("--num-workers", type=int, default=8)
    merge.add_argument("--real-stats", required=True)
    merge.add_argument("--shards", nargs="+", required=True)
    merge.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run_shard(args) if args.command == "shard" else run_merge(args)


if __name__ == "__main__":
    main()
