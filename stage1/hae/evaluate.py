"""Reconstruction evaluation of one checkpoint: PSNR, SSIM, LPIPS and rFID."""
from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .metrics import (FID_PROTOCOL, SSIM_PROTOCOL, FeatureMoments, create_fid_model, create_lpips_model,
                      dataset_fingerprint, frechet_distance, inception_features, json_number,
                      load_real_statistics, psnr_per_image, reconstruct,
                      save_real_statistics, ssim_per_image, write_json)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True, help="Same fixed validation images for every comparison.")
    parser.add_argument("--preprocess", choices=("squash_bilinear", "adm_center_crop"), default="squash_bilinear",
                        help="squash_bilinear: bilinear square resize. adm_center_crop: ADM/RAE center crop "
                             "(bicubic), as in RAE's val_256.npz.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--weights", choices=("raw", "ema", "auto"), default="ema")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, help="Pilot only: evaluate sorted first N; report marks this as a subset.")
    parser.add_argument("--lpips", action="store_true", help="LPIPS-VGG on float images scaled to [-1,1].")
    parser.add_argument("--ssim", action="store_true", help="Mean per-image SSIM (11x11 Gaussian, sigma 1.5) on float RGB [0,1].")
    parser.add_argument("--rfid", action="store_true", help="FID between original and reconstructed image distributions.")
    parser.add_argument("--real-stats", help="Optional .npz real FID cache; contents and preprocessing are verified.")
    parser.add_argument("--allow-downloads", action="store_true", help="Explicitly allow missing Inception/VGG weights download.")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser


def run(args) -> dict:
    from .checkpoints import load_model_weights
    from .config import load_config
    from .data import PREPROCESS_DESCRIPTIONS, ImageDataset
    from .encoders import build_autoencoder

    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("Batch size must be positive and workers nonnegative.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive.")
    if args.real_stats and not args.rfid:
        raise ValueError("--real-stats requires --rfid.")
    started = time.monotonic()
    config = load_config(args.config, args.set)
    image_size = int(config["data"]["image_size"])
    root = Path(args.data_root).expanduser().resolve(strict=True)
    dataset = ImageDataset(root, image_size=image_size, preprocess=args.preprocess)
    if not len(dataset):
        raise ValueError("Validation dataset is empty.")
    count = min(len(dataset), args.limit) if args.limit is not None else len(dataset)
    paths = [Path(path) for path in dataset.paths[:count]]
    protocol = {**FID_PROTOCOL, "image_size": image_size, "resize": PREPROCESS_DESCRIPTIONS[args.preprocess]}
    fingerprint = dataset_fingerprint(paths, root, protocol)
    if args.rfid and count < 2:
        raise ValueError("rFID requires at least two images; use the full validation set for a meaningful estimate.")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        torch.backends.cudnn.rnn.fp32_precision = "ieee"
    model = build_autoencoder(config["model"], load_pretrained=False)
    load_report = load_model_weights(Path(args.checkpoint), model, weights=args.weights, map_location="cpu")
    model = model.to(device).eval()
    lpips_model = create_lpips_model(device, args.allow_downloads) if args.lpips else None
    fid_model = create_fid_model(device, args.allow_downloads) if args.rfid else None
    real_moments = FeatureMoments() if args.rfid else None
    reconstruction_moments = FeatureMoments() if args.rfid else None
    cached_real = None
    if args.rfid and args.real_stats and Path(args.real_stats).is_file():
        cached_real = load_real_statistics(args.real_stats, fingerprint, protocol, count)
        if cached_real[0].shape != (2048,):
            raise ValueError("FID cache must contain 2048-dimensional pool3 statistics.")
    loader = DataLoader(Subset(dataset, range(count)), batch_size=args.batch_size,
                        num_workers=args.num_workers, shuffle=False, drop_last=False,
                        pin_memory=device.type == "cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_file = output_dir / "per_image.json.partial"
    psnr_sum = 0.0
    lpips_sum = 0.0
    ssim_sum = 0.0
    evaluated = 0
    with partial_file.open("w") as handle, torch.inference_mode():
        handle.write("[\n")
        for images in loader:
            if not isinstance(images, torch.Tensor):
                raise TypeError("ImageDataset must return RGB tensors without augmentation.")
            images = images.to(device)
            output = reconstruct(model, images)
            psnr = psnr_per_image(output, images).cpu().tolist()
            perceptual = None
            if lpips_model is not None:
                perceptual = lpips_model(output * 2 - 1, images * 2 - 1).flatten().cpu().tolist()
                if len(perceptual) != len(images) or not np.isfinite(perceptual).all():
                    raise ValueError("LPIPS returned invalid per-image metrics.")
                lpips_sum += sum(perceptual)
            similarity = ssim_per_image(output, images).cpu().tolist() if args.ssim else None
            if similarity is not None:
                ssim_sum += sum(similarity)
            if fid_model is not None:
                reconstruction_moments.update(inception_features(fid_model, output))
                if cached_real is None:
                    real_moments.update(inception_features(fid_model, images))
            for index, value in enumerate(psnr):
                row = {"index": evaluated, "image": paths[evaluated].relative_to(root).as_posix(),
                       "psnr_db": json_number(value)}
                if perceptual is not None:
                    row["lpips_vgg"] = perceptual[index]
                if similarity is not None:
                    row["ssim"] = similarity[index]
                if evaluated:
                    handle.write(",\n")
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False))
                evaluated += 1
            psnr_sum += sum(psnr)
            print(f"Evaluated {evaluated}/{count}", flush=True)
        handle.write("\n]\n")
    if evaluated != count:
        raise RuntimeError(f"Evaluated {evaluated} images, expected exactly {count}.")
    rfid = None
    if fid_model is not None:
        real_mean, real_covariance = cached_real if cached_real is not None else real_moments.finalize()
        reconstruction_mean, reconstruction_covariance = reconstruction_moments.finalize()
        if reconstruction_moments.count != evaluated:
            raise RuntimeError("Real and reconstructed FID sample counts differ.")
        if cached_real is None and real_moments.count != evaluated:
            raise RuntimeError("Real FID statistics sample count differs.")
        rfid = frechet_distance(real_mean, real_covariance, reconstruction_mean, reconstruction_covariance)
        if not np.isfinite(rfid):
            raise ValueError("rFID calculation returned a nonfinite value.")
        if args.real_stats and cached_real is None:
            save_real_statistics(args.real_stats, real_moments, fingerprint, protocol)
    partial_file.replace(output_dir / "per_image.json")
    summary = {
        "status": "COMPLETE", "checkpoint": str(Path(args.checkpoint).resolve()),
        "weights_requested": args.weights, "load_report": load_report,
        "model_config": config["model"], "image_size": image_size,
        "data_root": str(root), "preprocess": args.preprocess,
        "dataset_sha256": fingerprint, "dataset_images_available": len(dataset),
        "evaluated_images": evaluated, "is_subset": evaluated < len(dataset),
        "mean_per_image_psnr_db": json_number(psnr_sum / evaluated),
        "lpips_vgg": lpips_sum / evaluated if args.lpips else None,
        "mean_per_image_ssim": ssim_sum / evaluated if args.ssim else None,
        "rfid": rfid, "rfid_status": "MEASURED" if args.rfid else "NOT_REQUESTED",
        "psnr_protocol": "Mean per-image PSNR; clamped float reconstruction vs resized RGB [0,1], max_value=1.",
        "lpips_protocol": "lpips==0.1.4, VGG, version0.1, float RGB mapped to [-1,1]" if args.lpips else None,
        "ssim_protocol": SSIM_PROTOCOL if args.ssim else None,
        "fid_protocol": protocol if args.rfid else None,
        "real_statistics_cache": str(Path(args.real_stats).resolve()) if args.real_stats else None,
        "device": str(device), "seconds": time.monotonic() - started,
        "precision": "float32; no autocast" + ("; TF32 off (fp32_precision=ieee)" if device.type == "cuda" else ""),
    }
    summary["package_versions"] = {}
    for package in ("torch", "torchvision", "numpy", "Pillow", "scipy", "pytorch-fid", "lpips"):
        try:
            summary["package_versions"][package] = version(package)
        except PackageNotFoundError:
            summary["package_versions"][package] = None
    write_json(output_dir / "summary.json", summary)
    return summary


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run(args)
    except Exception as exc:
        write_json(Path(args.output_dir) / "summary.json", {
            "status": "FAIL", "checkpoint": args.checkpoint,
            "error_type": type(exc).__name__, "error": str(exc), "rfid_status": "NOT_COMPLETED"})
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 1
    print(json.dumps({key: summary[key] for key in
                      ("evaluated_images", "mean_per_image_psnr_db", "mean_per_image_ssim", "lpips_vgg",
                       "rfid", "rfid_status")},
                     ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
