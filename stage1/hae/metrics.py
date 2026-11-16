"""Shared reconstruction/evaluation primitives; optional metrics import lazily."""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FID_PROTOCOL = {
    "name": "historical_rgb_uint8_floor_pytorch_fid",
    "version": 1,
    "rgb": True,
    "resize": "PIL bilinear square; no center crop",
    "quantization": "clamp(0,1), multiply255, truncate uint8, divide255",
    "features": "pytorch-fid==0.3.0 FID InceptionV3 pool3 2048",
}


def normalize_images(images: torch.Tensor) -> torch.Tensor:
    mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (images - mean) / std


def reconstruction_tensor(result: Any) -> torch.Tensor:
    if isinstance(result, torch.Tensor):
        return result
    if isinstance(result, (tuple, list)) and result:
        return reconstruction_tensor(result[0])
    if isinstance(result, dict):
        for key in ("reconstruction", "recon", "reconstructed_images", "images"):
            if key in result:
                return reconstruction_tensor(result[key])
    raise TypeError("Expected a reconstruction tensor or (reconstruction, features).")


def reconstruct(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """ImageNet-normalize RGB [0,1] images, run the model and clamp its output to [0,1]."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Expected RGB images shaped [batch,3,height,width].")
    if not torch.isfinite(images).all():
        raise ValueError("Input contains nonfinite values.")
    output = reconstruction_tensor(model(normalize_images(images)))
    if output.shape != images.shape:
        raise ValueError(f"Reconstruction shape {tuple(output.shape)} != input {tuple(images.shape)}")
    if not torch.isfinite(output).all():
        raise ValueError("Model output contains nonfinite values (before clamping).")
    return output.float().clamp(0, 1)


def psnr_per_image(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("PSNR inputs must have identical [N,C,H,W] shapes.")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("PSNR inputs contain nonfinite values.")
    mse = (prediction.double() - target.double()).square().flatten(1).mean(1)
    return -10 * torch.log10(mse)


SSIM_PROTOCOL = ("Mean per-image SSIM (Wang et al. 2004) on clamped float RGB [0,1]: 11x11 Gaussian window, "
                 "sigma 1.5, K1 0.01, K2 0.03, data range 1, population covariance, float64, valid pixels only "
                 "(5-pixel border excluded), mean over the three channels. Same definition as skimage "
                 "structural_similarity(gaussian_weights=True, sigma=1.5, use_sample_covariance=False, "
                 "data_range=1, channel_axis=...).")


def ssim_per_image(prediction: torch.Tensor, target: torch.Tensor, window: int = 11, sigma: float = 1.5,
                   k1: float = 0.01, k2: float = 0.03, data_range: float = 1.0) -> torch.Tensor:
    """Per-image SSIM following SSIM_PROTOCOL."""
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("SSIM inputs must have identical [N,C,H,W] shapes.")
    if min(prediction.shape[-2:]) < window:
        raise ValueError(f"SSIM needs images of at least {window}x{window} pixels.")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("SSIM inputs contain nonfinite values.")
    x, y = prediction.double(), target.double()
    channels = x.shape[1]
    offsets = torch.arange(window, dtype=torch.float64, device=x.device) - (window - 1) / 2
    kernel = torch.exp(-offsets.square() / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    rows = kernel.view(1, 1, 1, window).repeat(channels, 1, 1, 1)
    columns = kernel.view(1, 1, window, 1).repeat(channels, 1, 1, 1)

    def blur(values):
        values = torch.nn.functional.conv2d(values, rows, groups=channels)
        return torch.nn.functional.conv2d(values, columns, groups=channels)

    mean_x, mean_y = blur(x), blur(y)
    variance_x = blur(x * x) - mean_x.square()
    variance_y = blur(y * y) - mean_y.square()
    covariance = blur(x * y) - mean_x * mean_y
    c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2
    ssim_map = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)
                / ((mean_x.square() + mean_y.square() + c1) * (variance_x + variance_y + c2)))
    return ssim_map.flatten(1).mean(1)


def uint8_roundtrip(images: torch.Tensor) -> torch.Tensor:
    """Quantize to uint8 by truncation, as written to PNG before FID."""
    if not torch.isfinite(images).all():
        raise ValueError("Cannot quantize nonfinite images.")
    return images.clamp(0, 1).mul(255).to(torch.uint8).float().div(255)


def json_number(value: float) -> float | str:
    """Strict JSON: use an explicit string for perfect-match PSNR, never Infinity."""
    value = float(value)
    if np.isnan(value):
        raise ValueError("NaN metric is not a valid result.")
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    return value


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class FeatureMoments:
    """Streaming float64 Chan/Welford moments; O(features^2), not O(images x pixels)."""

    def __init__(self, dimensions: int = 2048):
        self.count = 0
        self.mean = np.zeros(dimensions, dtype=np.float64)
        self.m2 = np.zeros((dimensions, dimensions), dtype=np.float64)

    def update(self, features: np.ndarray | torch.Tensor) -> None:
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.mean.size:
            raise ValueError("Feature dimensions do not match accumulator.")
        if not np.isfinite(values).all():
            raise ValueError("Inception features contain nonfinite values.")
        if len(values) == 0:
            return
        batch_mean = values.mean(axis=0)
        centered = values - batch_mean
        delta = batch_mean - self.mean
        new_count = self.count + len(values)
        self.m2 += centered.T @ centered
        self.m2 += np.outer(delta, delta) * (self.count * len(values) / new_count)
        self.mean += delta * (len(values) / new_count)
        self.count = new_count

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count < 2:
            raise ValueError("FID requires at least two samples in each distribution.")
        return self.mean.copy(), self.m2 / (self.count - 1)


def frechet_distance(mean_1, covariance_1, mean_2, covariance_2, eps: float = 1e-6) -> float:
    """pytorch-fid 0.3.0 calculate_frechet_distance, unchanged except for the sqrtm call:
    SciPy 1.18 removed sqrtm(disp=False), which pytorch-fid 0.3.0 still passes."""
    from scipy import linalg

    mean_1, mean_2 = np.atleast_1d(mean_1), np.atleast_1d(mean_2)
    covariance_1, covariance_2 = np.atleast_2d(covariance_1), np.atleast_2d(covariance_2)
    if mean_1.shape != mean_2.shape or covariance_1.shape != covariance_2.shape:
        raise ValueError("FID mean/covariance shapes differ.")
    difference = mean_1 - mean_2
    covariance_mean = linalg.sqrtm(covariance_1.dot(covariance_2))
    if not np.isfinite(covariance_mean).all():
        offset = np.eye(covariance_1.shape[0]) * eps
        covariance_mean = linalg.sqrtm((covariance_1 + offset).dot(covariance_2 + offset))
    if np.iscomplexobj(covariance_mean):
        if not np.allclose(np.diagonal(covariance_mean).imag, 0, atol=1e-3):
            raise ValueError(f"FID matrix square root has imaginary component {np.max(np.abs(covariance_mean.imag))}")
        covariance_mean = covariance_mean.real
    return float(difference.dot(difference) + np.trace(covariance_1) + np.trace(covariance_2)
                 - 2 * np.trace(covariance_mean))


def dataset_fingerprint(paths: Iterable[Path], root: Path, protocol: dict) -> str:
    """Hash image bytes and relative order: stale real statistics cannot be reused."""
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode())
    for path in paths:
        path = Path(path)
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def save_real_statistics(path: str | Path, moments: FeatureMoments, fingerprint: str, protocol: dict) -> None:
    mean, covariance = moments.finalize()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(handle, mean=mean, covariance=covariance,
                            count=np.asarray(moments.count), fingerprint=np.asarray(fingerprint),
                            protocol=np.asarray(json.dumps(protocol, sort_keys=True)))


def load_real_statistics(path: str | Path, fingerprint: str, protocol: dict, expected_count: int):
    with np.load(path, allow_pickle=False) as data:
        if str(data["fingerprint"]) != fingerprint:
            raise ValueError("Real-statistics cache dataset fingerprint mismatch; use a different cache path.")
        if str(data["protocol"]) != json.dumps(protocol, sort_keys=True):
            raise ValueError("Real-statistics cache preprocessing/protocol mismatch.")
        if int(data["count"]) != expected_count:
            raise ValueError("Real-statistics cache sample count mismatch.")
        mean, covariance = data["mean"].copy(), data["covariance"].copy()
    if mean.ndim != 1 or covariance.shape != (len(mean), len(mean)):
        raise ValueError("Malformed real-statistics cache dimensions.")
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError("Real-statistics cache contains nonfinite values.")
    return mean, covariance


def create_fid_model(device: torch.device, allow_downloads: bool = False):
    try:
        from pytorch_fid.inception import InceptionV3
    except ImportError as exc:
        raise RuntimeError("rFID requires optional dependency pytorch-fid==0.3.0.") from exc
    if version("pytorch-fid") != "0.3.0":
        raise RuntimeError("This evaluation protocol requires pytorch-fid==0.3.0.")
    weight = Path(torch.hub.get_dir()) / "checkpoints/pt_inception-2015-12-05-6726825d.pth"
    if not allow_downloads and not weight.is_file():
        raise FileNotFoundError(f"FID Inception weights missing: {weight}. Cache explicitly or use --allow-downloads.")
    return InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()


def create_lpips_model(device: torch.device, allow_downloads: bool = False):
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError("LPIPS requires optional dependency lpips==0.1.4.") from exc
    if version("lpips") != "0.1.4":
        raise RuntimeError("This evaluation protocol requires lpips==0.1.4.")
    weight = Path(torch.hub.get_dir()) / "checkpoints/vgg16-397923af.pth"
    if not allow_downloads and not weight.is_file():
        raise FileNotFoundError(f"LPIPS VGG16 weights missing: {weight}. Cache explicitly or use --allow-downloads.")
    return lpips.LPIPS(net="vgg", version="0.1").to(device).eval()


def inception_features(model, images: torch.Tensor) -> torch.Tensor:
    values = model(uint8_roundtrip(images))[0]
    if values.shape[-2:] != (1, 1):
        values = torch.nn.functional.adaptive_avg_pool2d(values, (1, 1))
    return values.flatten(1)
