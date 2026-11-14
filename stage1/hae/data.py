"""Deterministic image loading shared by training and evaluation."""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PREPROCESS_DESCRIPTIONS = {
    "squash_bilinear": "PIL bilinear square; no center crop",
    "adm_center_crop": "ADM center_crop_arr: PIL box halving while >= 2x, bicubic short side, center crop",
}


def preprocess_image(image, image_size, preprocess="squash_bilinear"):
    """Resize an RGB PIL image to image_size x image_size and return uint8 HWC pixels.

    squash_bilinear: bilinear resize to a square, no crop. adm_center_crop: guided-diffusion's
    center_crop_arr (box halving, bicubic short side, center crop), the protocol of RAE's val_256.npz.
    """
    if preprocess == "squash_bilinear":
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        return np.array(image, dtype=np.uint8, copy=True)
    if preprocess == "adm_center_crop":
        while min(*image.size) >= 2 * image_size:
            image = image.resize(tuple(x // 2 for x in image.size), resample=Image.Resampling.BOX)
        scale = image_size / min(*image.size)
        image = image.resize(tuple(round(x * scale) for x in image.size), resample=Image.Resampling.BICUBIC)
        pixels = np.array(image, dtype=np.uint8, copy=True)
        top, left = (pixels.shape[0] - image_size) // 2, (pixels.shape[1] - image_size) // 2
        return pixels[top:top + image_size, left:left + image_size].copy()
    raise ValueError(f"Unknown preprocess {preprocess!r}; choose one of {sorted(PREPROCESS_DESCRIPTIONS)}.")


def image_paths(root):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Image dataset directory not found: {root}")
    paths = sorted((p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                   key=lambda p: p.relative_to(root).as_posix())
    if not paths:
        raise ValueError(f"No supported images found in {root}")
    return paths


class ImageDataset(Dataset):
    """Images under ``root`` (class folders or flat) as RGB float32 CHW in [0, 1].

    No augmentation, EXIF rotation or normalization; the training engine normalizes the model input.
    """

    def __init__(self, root, image_size=256, preprocess="squash_bilinear"):
        if not isinstance(image_size, int) or image_size <= 0:
            raise ValueError("image_size must be a positive integer.")
        if preprocess not in PREPROCESS_DESCRIPTIONS:
            raise ValueError(f"Unknown preprocess {preprocess!r}; choose one of {sorted(PREPROCESS_DESCRIPTIONS)}.")
        self.root = Path(root).expanduser().resolve()
        self.paths = image_paths(self.root)
        self.image_size = image_size
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        try:
            with Image.open(path) as image:
                pixels = preprocess_image(image.convert("RGB"), self.image_size, self.preprocess)
        except Exception as error:
            raise RuntimeError(f"Failed to decode dataset image: {path}") from error
        return torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255)


def dataset_fingerprint(root, *, hash_contents=False):
    """SHA-256 over the sorted relative paths and file sizes (and the contents with ``hash_contents``)."""
    root = Path(root).expanduser().resolve()
    paths = image_paths(root)
    digest = hashlib.sha256()
    for path in paths:
        name = path.relative_to(root).as_posix()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(path.stat().st_size).encode("ascii") + b"\0")
        if hash_contents:
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        digest.update(b"\0")
    return {"root": str(root), "image_count": len(paths), "sha256": digest.hexdigest(),
            "method": "relative_path_size_and_contents" if hash_contents else "relative_path_and_size"}
