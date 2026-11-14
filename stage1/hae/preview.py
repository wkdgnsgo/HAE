"""Fixed validation-image previews which cannot change the training trajectory."""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .checkpoints import capture_rng_state, restore_rng_state
from .data import PREPROCESS_DESCRIPTIONS, ImageDataset, preprocess_image
from .metrics import psnr_per_image, reconstruct, write_json


class FixedPreview:
    """Cache a small, sorted validation subset and its content manifest.

    An existing manifest must match on every construction, so a resumed run keeps its preview images.
    """

    def __init__(self, config: dict, output_dir: str | Path):
        count = config["logging"]["wandb"]["num_images"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("logging.wandb.num_images must be a positive integer.")
        preprocess = config["data"].get("preprocess", "squash_bilinear")
        dataset = ImageDataset(config["data"]["val_root"], config["data"]["image_size"], preprocess=preprocess)
        selected = dataset.paths[:count]
        self.paths = [path.relative_to(dataset.root).as_posix() for path in selected]
        self.originals = []
        records = []
        for name, path in zip(self.paths, selected):
            content = path.read_bytes()
            with Image.open(BytesIO(content)) as image:
                array = preprocess_image(image.convert("RGB"), dataset.image_size, preprocess)
            self.originals.append(torch.from_numpy(array).permute(2, 0, 1).float().div_(255))
            records.append({"path": name, "sha256": hashlib.sha256(content).hexdigest()})
        self.manifest = {
            "format_version": 1, "root": str(dataset.root), "requested_count": count,
            "image_count": len(selected), "images": records,
            "preprocessing": {"image_size": dataset.image_size, "rgb": True,
                              "resize": PREPROCESS_DESCRIPTIONS[preprocess]},
        }
        manifest_path = Path(output_dir) / "preview_manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            if existing != self.manifest:
                raise ValueError("Preview manifest mismatch: validation selection, contents, or "
                                 "preprocessing changed. Preserve the original subset on resume.")
        else:
            write_json(manifest_path, self.manifest)

    def render(self, engine) -> dict:
        """Render the EMA weights in float32; model modes, weights and RNG are restored afterwards.

        No distributed collectives: call on the logging rank only. The PSNR is the mean over this subset.
        """
        model = engine.model
        flags = [(module, module.training) for module in model.modules()]
        rng_state = capture_rng_state()
        reconstructions, scores = [], []
        try:
            if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
                raise ValueError("Fixed previews require float32 master model parameters.")
            model.eval()
            with engine.ema.average_parameters(model):
                with torch.inference_mode(), torch.autocast(engine.device.type, enabled=False):
                    for original in self.originals:
                        target = original.unsqueeze(0).to(engine.device, dtype=torch.float32)
                        reconstruction = reconstruct(model, target)
                        scores.append(psnr_per_image(reconstruction, target).item())
                        reconstructions.append(reconstruction[0].cpu())
        finally:
            for module, training in flags:
                module.training = training
            restore_rng_state(rng_state)
        return {"images": self._grid(reconstructions),
                "psnr_db": sum(scores) / len(scores), "image_count": len(scores),
                "paths": list(self.paths), "weights": "ema"}

    def _grid(self, reconstructions: list[torch.Tensor]) -> Image.Image:
        count, padding = len(self.originals), 2
        columns = min(5, count)
        rows = math.ceil(count / columns)
        height, width = self.originals[0].shape[-2:]
        pixels = np.zeros((2 * rows * (height + padding) + padding,
                           columns * (width + padding) + padding, 3), dtype=np.uint8)
        for section, images in enumerate((self.originals, reconstructions)):
            for index, tensor in enumerate(images):
                y = padding + (section * rows + index // columns) * (height + padding)
                x = padding + (index % columns) * (width + padding)
                array = tensor.clamp(0, 1).mul(255).to(torch.uint8).permute(1, 2, 0).numpy()
                pixels[y:y + height, x:x + width] = array
        return Image.fromarray(pixels)
