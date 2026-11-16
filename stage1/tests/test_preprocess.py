"""ADM/RAE center-crop preprocessing and its wiring into data, preview, config and evaluation (CPU)."""
import json

import numpy as np
from PIL import Image
import pytest
import torch

from hae.config import assert_resume_compatible, load_config, validate_config
from hae.data import PREPROCESS_DESCRIPTIONS, ImageDataset, preprocess_image
from test_evaluation import ConstantOutput, stub_build_and_load


def adm_reference(pil_image, image_size):
    """guided-diffusion image_datasets.center_crop_arr (also RAE pack_images.py), verbatim logic."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


def random_image(width, height, seed):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8))


def as_tensor(pixels):
    return torch.from_numpy(np.ascontiguousarray(pixels)).permute(2, 0, 1).float() / 255


@pytest.mark.parametrize("size", [(37, 50), (70, 33), (16, 16), (40, 90)])
def test_adm_center_crop_matches_guided_diffusion(size):
    image = random_image(*size, seed=sum(size))
    np.testing.assert_array_equal(preprocess_image(image, 16, "adm_center_crop"), adm_reference(image, 16))


def test_dataset_default_is_unchanged_and_adm_is_selectable(tmp_path):
    image = random_image(30, 20, seed=1)
    image.save(tmp_path / "a.png")
    squash = np.array(image.resize((16, 16), Image.Resampling.BILINEAR))
    torch.testing.assert_close(ImageDataset(tmp_path, 16)[0], as_tensor(squash))
    torch.testing.assert_close(ImageDataset(tmp_path, 16, preprocess="adm_center_crop")[0],
                               as_tensor(adm_reference(image, 16)))
    with pytest.raises(ValueError, match="Unknown preprocess"):
        ImageDataset(tmp_path, 16, preprocess="center_crop")


def test_preview_uses_configured_preprocess(tmp_path):
    from hae.preview import FixedPreview
    root = tmp_path / "val"
    root.mkdir()
    image = random_image(40, 24, seed=2)
    image.save(root / "a.png")
    config = {"logging": {"wandb": {"num_images": 1}},
              "data": {"val_root": str(root), "image_size": 16, "preprocess": "adm_center_crop"}}
    preview = FixedPreview(config, tmp_path)
    assert preview.manifest["preprocessing"]["resize"] == PREPROCESS_DESCRIPTIONS["adm_center_crop"]
    torch.testing.assert_close(preview.originals[0], as_tensor(adm_reference(image, 16)))


def test_resume_treats_missing_preprocess_as_squash():
    current = load_config("configs/stage1_alignment.yaml")
    saved = json.loads(json.dumps(current))
    del saved["data"]["preprocess"]
    assert_resume_compatible(current, saved)
    changed = load_config("configs/stage1_alignment.yaml", ["data.preprocess=adm_center_crop"])
    with pytest.raises(ValueError, match="Resume config differs"):
        assert_resume_compatible(changed, saved)


def test_evaluate_records_preprocess_and_separates_dataset_hash(tmp_path, monkeypatch):
    from hae.evaluate import main
    stub_build_and_load(monkeypatch, ConstantOutput(0.5), image_size=16)
    dataset = tmp_path / "images"
    dataset.mkdir()
    for index in range(2):
        random_image(30, 20, seed=10 + index).save(dataset / f"{index}.png")
    summaries = {}
    for preprocess in ("squash_bilinear", "adm_center_crop"):
        output = tmp_path / preprocess
        code = main(["--config", "dummy.yaml", "--checkpoint", "dummy.pt", "--data-root", str(dataset),
                     "--output-dir", str(output), "--preprocess", preprocess])
        assert code == 0
        summaries[preprocess] = json.loads((output / "summary.json").read_text())
        assert summaries[preprocess]["preprocess"] == preprocess
    assert summaries["squash_bilinear"]["dataset_sha256"] != summaries["adm_center_crop"]["dataset_sha256"]
