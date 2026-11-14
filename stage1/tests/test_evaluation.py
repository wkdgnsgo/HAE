"""CPU tests of the validation gate and shared evaluation math, no pretrained weights."""
import json

import numpy as np
from PIL import Image
import pytest
import torch

from hae.metrics import (FeatureMoments, dataset_fingerprint, frechet_distance,
                                  uint8_roundtrip, load_real_statistics,
                                  psnr_per_image, reconstruct, save_real_statistics, ssim_per_image)


class InvertImageNet(torch.nn.Module):
    def forward(self, normalized):
        mean = normalized.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = normalized.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        return normalized * std + mean, None


class ConstantOutput(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, normalized):
        return torch.full_like(normalized, self.value)


def make_grid_file(path, originals, reconstructions, size=4):
    count = len(originals)
    assert count == 10
    pixels = np.zeros((4 * (size + 2) + 2, 5 * (size + 2) + 2, 3), dtype=np.uint8)
    for index, value in enumerate(list(originals) + list(reconstructions)):
        y = 2 + index // 5 * (size + 2)
        x = 2 + index % 5 * (size + 2)
        pixels[y:y + size, x:x + size] = value
    Image.fromarray(pixels).save(path)


def test_psnr_uses_per_image_mean_and_known_error():
    target = torch.zeros(2, 3, 4, 4)
    prediction = torch.stack([torch.full((3, 4, 4), 0.1), torch.ones(3, 4, 4)])
    torch.testing.assert_close(psnr_per_image(prediction, target), torch.tensor([20., 0.], dtype=torch.float64))
    assert torch.isposinf(psnr_per_image(target, target)).all()


def scipy_reference_ssim(prediction, target):
    """Independent skimage-style SSIM: scipy Gaussian (sigma 1.5, truncate 3.5 = 11 taps), crop 5, mean."""
    from scipy.ndimage import gaussian_filter

    def blur(values):
        return gaussian_filter(values, sigma=1.5, truncate=3.5, mode="reflect")

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    scores = []
    for image_x, image_y in zip(prediction.numpy(), target.numpy()):
        channel_scores = []
        for x, y in zip(image_x, image_y):
            mean_x, mean_y = blur(x), blur(y)
            variance_x = blur(x * x) - mean_x ** 2
            variance_y = blur(y * y) - mean_y ** 2
            covariance = blur(x * y) - mean_x * mean_y
            ssim_map = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)
                        / ((mean_x ** 2 + mean_y ** 2 + c1) * (variance_x + variance_y + c2)))
            channel_scores.append(ssim_map[5:-5, 5:-5].mean())
        scores.append(np.mean(channel_scores))
    return np.array(scores)


def test_ssim_matches_independent_scipy_gaussian_reference():
    generator = torch.Generator().manual_seed(0)
    target = torch.rand(3, 3, 24, 31, generator=generator, dtype=torch.float64)
    prediction = (target + 0.1 * torch.randn(target.shape, generator=generator, dtype=torch.float64)).clamp(0, 1)
    prediction[2] = target[2].flip(-1)
    np.testing.assert_allclose(ssim_per_image(prediction, target).numpy(),
                               scipy_reference_ssim(prediction, target), rtol=1e-10, atol=1e-12)


def test_ssim_known_values_and_minimum_size():
    image = torch.rand(2, 3, 16, 16, generator=torch.Generator().manual_seed(1))
    torch.testing.assert_close(ssim_per_image(image, image), torch.ones(2, dtype=torch.float64))
    half, black = torch.full((1, 3, 16, 16), 0.5), torch.zeros(1, 3, 16, 16)
    torch.testing.assert_close(ssim_per_image(half, black),
                               torch.tensor([1e-4 / (0.25 + 1e-4)], dtype=torch.float64))
    with pytest.raises(ValueError, match="at least 11x11"):
        ssim_per_image(torch.zeros(1, 3, 10, 16), torch.zeros(1, 3, 10, 16))


def test_frechet_distance_closed_forms():
    assert frechet_distance(np.array([0., 1, 2]), np.diag([1., 4, 9]),
                            np.array([1., 1, 0]), np.diag([4., 1, 1])) == pytest.approx(11.0, abs=1e-9)
    factor = np.random.default_rng(3).normal(size=(6, 6))
    covariance = factor @ factor.T + np.eye(6)
    assert frechet_distance(np.ones(6), covariance, np.ones(6), covariance) == pytest.approx(0.0, abs=1e-8)
    with pytest.raises(ValueError, match="shapes differ"):
        frechet_distance(np.zeros(2), np.eye(2), np.zeros(3), np.eye(3))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_model_outputs_rejected_before_clamp(value):
    with pytest.raises(ValueError, match="nonfinite"):
        reconstruct(ConstantOutput(value), torch.zeros(1, 3, 4, 4))


def test_normalization_applied_once_and_return_tuple_supported():
    inputs = torch.rand(2, 3, 4, 4)
    torch.testing.assert_close(reconstruct(InvertImageNet(), inputs), inputs)


def test_uint8_protocol_truncates_rather_than_rounds():
    values = torch.tensor([[-1, 0.5, 0.999, 2.]])
    expected = torch.tensor([[0, 127, 254, 255.]]) / 255
    torch.testing.assert_close(uint8_roundtrip(values), expected)


def test_streaming_feature_moments_equal_full_numpy():
    values = np.random.default_rng(42).normal(size=(17, 5)) + 1e5
    moments = FeatureMoments(5)
    for chunk in (values[:1], values[1:6], values[6:16], values[16:]):
        moments.update(chunk)
    mean, covariance = moments.finalize()
    np.testing.assert_allclose(mean, values.mean(0), rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(covariance, np.cov(values, rowvar=False), rtol=1e-10, atol=1e-10)
    assert moments.count == len(values)
    with pytest.raises(ValueError, match="at least two"):
        FeatureMoments(5).finalize()


def test_real_stats_cache_rejects_wrong_data_protocol_count(tmp_path):
    moments = FeatureMoments(3)
    moments.update(np.eye(3))
    path = tmp_path / "cache.npz"
    protocol = {"size": 256}
    save_real_statistics(path, moments, "correct", protocol)
    mean, covariance = load_real_statistics(path, "correct", protocol, 3)
    np.testing.assert_allclose(mean, np.ones(3) / 3)
    assert covariance.shape == (3, 3)
    for fingerprint, config, count in [("wrong", protocol, 3), ("correct", {"size": 512}, 3), ("correct", protocol, 4)]:
        with pytest.raises(ValueError, match="mismatch"):
            load_real_statistics(path, fingerprint, config, count)


def test_fingerprint_detects_same_size_content_change(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"abcd")
    before = dataset_fingerprint([path], tmp_path, {"size": 256})
    path.write_bytes(b"abce")
    assert before != dataset_fingerprint([path], tmp_path, {"size": 256})
    assert before != dataset_fingerprint([path], tmp_path, {"size": 512})


def stub_build_and_load(monkeypatch, model, image_size=4):
    """Exercise CLI plumbing without a large DINO model or external weights."""
    from hae import config, models, checkpoints
    monkeypatch.setattr(config, "load_config", lambda path, overrides: {
        "model": {"decoder": {}, "dino": {}}, "data": {"image_size": image_size}})
    monkeypatch.setattr(models, "build_autoencoder", lambda config, load_pretrained: model)
    monkeypatch.setattr(checkpoints, "load_model_weights", lambda *args, **kwargs: {"strict": True})


def test_evaluate_psnr_streams_all_images_and_marks_no_fid(tmp_path, monkeypatch):
    from hae.evaluate import main
    stub_build_and_load(monkeypatch, ConstantOutput(0.5))
    dataset = tmp_path / "images"
    dataset.mkdir()
    for index in range(3):
        Image.fromarray(np.full((5, 5, 3), 50 + index * 10, dtype=np.uint8)).save(dataset / f"{index}.png")
    output = tmp_path / "results"
    code = main(["--config", "dummy.yaml", "--checkpoint", "dummy.pt", "--data-root", str(dataset),
                 "--output-dir", str(output), "--batch-size", "2"])
    assert code == 0
    summary = json.loads((output / "summary.json").read_text())
    records = json.loads((output / "per_image.json").read_text())
    assert summary["evaluated_images"] == len(records) == 3
    assert summary["rfid"] is None and summary["rfid_status"] == "NOT_REQUESTED"
    assert summary["mean_per_image_psnr_db"] == pytest.approx(np.mean([row["psnr_db"] for row in records]))
    assert not summary["is_subset"]
    assert summary["mean_per_image_ssim"] is None and summary["ssim_protocol"] is None
    assert "ssim" not in records[0]


def test_evaluate_ssim_records_each_image_and_mean(tmp_path, monkeypatch):
    from hae.evaluate import main
    stub_build_and_load(monkeypatch, ConstantOutput(0.5), image_size=16)
    dataset = tmp_path / "images"
    dataset.mkdir()
    rng = np.random.default_rng(0)
    for index in range(3):
        Image.fromarray(rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)).save(dataset / f"{index}.png")
    output = tmp_path / "results"
    code = main(["--config", "dummy.yaml", "--checkpoint", "dummy.pt", "--data-root", str(dataset),
                 "--output-dir", str(output), "--batch-size", "2", "--ssim"])
    assert code == 0
    summary = json.loads((output / "summary.json").read_text())
    records = json.loads((output / "per_image.json").read_text())
    assert len(records) == 3 and all(-1 <= row["ssim"] < 0.1 for row in records)
    assert summary["mean_per_image_ssim"] == pytest.approx(np.mean([row["ssim"] for row in records]))
    assert "Wang et al. 2004" in summary["ssim_protocol"]
