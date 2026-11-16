"""Checkpoint reading and writing, RNG state and strict weight loading."""
from collections.abc import Mapping
import hashlib
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from .ema import ModelEMA, _check_tensors


TRAINING_KEYS = {"format_version", "model", "ema", "optimizer_g", "optimizer_d",
                 "discriminator", "progress", "config", "rng_states"}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state():
    """Use tensors and Python primitives accepted by torch.load(weights_only=True)."""
    numpy_state = np.random.get_state()
    return {"format_version": 1, "python": random.getstate(),
            "numpy": {"algorithm": numpy_state[0], "keys": numpy_state[1].tolist(),
                      "position": int(numpy_state[2]), "has_gauss": int(numpy_state[3]),
                      "cached_gaussian": float(numpy_state[4])},
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": {"device_count": torch.cuda.device_count(),
                           "device_index": torch.cuda.current_device(),
                           "state": torch.cuda.get_rng_state()}
            if torch.cuda.is_available() else None}


def restore_rng_state(state):
    required = {"format_version", "python", "numpy", "torch_cpu", "torch_cuda"}
    if not isinstance(state, Mapping) or set(state) != required or state["format_version"] != 1:
        raise ValueError("Unsupported RNG state schema.")
    cuda_state = state["torch_cuda"]
    if cuda_state is not None:
        if (not torch.cuda.is_available()
                or cuda_state["device_count"] != torch.cuda.device_count()
                or cuda_state["device_index"] != torch.cuda.current_device()):
            raise ValueError("Exact RNG resume requires the same CUDA topology and rank device.")
    numpy_state = state["numpy"]
    random.setstate(state["python"])
    np.random.set_state((numpy_state["algorithm"], np.asarray(numpy_state["keys"], dtype=np.uint32),
                         numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state["state"].cpu())


def atomic_save(payload, path):
    """Write in the destination filesystem and replace only after a complete save."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load(path, map_location="cpu"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path), device=str(map_location))
    return torch.load(path, map_location=map_location, weights_only=True)


def _is_tensor_dict(value):
    return isinstance(value, Mapping) and bool(value) and all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in value.items())


def read_training_checkpoint(path):
    payload = _load(path)
    if not isinstance(payload, Mapping) or not TRAINING_KEYS.issubset(payload):
        raise ValueError("Expected a complete training checkpoint written by hae.train.")
    if payload["format_version"] != 1 or not _is_tensor_dict(payload["model"]):
        raise ValueError("Unsupported training format or invalid model state.")
    for name in ("optimizer_g", "progress", "config"):
        if not isinstance(payload[name], Mapping):
            raise ValueError(f"Training checkpoint {name} must be a mapping.")
    for name in ("optimizer_d", "discriminator", "ema"):
        if payload[name] is not None and not isinstance(payload[name], Mapping):
            raise ValueError(f"Training checkpoint {name} must be a mapping or None.")
    if (payload["optimizer_d"] is None) != (payload["discriminator"] is None):
        raise ValueError("Discriminator and its optimizer must both be present or both be None.")
    if not isinstance(payload["rng_states"], list) or not payload["rng_states"]:
        raise ValueError("Exact resume requires RNG states for every training rank.")
    return payload


def _merge_package_ema(raw, ema):
    if not isinstance(ema, Mapping) or set(ema) != {"format_version", "decay", "shadow", "buffers"}:
        raise ValueError("Invalid package EMA schema.")
    if ema["format_version"] != 1:
        raise ValueError("Unsupported EMA format.")
    ModelEMA._validate_decay(ema["decay"])
    merged = dict(raw)
    for section in ("shadow", "buffers"):
        values = ema[section]
        if not isinstance(values, Mapping):
            raise ValueError(f"Invalid EMA {section}.")
        for name, value in values.items():
            if name not in raw or not isinstance(value, torch.Tensor) or value.shape != raw[name].shape:
                raise ValueError(f"EMA {section} has an incompatible tensor: {name}")
            merged[name] = value
    return merged


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model_weights(path, model, weights="auto", map_location="cpu"):
    """Strictly load complete model weights and return a JSON-serializable report.

    ``path`` is a training checkpoint written by ``hae.train`` (raw and EMA weights) or a plain tensor
    state_dict. ``weights`` selects raw or EMA; ``auto`` is refused when a checkpoint holds both.
    """
    if weights not in {"auto", "raw", "ema"}:
        raise ValueError("weights must be auto, raw, or ema.")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    payload = _load(path, map_location)
    if _is_tensor_dict(payload):
        state = payload
        selected = weights if weights != "auto" else "unlabelled"
        kind = "state_dict"
    elif isinstance(payload, Mapping) and payload.get("format_version") == 1 and "model" in payload:
        raw = payload["model"]
        if not _is_tensor_dict(raw):
            raise ValueError("Checkpoint model is not a tensor state_dict.")
        has_ema = payload.get("ema") is not None
        if weights == "auto" and has_ema:
            raise ValueError("Both raw and EMA exist; select weights='raw' or 'ema' explicitly.")
        if weights == "ema" and not has_ema:
            raise ValueError("EMA was requested but is missing; refusing a raw fallback.")
        selected = "ema" if weights == "ema" else "raw"
        state = _merge_package_ema(raw, payload["ema"]) if selected == "ema" else raw
        kind = "training_checkpoint"
    else:
        raise ValueError("Unsupported checkpoint format; expected a tensor state_dict or a training checkpoint.")
    _check_tensors(state, model.state_dict(), "Model checkpoint")
    model.load_state_dict(state, strict=True)
    return {"path": str(path.resolve()), "kind": kind, "selected_weights": selected,
            "tensor_count": len(state), "sha256": _sha256(path)}
