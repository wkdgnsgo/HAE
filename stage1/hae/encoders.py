"""Encoder dispatch: ``model.encoder.type`` selects DINOv3 (default) or SigLIP2."""
from __future__ import annotations

from . import models as _dinov3

ENCODER_TYPES = ("dinov3", "siglip2")


def encoder_type(model_config: dict) -> str:
    encoder = model_config.get("encoder") or {}
    if not isinstance(encoder, dict):
        raise ValueError("model.encoder must be a mapping")
    kind = encoder.get("type", "dinov3")
    if kind not in ENCODER_TYPES:
        raise ValueError(f"Unsupported model.encoder.type={kind!r}; expected one of {ENCODER_TYPES}")
    return kind


def build_autoencoder(model_config: dict, load_pretrained: bool = True):
    if encoder_type(model_config) == "siglip2":
        from .siglip2.factory import build_autoencoder as build
        return build(model_config, load_pretrained)
    return _dinov3.build_autoencoder(model_config, load_pretrained)


def build_teacher(model_config: dict):
    if encoder_type(model_config) == "siglip2":
        from .siglip2.factory import build_teacher as build
        return build(model_config)
    return _dinov3.build_teacher(model_config)
