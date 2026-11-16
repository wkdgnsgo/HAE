"""SigLIP2 autoencoder and teacher construction.

``model.encoder.hcpe_mode``: ``replace`` puts the HCPE in place of the patchify (as for DINOv3);
``none`` keeps the original SigLIP2 encoder, entirely frozen (decoder-only baseline).
``model.encoder.hcpe_init`` (replace only) starts the HCPE from weights distilled from the original
patchify (scripts/pretrain_hcpe_patchify.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from ..models.factory import Autoencoder, configure_trainable
from ..models.generator import CnnPatchEmbed, DecoderConfig
from .vision import PatchifyEmbed, Siglip2VisionArch, Siglip2VisionTower, load_vision_weights

ENCODER_TYPE = "siglip2"
DEFAULT_WEIGHTS = "weights/siglip2/siglip2-large-patch16-256/vision_model.safetensors"
HCPE_MODES = ("replace", "none")


def encoder_config(model_config: dict) -> dict:
    encoder = model_config.get("encoder")
    if not isinstance(encoder, dict) or encoder.get("type") != ENCODER_TYPE:
        raise ValueError("model.encoder.type must be 'siglip2' for the SigLIP2 factory")
    return encoder


def resolve_arch(encoder: dict) -> Siglip2VisionArch:
    """Inline ``arch`` wins; otherwise read the Hugging Face ``config.json`` next to the weights."""
    inline = encoder.get("arch")
    if inline:
        if not isinstance(inline, dict):
            raise ValueError("model.encoder.arch must be a mapping of Siglip2VisionArch fields")
        return Siglip2VisionArch(**inline)
    weights = Path(encoder.get("weights_path") or DEFAULT_WEIGHTS)
    config_path = Path(encoder.get("config_path") or weights.parent / "config.json")
    if not config_path.is_file():
        raise FileNotFoundError(f"SigLIP2 config.json not found: {config_path}. "
                                "Run scripts/prepare_siglip2.py or set model.encoder.arch.")
    return Siglip2VisionArch.from_hf_config(json.loads(config_path.read_text()))


def validate_model_config(model_config: dict, image_size: int | None = None) -> Siglip2VisionArch:
    """Fail early on shape mismatches between SigLIP2, the HCPE grid and the decoder."""
    encoder = encoder_config(model_config)
    arch = resolve_arch(encoder)
    latent = int(model_config["decoder"]["latent_channels"])
    if arch.hidden_size != latent:
        raise ValueError(f"decoder.latent_channels={latent} must equal the SigLIP2 hidden_size={arch.hidden_size}")
    feature_layer = encoder.get("feature_layer")
    if feature_layer is not None and (isinstance(feature_layer, bool) or not isinstance(feature_layer, int)
                                      or not 1 <= feature_layer <= arch.num_hidden_layers):
        raise ValueError(f"model.encoder.feature_layer must be null or an integer in [1, {arch.num_hidden_layers}]")
    for key in ("post_layernorm", "allow_pos_interpolation", "frozen_backbone_eval"):
        if key in encoder and not isinstance(encoder[key], bool):
            raise ValueError(f"model.encoder.{key} must be true or false")
    if encoder.get("hcpe_mode", "replace") not in HCPE_MODES:
        raise ValueError(f"model.encoder.hcpe_mode must be one of {HCPE_MODES}")
    hcpe_init = encoder.get("hcpe_init")
    if hcpe_init is not None:
        if not isinstance(hcpe_init, str) or not hcpe_init.strip():
            raise ValueError("model.encoder.hcpe_init must be null or the path of saved HCPE weights")
        if encoder.get("hcpe_mode", "replace") != "replace":
            raise ValueError("model.encoder.hcpe_init needs hcpe_mode replace (the HCPE in place of the patchify)")
    weights = encoder.get("weights_path", DEFAULT_WEIGHTS)
    if not isinstance(weights, str) or not weights.strip():
        raise ValueError("model.encoder.weights_path must name the local SigLIP2 vision weights")
    if image_size is not None:
        if image_size % arch.patch_size:
            raise ValueError(f"data.image_size={image_size} must be a multiple of patch_size={arch.patch_size}")
        grid = image_size // arch.patch_size
        if grid != arch.grid_size and not encoder.get("allow_pos_interpolation", False):
            raise ValueError(f"Token grid {grid}x{grid} differs from the pretrained {arch.grid_size}x{arch.grid_size} "
                             "grid; use a matching checkpoint or set model.encoder.allow_pos_interpolation=true")
    return arch


def _build_tower(model_config: dict, load_pretrained: bool) -> Siglip2VisionTower:
    encoder = encoder_config(model_config)
    arch = validate_model_config(model_config)
    tower = Siglip2VisionTower(
        arch, feature_layer=encoder.get("feature_layer"),
        post_layernorm=bool(encoder.get("post_layernorm", True)),
        input_normalization="imagenet",
        allow_pos_interpolation=bool(encoder.get("allow_pos_interpolation", False)))
    if load_pretrained:
        load_vision_weights(tower, encoder.get("weights_path") or DEFAULT_WEIGHTS)
    return tower


class Siglip2HcpeAutoencoder(Autoencoder):
    """SigLIP2 tower with the HCPE in place of its patchify (or without it) plus the DC-AE decoder.

    The tower is stored as ``dino_model`` so checkpoints share one layout across encoders.
    """

    def __init__(self, backbone: Siglip2VisionTower, decoder_cfg: DecoderConfig,
                 frozen_backbone_eval: bool = True, hcpe_mode: str = "replace"):
        if not isinstance(backbone, Siglip2VisionTower):
            raise TypeError("Siglip2HcpeAutoencoder requires a Siglip2VisionTower backbone")
        if hcpe_mode not in HCPE_MODES:
            raise ValueError(f"hcpe_mode must be one of {HCPE_MODES}")
        original = backbone.patch_embed
        super().__init__(backbone, decoder_cfg, frozen_backbone_eval=frozen_backbone_eval)
        if not isinstance(self.dino_model.patch_embed, CnnPatchEmbed):
            raise RuntimeError("The HCPE was not installed in the SigLIP2 tower")
        self.hcpe_mode = hcpe_mode
        if hcpe_mode == "none":
            if not isinstance(original, PatchifyEmbed):
                raise TypeError("hcpe_mode none needs the tower's original PatchifyEmbed")
            self.dino_model.patch_embed = original
            configure_trainable(self, "decoder")

    @property
    def siglip2(self) -> Siglip2VisionTower:
        return self.dino_model

    @property
    def hcpe(self) -> CnnPatchEmbed | None:
        patch_embed = self.dino_model.patch_embed
        return patch_embed if isinstance(patch_embed, CnnPatchEmbed) else None


def load_hcpe_init(model: Siglip2HcpeAutoencoder, path: str) -> None:
    """Start the HCPE from weights distilled from the original patchify (the decoder is untouched)."""
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"model.encoder.hcpe_init not found: {file}. "
                                "Run scripts/pretrain_hcpe_patchify.py first.")
    if model.hcpe_mode != "replace" or not isinstance(model.hcpe, CnnPatchEmbed):
        raise ValueError("hcpe_init needs hcpe_mode replace")
    model.hcpe.load_state_dict(torch.load(file, map_location="cpu", weights_only=True), strict=True)


def build_autoencoder(model_config: dict, load_pretrained: bool = True) -> Siglip2HcpeAutoencoder:
    decoder_config = DecoderConfig(**dict(model_config["decoder"]))
    tower = _build_tower(model_config, load_pretrained)
    model = Siglip2HcpeAutoencoder(
        tower, decoder_config,
        frozen_backbone_eval=bool(model_config["encoder"].get("frozen_backbone_eval", True)),
        hcpe_mode=model_config["encoder"].get("hcpe_mode", "replace"))
    if load_pretrained and model_config["encoder"].get("hcpe_init"):
        load_hcpe_init(model, model_config["encoder"]["hcpe_init"])
    return model


def build_teacher(model_config: dict) -> nn.Module:
    """The original SigLIP2 tower (single-conv patchify) as the frozen alignment target."""
    teacher = _build_tower(model_config, load_pretrained=True)
    if not isinstance(teacher.patch_embed, PatchifyEmbed):
        raise RuntimeError("Teacher must keep the original SigLIP2 patch embedding")
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher
