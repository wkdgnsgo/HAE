"""Autoencoder construction and trainable-scope selection."""
from contextlib import nullcontext
import importlib
from pathlib import Path
import sys

import torch
from torch import nn

from .generator import CnnPatchEmbed, DCAE_Decoder, DecoderConfig


def _backbone_constructor(repo: Path, model_name: str):
    """Import backbone entrypoints only, avoiding optional DINO task-head packages."""
    repo = repo.resolve()
    existing = sys.modules.get("dinov3")
    if existing is not None and Path(existing.__file__).resolve().parent != repo / "dinov3":
        raise RuntimeError("A different dinov3 package is already imported; use a fresh process")
    sys.path.insert(0, str(repo))
    try:
        module = importlib.import_module("dinov3.hub.backbones")
        if not model_name.startswith("dinov3_vit") or not hasattr(module, model_name):
            raise ValueError(f"Unsupported DINOv3 ViT backbone: {model_name}")
        return getattr(module, model_name)
    finally:
        sys.path.remove(str(repo))


def _load_dino(config: dict, load_pretrained: bool) -> nn.Module:
    repo = Path(config.get("repo_path") or Path(__file__).resolve().parents[2] / "vendor/dinov3")
    if not (repo / "hubconf.py").is_file():
        raise FileNotFoundError(f"DINOv3 source repository not found: {repo}")
    weights_path = None
    if load_pretrained:
        weights_path = Path(config.get("weights_path") or "__missing_dino_weights__")
        if not weights_path.is_file():
            raise FileNotFoundError(f"Provide local DINOv3 pretrained weights: {weights_path}")
    constructor = _backbone_constructor(repo, config.get("model_name", "dinov3_vitl16"))
    backbone = constructor(pretrained=False)
    if load_pretrained:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        backbone.load_state_dict(state, strict=True)
    return backbone


class Autoencoder(nn.Module):
    """Frozen ViT encoder whose patch embedding is replaced by the HCPE, plus a DC-AE decoder.

    Inputs are ImageNet-normalized. ``forward`` returns ``(reconstruction, patch_tokens)``; the
    reconstruction targets RGB in [0,1] and is clamped by the losses and the evaluation.
    """

    def __init__(self, backbone: nn.Module, decoder_cfg: DecoderConfig,
                 frozen_backbone_eval: bool = True):
        super().__init__()
        self.dino_model = backbone
        if self.dino_model.embed_dim != decoder_cfg.latent_channels:
            raise ValueError("Decoder latent_channels must equal DINOv3 embed_dim")
        self.dino_model.patch_embed = CnnPatchEmbed(embed_dim=self.dino_model.embed_dim)
        self.dino_model.patch_embed.reset_parameters()
        self.decoder = DCAE_Decoder(decoder_cfg)
        self.encoder_frozen = False
        self.trainable_scope = "cnn_decoder"
        self.frozen_backbone_eval = frozen_backbone_eval
        configure_trainable(self, "cnn_decoder")

    def forward(self, image: torch.Tensor):
        patch_tokens = self.encode(image)
        B, N, C = patch_tokens.shape
        H = W = int(N**0.5)
        decoder_input = patch_tokens.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
        reconstructed_image = self.decode(decoder_input)
        return reconstructed_image, patch_tokens

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        context = torch.no_grad() if self.encoder_frozen else nullcontext()
        with context:
            dino_output = self.dino_model.forward_features(image)
            return dino_output["x_norm_patchtokens"]

    def decode(self, decoder_input):
        return self.decoder(decoder_input)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen_backbone_eval:
            self.dino_model.eval()
        if not self.encoder_frozen:
            self.dino_model.patch_embed.train(mode)
        if self.trainable_scope == "head":
            self.decoder.eval()
            self.decoder.project_out.train(mode)
        return self


def build_autoencoder(model_config: dict, load_pretrained: bool = True):
    decoder_config = DecoderConfig(**dict(model_config["decoder"]))
    backbone = _load_dino(dict(model_config["dino"]), load_pretrained)
    return Autoencoder(
        backbone, decoder_config,
        frozen_backbone_eval=bool(model_config["dino"].get("frozen_backbone_eval", True)),
    )


def build_teacher(model_config: dict):
    teacher = _load_dino(dict(model_config["dino"]), load_pretrained=True)
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


def configure_trainable(model: Autoencoder, scope: str = "cnn_decoder"):
    if scope not in {"cnn_decoder", "decoder", "head"}:
        raise ValueError(f"Unknown trainable scope: {scope}")
    model.requires_grad_(False)
    if scope == "cnn_decoder":
        model.dino_model.patch_embed.requires_grad_(True)
        model.decoder.requires_grad_(True)
    elif scope == "decoder":
        model.decoder.requires_grad_(True)
    else:
        model.decoder.project_out.requires_grad_(True)
    model.encoder_frozen = scope != "cnn_decoder"
    model.trainable_scope = scope
    model.train(model.training)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def get_last_layer(model: Autoencoder):
    """Final output convolution weight for adaptive GAN gradient balancing."""
    convolutions = [module for module in model.decoder.project_out.modules()
                    if isinstance(module, nn.Conv2d)]
    if not convolutions:
        raise ValueError("decoder.project_out has no output convolution")
    return convolutions[-1].weight
