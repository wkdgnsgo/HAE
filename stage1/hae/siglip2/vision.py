"""Standalone SigLIP2 vision tower that loads Hugging Face ``SiglipModel`` checkpoints.

Mirrors ``transformers`` ``SiglipVisionTransformer`` (patchify conv, learned absolute position
embedding, pre-LayerNorm blocks with tanh-GELU MLPs, final LayerNorm) without the attention-pooling
head and without depending on ``transformers``.

Backbone interface: ``embed_dim``, a replaceable ``patch_embed`` producing ``[B, H, W, C]`` tokens
and ``forward_features`` returning ``{"x_norm_patchtokens": [B, N, C]}``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

HF_PREFIX = "vision_model."
HF_HEAD_PREFIX = "vision_model.head."
HF_IGNORED_PREFIXES = ("text_model.", "logit_scale", "logit_bias")

SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Siglip2VisionArch:
    """Architecture fields of a Hugging Face ``SiglipVisionConfig``."""
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    image_size: int = 224
    patch_size: int = 16
    layer_norm_eps: float = 1e-6
    num_channels: int = 3

    def __post_init__(self):
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.image_size % self.patch_size:
            raise ValueError("Pretraining image_size must be a multiple of patch_size")
        if min(self.hidden_size, self.intermediate_size, self.num_hidden_layers,
               self.num_attention_heads, self.image_size, self.patch_size, self.num_channels) <= 0:
            raise ValueError("Architecture sizes must be positive")

    @classmethod
    def from_hf_config(cls, config: dict) -> "Siglip2VisionArch":
        vision = config.get("vision_config", config)
        model_type = vision.get("model_type", "siglip_vision_model")
        if model_type not in ("siglip_vision_model", "siglip"):
            raise ValueError(f"Not a SigLIP vision config: model_type={model_type}")
        if vision.get("hidden_act", "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
            raise ValueError("Only the SigLIP default hidden_act=gelu_pytorch_tanh is implemented")
        if float(vision.get("attention_dropout", 0.0)) != 0.0:
            raise ValueError("Attention dropout is not implemented (SigLIP2 checkpoints use 0.0)")
        fields = {name: vision[name] for name in cls.__dataclass_fields__ if name in vision}
        return cls(**fields)

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def to_dict(self) -> dict:
        return asdict(self)


class PatchifyEmbed(nn.Module):
    """The original single-convolution patch embedding; output is ``[B, H, W, C]``."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).permute(0, 2, 3, 1)


class Attention(nn.Module):
    def __init__(self, arch: Siglip2VisionArch):
        super().__init__()
        self.num_heads = arch.num_attention_heads
        self.head_dim = arch.head_dim
        self.q_proj = nn.Linear(arch.hidden_size, arch.hidden_size)
        self.k_proj = nn.Linear(arch.hidden_size, arch.hidden_size)
        self.v_proj = nn.Linear(arch.hidden_size, arch.hidden_size)
        self.out_proj = nn.Linear(arch.hidden_size, arch.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        shape = (batch, length, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(batch, length, self.num_heads * self.head_dim)
        return self.out_proj(out)


class Mlp(nn.Module):
    def __init__(self, arch: Siglip2VisionArch):
        super().__init__()
        self.fc1 = nn.Linear(arch.hidden_size, arch.intermediate_size)
        self.fc2 = nn.Linear(arch.intermediate_size, arch.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class EncoderLayer(nn.Module):
    def __init__(self, arch: Siglip2VisionArch):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(arch.hidden_size, eps=arch.layer_norm_eps)
        self.self_attn = Attention(arch)
        self.layer_norm2 = nn.LayerNorm(arch.hidden_size, eps=arch.layer_norm_eps)
        self.mlp = Mlp(arch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


class Siglip2VisionTower(nn.Module):
    """SigLIP2 vision encoder without the pooling head.

    ``input_normalization``: ``"imagenet"`` inputs are converted exactly to SigLIP's ``[-1, 1]`` scaling
    before the patch embedding; ``"siglip"`` inputs pass through unchanged. ``feature_layer`` selects
    how many blocks run (default: all); ``post_layernorm`` applies the final LayerNorm.
    """

    def __init__(self, arch: Siglip2VisionArch, *, feature_layer: int | None = None,
                 post_layernorm: bool = True, input_normalization: str = "imagenet",
                 allow_pos_interpolation: bool = False):
        super().__init__()
        if feature_layer is None:
            feature_layer = arch.num_hidden_layers
        if isinstance(feature_layer, bool) or not isinstance(feature_layer, int) \
                or not 1 <= feature_layer <= arch.num_hidden_layers:
            raise ValueError(f"feature_layer must be an integer in [1, {arch.num_hidden_layers}]")
        if input_normalization not in ("imagenet", "siglip"):
            raise ValueError("input_normalization must be 'imagenet' or 'siglip'")
        self.arch = arch
        self.embed_dim = arch.hidden_size
        self.patch_size = arch.patch_size
        self.feature_layer = feature_layer
        self.post_layernorm = bool(post_layernorm)
        self.input_normalization = input_normalization
        self.allow_pos_interpolation = bool(allow_pos_interpolation)
        self.patch_embed = PatchifyEmbed(arch.num_channels, arch.hidden_size, arch.patch_size)
        self.pos_embed = nn.Embedding(arch.grid_size ** 2, arch.hidden_size)
        self.blocks = nn.ModuleList([EncoderLayer(arch) for _ in range(arch.num_hidden_layers)])
        self.norm = nn.LayerNorm(arch.hidden_size, eps=arch.layer_norm_eps)
        scale = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1) / torch.tensor(SIGLIP_STD).view(1, 3, 1, 1)
        shift = (torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1) - torch.tensor(SIGLIP_MEAN).view(1, 3, 1, 1)) \
            / torch.tensor(SIGLIP_STD).view(1, 3, 1, 1)
        self.register_buffer("renorm_scale", scale, persistent=False)
        self.register_buffer("renorm_shift", shift, persistent=False)

    def to_siglip_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_normalization == "siglip":
            return x
        if x.shape[1] != 3:
            raise ValueError("ImageNet-to-SigLIP renormalization expects 3 channels")
        return x * self.renorm_scale.to(x.dtype) + self.renorm_shift.to(x.dtype)

    def position_embeddings(self, height: int, width: int) -> torch.Tensor:
        grid = self.arch.grid_size
        weight = self.pos_embed.weight
        if (height, width) == (grid, grid):
            return weight.unsqueeze(0)
        if not self.allow_pos_interpolation:
            raise ValueError(f"Token grid {height}x{width} differs from the pretrained {grid}x{grid}; "
                             "set allow_pos_interpolation to interpolate the position embedding")
        table = weight.reshape(1, grid, grid, -1).permute(0, 3, 1, 2).float()
        table = F.interpolate(table, size=(height, width), mode="bicubic", align_corners=False)
        return table.permute(0, 2, 3, 1).reshape(1, height * width, -1).to(weight.dtype)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(self.to_siglip_input(x))
        if tokens.ndim != 4 or tokens.shape[-1] != self.embed_dim:
            raise ValueError("patch_embed must return [B, H, W, embed_dim] tokens")
        batch, height, width, channels = tokens.shape
        return tokens.reshape(batch, height * width, channels) + self.position_embeddings(height, width)

    def forward_features(self, x: torch.Tensor) -> dict:
        tokens = self.embed(x)
        for block in self.blocks[:self.feature_layer]:
            tokens = block(tokens)
        normalized = self.norm(tokens) if self.post_layernorm else tokens
        return {"x_norm_patchtokens": normalized, "x_prenorm_patchtokens": tokens}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)["x_norm_patchtokens"]


def map_hf_vision_state_dict(state: dict) -> dict:
    """Rename Hugging Face ``SiglipModel``/``SiglipVisionModel`` keys to this tower's names."""
    mapped = {}
    for key, value in state.items():
        if key.startswith(HF_IGNORED_PREFIXES) or key.startswith(HF_HEAD_PREFIX):
            continue
        if not key.startswith(HF_PREFIX):
            raise KeyError(f"Unexpected checkpoint key: {key}")
        name = key[len(HF_PREFIX):]
        if name.startswith("embeddings.patch_embedding."):
            target = "patch_embed.proj." + name[len("embeddings.patch_embedding."):]
        elif name == "embeddings.position_embedding.weight":
            target = "pos_embed.weight"
        elif name.startswith("encoder.layers."):
            target = "blocks." + name[len("encoder.layers."):]
        elif name.startswith("post_layernorm."):
            target = "norm." + name[len("post_layernorm."):]
        else:
            raise KeyError(f"Unexpected SigLIP vision key: {key}")
        mapped[target] = value
    if not mapped:
        raise ValueError("Checkpoint contains no SigLIP vision tensors")
    return mapped


def load_vision_weights(tower: Siglip2VisionTower, path: str | Path) -> dict:
    """Strictly load a local safetensors/pt checkpoint; never downloads."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"SigLIP2 weights not found: {path}. Run scripts/prepare_siglip2.py first.")
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
    mapped = map_hf_vision_state_dict(state)
    if not isinstance(tower.patch_embed, PatchifyEmbed):
        raise TypeError("Load pretrained weights before replacing patch_embed with the HCPE")
    tower.load_state_dict(mapped, strict=True)
    return {"path": str(path.resolve()), "tensor_count": len(mapped),
            "parameter_count": sum(int(t.numel()) for t in mapped.values())}
