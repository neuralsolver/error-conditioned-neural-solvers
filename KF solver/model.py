from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from k_diffusion.models.image_transformer_v2 import (
    AdaRMSNorm,
    GlobalAttentionSpec,
    ImageTransformerDenoiserModelV2Orig,
    LevelSpec,
    MappingSpec,
    NeighborhoodAttentionSpec,
)


@dataclass
class TransformerConfig:
    depths: list[int] = field(default_factory=lambda: [2, 11])
    widths: list[int] = field(default_factory=lambda: [384, 768])
    patch_size: tuple[int, int] = (8, 8)
    d_head: int = 64
    dropout: float = 0.0
    mapping_depth: int = 1
    mapping_width: int = 768

    @property
    def d_ffs(self):
        return [width * 3 for width in self.widths]

    @property
    def mapping_d_ff(self):
        return self.mapping_width * 3


class CNNFeedForwardBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        cond_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = AdaRMSNorm(dim, cond_dim)
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        residual = x
        x = self.norm(x, cond)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.net(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return residual + x


def build_model(
    in_channels: int,
    out_channels: int,
    config: TransformerConfig | None = None,
):
    config = config or TransformerConfig()

    self_attns = [
        NeighborhoodAttentionSpec(d_head=config.d_head, kernel_size=7),
        GlobalAttentionSpec(d_head=config.d_head),
    ]
    levels = [
        LevelSpec(
            depth=depth,
            width=width,
            d_ff=d_ff,
            self_attn=self_attn,
            dropout=config.dropout,
        )
        for depth, width, d_ff, self_attn in zip(
            config.depths,
            config.widths,
            config.d_ffs,
            self_attns,
        )
    ]
    mapping = MappingSpec(
        depth=config.mapping_depth,
        width=config.mapping_width,
        d_ff=config.mapping_d_ff,
        dropout=config.dropout,
    )

    model = ImageTransformerDenoiserModelV2Orig(
        levels=levels,
        mapping=mapping,
        in_channels=in_channels,
        out_channels=out_channels,
        patch_size=list(config.patch_size),
    )
    _replace_shallow_feedforward(model, config)
    return model


def _replace_shallow_feedforward(
    model: ImageTransformerDenoiserModelV2Orig,
    config: TransformerConfig,
):
    hidden_dim = config.widths[0] * 3
    cond_dim = config.mapping_width

    for layer in model.up_levels[-1]:
        layer.ff = CNNFeedForwardBlock(
            dim=config.widths[0],
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
            dropout=config.dropout,
        )

    for layer in model.down_levels[-1]:
        layer.ff = CNNFeedForwardBlock(
            dim=config.widths[0],
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
            dropout=config.dropout,
        )


def build_kf_models(
    time_steps: int,
    config: TransformerConfig | None = None,
):
    config = config or TransformerConfig()
    model0 = build_model(in_channels=1, out_channels=time_steps, config=config)
    correction_in_channels = 1 + time_steps + max(time_steps - 2, 0)
    model1 = build_model(
        in_channels=correction_in_channels,
        out_channels=time_steps,
        config=config,
    )
    return model0, model1


def sigma_model0(batch_size: int, device: torch.device):
    return torch.ones(batch_size, device=device)


def sigma_model1(
    step_index: int,
    total_steps: int,
    batch_size: int,
    device: torch.device,
):
    value = (step_index + 1) / total_steps
    return torch.full((batch_size,), value, device=device)
