from __future__ import annotations

import torch
import torch.nn as nn

from image_transformer_v2 import (
    AdaRMSNorm,
    GlobalAttentionSpec,
    ImageTransformerDenoiserModelV2Orig,
    LevelSpec,
    MappingSpec,
    NeighborhoodAttentionSpec,
)


class CNNFeedForwardBlock(nn.Module):
    def __init__(
        self,
        dim: int = 384,
        cond_dim: int = 768,
        hidden_dim: int = 1152,
        groups: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = AdaRMSNorm(dim, cond_dim)
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x, cond)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.net(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return residual + x


def build_model(model_config, device):
    widths = model_config["widths"]
    d_ffs = [w * model_config.get("d_ff_multiplier", 3) for w in widths]
    self_attns = [
        NeighborhoodAttentionSpec(
            d_head=model_config.get("d_head", 64),
            kernel_size=model_config.get("neighborhood_kernel_size", 7),
        ),
        GlobalAttentionSpec(
            d_head=model_config.get("d_head", 64)
        ),
    ]
    levels = [
        LevelSpec(
            depth=d,
            width=w,
            d_ff=ff,
            self_attn=sa,
            dropout=model_config.get("dropout", 0.0),
        )
        for d, w, ff, sa in zip(
            model_config["depths"],
            widths,
            d_ffs,
            self_attns,
        )
    ]
    mapping_width = model_config.get("mapping_width", 768)
    mapping = MappingSpec(
        depth=model_config.get("mapping_depth", 1),
        width=mapping_width,
        d_ff=mapping_width
        * model_config.get("mapping_d_ff_multiplier", 3),
        dropout=model_config.get("dropout", 0.0),
    )
    
    model = ImageTransformerDenoiserModelV2Orig(
        levels=levels,
        mapping=mapping,
        in_channels=model_config["in_channels"],
        out_channels=model_config["out_channels"],
        patch_size=model_config.get("patch_size", [8, 8]),
    )

    return model.to(device)


def apply_cnn_ff(
    model: nn.Module,
    dim: int = 384,
    cond_dim: int = 768,
    hidden_dim: int = 1152,
    dropout: float = 0.0,
) -> nn.Module:
    for layer in model.up_levels[-1]:
        layer.ff = CNNFeedForwardBlock(
            dim=dim,
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    for layer in model.down_levels[-1]:
        layer.ff = CNNFeedForwardBlock(
            dim=dim,
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    return model
