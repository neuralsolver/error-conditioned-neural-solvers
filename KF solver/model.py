import torch
import torch.nn as nn

from neuralop.models import FNO

import k_diffusion as K
from k_diffusion.models.image_transformer_v2 import (
    ImageTransformerDenoiserModelV2Orig,
    GlobalAttentionSpec,
    NeighborhoodAttentionSpec,
    LevelSpec,
    MappingSpec,
    AdaRMSNorm,
)

from config import (
    DEPTHS,
    WIDTHS,
    D_FFS,
    SELF_ATTNS,
    MAPPING_DEPTH,
    MAPPING_WIDTH,
    MAPPING_D_FF,
    PATCH_SIZE,
)


class FNO_CNN(FNO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lifting = nn.Sequential(
            nn.Conv2d(self.in_channels+2, self.hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),

            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),

            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=1),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),

            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),

            nn.Conv2d(self.hidden_channels, self.out_channels, kernel_size=1)
        )

    def forward(self, x, split=False):
          out = super().forward(x)
          if split:
              u_pred   = out[:, :out.shape[1]//2, ...]
              u_pred_t = out[:, out.shape[1]//2:, ...]
              return u_pred, u_pred_t
          return out


class CNNFeedForwardBlock(nn.Module):
    def __init__(self, dim=384, cond_dim=768, hidden_dim=1152, groups=32, dropout=0.0):
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

    def forward(self, x, cond):
        residual = x
        x = self.norm(x, cond)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.net(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return residual + x


def build_model(in_channels, out_channels):
    levels = [
        LevelSpec(depth=d, width=w, d_ff=ff, self_attn=sa, dropout=0.0)
        for d, w, ff, sa in zip(DEPTHS, WIDTHS, D_FFS, SELF_ATTNS)
    ]
    mapping = MappingSpec(
        depth=MAPPING_DEPTH,
        width=MAPPING_WIDTH,
        d_ff=MAPPING_D_FF,
        dropout=0.0,
    )
    model = ImageTransformerDenoiserModelV2Orig(
        levels=levels,
        mapping=mapping,
        in_channels=in_channels,
        out_channels=out_channels,
        patch_size=PATCH_SIZE,
    )

    #model.up_levels[-1] = CNNLevel(levels[0].width, levels[0].depth)

    return model


def apply_cnn_ff(model, dim=384, cond_dim=768, hidden_dim=1152, dropout=0.0):
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
