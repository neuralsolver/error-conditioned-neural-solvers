from k_diffusion.models.image_transformer_v2 import (
    GlobalAttentionSpec,
    NeighborhoodAttentionSpec,
)


DEPTHS  = [2, 11]
WIDTHS  = [384, 768]
D_FFS   = [w * 3 for w in WIDTHS]
SELF_ATTNS = [
    NeighborhoodAttentionSpec(d_head=64, kernel_size=7),
    GlobalAttentionSpec(d_head=64),
]
MAPPING_DEPTH = 1
MAPPING_WIDTH = 768
MAPPING_D_FF  = MAPPING_WIDTH * 3

IN_CHANNELS  = 39
OUT_CHANNELS = 20
PATCH_SIZE   = [8, 8]
