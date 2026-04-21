import torch
from torch import Tensor

#################################################################################
#                          Token Masking and Unmasking                          #
#################################################################################


def get_mask(batch, length, mask_ratio, device):
    """
    Get the binary mask for the input sequence.
    Args:
        - batch: batch size
        - length: sequence length
        - mask_ratio: ratio of tokens to mask
    return:
        mask_dict with following keys:
        - mask: binary mask, 0 is keep, 1 is remove
        - ids_keep: indices of tokens to keep
        - ids_restore: indices to restore the original order
    """
    len_keep = int(length * (1 - mask_ratio))
    noise = torch.rand(batch, length, device=device)  # noise in [0, 1]
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    ids_remove = ids_shuffle[:, len_keep:]

    mask = torch.ones([batch, length], device=device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return {
        "mask": mask,
        "ids_keep": ids_keep,
        "ids_remove": ids_remove,
        "ids_restore": ids_restore,
    }


def mask_out_token(x, ids_keep):
    """
    Mask out the tokens specified by ids_keep.
    Args:
        - x: input sequence, [N, L, D]
        - ids_keep: indices of tokens to keep
    return:
        - x_masked: masked sequence
    """
    N, L, D = x.shape  # batch, length, dim
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    return x_masked


def mask_tokens(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


def unmask_tokens(x, ids_restore, mask_token, extras=0):
    # x: [N, T, D] if extras == 0 (i.e., no cls token) else x: [N, T+1, D]
    x_ = torch.cat([x[:, extras:, :], mask_token], dim=1)  # no cls token
    x_ = torch.gather(
        x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
    )  # unshuffle
    x = torch.cat([x[:, :extras, :], x_], dim=1)  # append cls token
    return x


def split_tokens(x: Tensor, mask_dict: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    keep = mask_out_token(x, mask_dict["ids_keep"])
    remove = mask_out_token(x, mask_dict["ids_remove"])
    return keep, remove


def combine_tokens(
    x: Tensor, y: Tensor, ids_keep: Tensor, ids_restore: Tensor
) -> Tensor:
    x = mask_out_token(x, ids_keep)
    y = mask_out_token(y, 1 - ids_keep)
    return unmask_tokens(x, ids_restore, y)
