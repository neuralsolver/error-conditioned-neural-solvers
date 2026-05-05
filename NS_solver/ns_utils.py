import math

import torch


def make_forcing(n, device, forcing="default"):
    a = torch.linspace(0, 1, n + 1, device=device)
    a = a[0:-1]
    X, Y = torch.meshgrid(a, a, indexing="ij")

    if forcing == "default":
        return 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))
    if forcing == "KF":
        return -4.0 * torch.cos(2 * math.pi * 4 * Y)
    if forcing == "forcing_shift":
        return 0.1 * (torch.sin(4 * math.pi * (X + Y)) + torch.cos(4 * math.pi * (X + Y)))
    raise ValueError(f"Unknown forcing: {forcing}")
