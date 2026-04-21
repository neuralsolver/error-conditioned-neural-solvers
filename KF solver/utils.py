import random

import numpy as np
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sigma_model0(batch_size):
    return torch.ones(batch_size, device=device)


def sigma_model1(j, T, batch_size):
    val = (j + 1) / T
    return torch.full((batch_size,), val, device=device)


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
