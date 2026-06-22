import math
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def resolve_data_file(data_path: str | Path, file_name: str | None = None) -> Path:
    data_path = Path(data_path)
    if file_name is not None:
        return data_path / file_name
    if data_path.is_file():
        return data_path

    candidates = sorted(list(data_path.glob("*.h5")) + list(data_path.glob("*.hdf5")))
    if not candidates:
        raise FileNotFoundError(f"No .h5/.hdf5 file found in {data_path}")
    return candidates[0]


def load_ns_tensors(path: str | Path, input_key: str = "a", u_key: str = "u") -> tuple[torch.Tensor, torch.Tensor]:
    with h5py.File(path, "r") as data:
        u_in = torch.from_numpy(data[input_key][:]).float().unsqueeze(1)
        u_output = torch.from_numpy(data[u_key][:]).permute(0, 3, 1, 2).float()
    u_out = torch.cat([u_in, u_output], dim=1)
    return u_out, u_in


def make_loader(
    u_out: torch.Tensor,
    u_in: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    dataset = TensorDataset(u_out, u_in)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


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


def sigma_model0(batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.ones(batch_size, device=device)


def sigma_model1(j: int, T: int, batch_size: int, device: torch.device) -> torch.Tensor:
    val = (j + 1) / T
    return torch.full((batch_size,), val, device=device)


def relative_l2(u_pred: torch.Tensor, u_true: torch.Tensor) -> torch.Tensor:
    return torch.mean(
        torch.linalg.vector_norm(u_pred - u_true, dim=(1, 2, 3))
        / (torch.linalg.vector_norm(u_true, dim=(1, 2, 3)) + 1e-12)
    )
