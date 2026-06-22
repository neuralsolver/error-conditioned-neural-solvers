import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_device(device_name="auto"):
    if device_name == "auto":
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_name)


def find_h5_file(data_path, file_name=None):
    data_path = Path(data_path)
    if file_name is not None:
        return data_path / file_name

    h5_files = sorted(list(data_path.glob("*.h5")) + list(data_path.glob("*.hdf5")))
    if len(h5_files) == 0:
        raise FileNotFoundError(f"No .h5 file found in {data_path}")
    if len(h5_files) > 1:
        names = ", ".join(str(path.name) for path in h5_files)
        raise ValueError(f"More than one .h5 file found in {data_path}: {names}. Please set file name in config.")
    return h5_files[0]


def load_ns_h5(data_path, file_name=None, input_key="a", u_key="u"):
    h5_path = find_h5_file(data_path, file_name)
    with h5py.File(h5_path, "r") as f:
        train_u_in = f[input_key][:]
        train_u_out = f[u_key][:]
    return train_u_in, train_u_out, h5_path


def make_loader(u_in, u_out, batch_size, shuffle=True, num_workers=0, pin_memory=False):
    u_in = torch.from_numpy(u_in).float().unsqueeze(1)
    u_output = torch.from_numpy(data['u'][:]).permute(0,3,1,2).float()
    u_out = torch.cat([train_u_in, train_u_output], dim=1)

    dataset = TensorDataset(u_out, u_in)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return u_out, u_in, dataset, loader


def load_train_loader(config):
    data_config = config["data"]
    training = config["training"]
    train_u_in_np, train_u_out_np, train_path = load_ns_h5(
        data_config["training_path"],
        data_config.get("train_file"),
        data_config.get("input_key", "a"),
        data_config.get("u_key", "u"),
    )
    train_u_out, train_u_in, train_dataset, train_loader = make_loader(
        train_u_in_np,
        train_u_out_np,
        training["batch_size"],
        shuffle=True,
        num_workers=training.get("num_workers", 0),
        pin_memory=training.get("pin_memory", False),
    )
    return train_u_out, train_u_in, train_dataset, train_loader, train_path


def load_test_loader(config):
    data_config = config["data"]
    testing = config["testing"]
    test_u_in_np, test_u_out_np, test_path = load_ns_h5(
        data_config["testing_path"],
        data_config.get("test_file"),
        data_config.get("input_key", "a"),
        data_config.get("u_key", "u"),
    )
    test_u_out, test_u_in, test_dataset, test_loader = make_loader(
        test_u_in_np,
        test_u_out_np,
        testing["batch_size"],
        shuffle=False,
    )
    return test_u_out, test_u_in, test_dataset, test_loader, test_path


def relative_l2_error(u_pred, u_true):
    return torch.mean(
        torch.linalg.vector_norm(u_pred - u_true, dim=(1,2,3)) /
        (torch.linalg.vector_norm(u_true, dim=(1,2,3)) + 1e-12)
    )
