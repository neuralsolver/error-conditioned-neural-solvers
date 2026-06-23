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

    h5_files = sorted(list(data_path.glob("*.h5")) + list(data_path.glob("*.hdf5")) + list(data_path.glob("*.mat")))
    if len(h5_files) == 0:
        raise FileNotFoundError(f"No .h5 file found in {data_path}")
    if len(h5_files) > 1:
        names = ", ".join(str(path.name) for path in h5_files)
        raise ValueError(f"More than one .h5 file found in {data_path}: {names}. Please set file name in config.")
    return h5_files[0]


def load_darcy_h5(data_path, file_name=None, a_key="thresh_a_data", u_key="thresh_p_data"):
    h5_path = find_h5_file(data_path, file_name)
    with h5py.File(h5_path, "r") as f:
        A = f[a_key][:]
        U = f[u_key][:]
    return U, A, h5_path


def make_loader(U, A, batch_size, shuffle=True, device=None):
    U = torch.from_numpy(U).float().permute(2,0,1).unsqueeze(1)
    A = torch.from_numpy(A).float().permute(2,0,1).unsqueeze(1)

    if device is not None:
        U = U.to(device)
        A = A.to(device)
        
    dataset = TensorDataset(U, A)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return U, A, dataset, loader


def load_train_loader(config, device):
    data_config = config["data"]
    training = config["training"]
    
    U, A, train_path = load_darcy_h5(
        data_config["training_path"],
        data_config.get("train_file"),
        data_config.get("a_key", "thresh_a_data"),
        data_config.get("u_key", "thresh_p_data"),
    )
    train_u, train_a, train_dataset, train_loader = make_loader(
        U, A, training["batch_size"], shuffle=True, device=device
    )
    return train_u, train_a, train_dataset, train_loader, train_path


def load_test_loader(config, device):
    data_config = config["data"]
    testing = config["testing"]
    
    U_test, A_test, test_path = load_darcy_h5(
        data_config["testing_path"],
        data_config.get("test_file"),
        data_config.get("a_key", "thresh_a_data"),
        data_config.get("u_key", "thresh_p_data"),
    )
    test_u, test_a, test_dataset, test_loader = make_loader(
        U_test, A_test, testing["batch_size"], shuffle=False, device=device
    )
    return test_u, test_a, test_dataset, test_loader, test_path


def relative_l2_error(u_pred, u_true):
    return torch.mean(
        torch.linalg.vector_norm(u_pred - u_true, dim=(1,2,3)) /
        (torch.linalg.vector_norm(u_true, dim=(1,2,3)) + 1e-12)
    )
