import json
import random
from pathlib import Path

import numpy as np
import scipy.io
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


def find_mat_file(data_path, file_name=None):
    data_path = Path(data_path)
    if file_name is not None:
        return data_path / file_name

    mat_files = sorted(data_path.glob("*.mat"))
    if len(mat_files) == 0:
        raise FileNotFoundError(f"No .mat file found in {data_path}")
    if len(mat_files) > 1:
        names = ", ".join(str(path.name) for path in mat_files)
        raise ValueError(f"More than one .mat file found in {data_path}: {names}. Please set file name in config.")
    return mat_files[0]


def load_ps_mat(data_path, file_name=None, f_key="f_data", u_key="phi_data"):
    mat_path = find_mat_file(data_path, file_name)
    data = scipy.io.loadmat(mat_path)
    F_data = data[f_key]
    U = data[u_key]
    return U, F_data, mat_path


def make_loader(U, F_data, batch_size, shuffle=True, device=None):
    U = torch.from_numpy(U).float().unsqueeze(1)
    F_data = torch.from_numpy(F_data).float().unsqueeze(1)

    if device is not None:
        U = U.to(device)
        F_data = F_data.to(device)

    dataset = TensorDataset(U, F_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return U, F_data, dataset, loader


def load_train_loader(config, device):
    data_config = config["data"]
    training = config["training"]
    U, F_data, train_path = load_ps_mat(
        data_config["training_path"],
        data_config.get("train_file"),
        data_config.get("f_key", "f_data"),
        data_config.get("u_key", "phi_data"),
    )
    train_u, train_f, train_dataset, train_loader = make_loader(
        U, F_data, training["batch_size"], shuffle=True, device=device
    )
    return train_u, train_f, train_dataset, train_loader, train_path


def load_test_loader(config, device):
    data_config = config["data"]
    testing = config["testing"]
    U_test, F_data_test, test_path = load_ps_mat(
        data_config["testing_path"],
        data_config.get("test_file"),
        data_config.get("f_key", "f_data"),
        data_config.get("u_key", "phi_data"),
    )
    test_u, test_f, test_dataset, test_loader = make_loader(
        U_test, F_data_test, testing["batch_size"], shuffle=False, device=device
    )
    return test_u, test_f, test_dataset, test_loader, test_path


def relative_l2_error(u_pred, u_true):
    return torch.mean(
        torch.linalg.vector_norm(u_pred - u_true, dim=(1,2,3)) /
        (torch.linalg.vector_norm(u_true, dim=(1,2,3)) + 1e-12)
    )
