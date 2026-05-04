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


def load_darcy_h5(data_path, file_name=None, a_key="a", f_key="f", u_key="pp", aBC_key="aBc"):
    h5_path = find_h5_file(data_path, file_name)
    with h5py.File(h5_path, "r") as f:
        A = f[a_key][:]
        F_data = f[f_key][:]
        U = f[u_key][:]
        aBC = f[aBC_key][:] if aBC_key in f else None
    return U, F_data, A, aBC, h5_path


def make_loader(U, F_data, A, aBC, batch_size, shuffle=True, device=None, include_aBC=True):
    U = torch.from_numpy(U).float().unsqueeze(1)
    F_data = torch.from_numpy(F_data).float().unsqueeze(1)
    A = torch.from_numpy(A).float().unsqueeze(1)

    if device is not None:
        U = U.to(device)
        F_data = F_data.to(device)
        A = A.to(device)

    if include_aBC:
        if aBC is None:
            aBC = np.zeros((U.shape[0], 2 * (U.shape[2] + U.shape[3])), dtype=np.float32)
        aBC = torch.from_numpy(aBC).float()
        if device is not None:
            aBC = aBC.to(device)
        dataset = TensorDataset(U, F_data, A, aBC)
    else:
        dataset = TensorDataset(U, F_data, A)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return U, F_data, A, aBC, dataset, loader


def load_train_loader(config, device):
    data_config = config["data"]
    training = config["training"]
    U, F_data, A, aBC, train_path = load_darcy_h5(
        data_config["training_path"],
        data_config.get("train_file"),
        data_config.get("a_key", "a"),
        data_config.get("f_key", "f"),
        data_config.get("u_key", "pp"),
        data_config.get("aBC_key", "aBc"),
    )
    train_u, train_f, train_a, aBC, train_dataset, train_loader = make_loader(
        U, F_data, A, aBC, training["batch_size"], shuffle=True, device=device, include_aBC=True
    )
    return train_u, train_f, train_a, aBC, train_dataset, train_loader, train_path


def load_test_loader(config, device):
    data_config = config["data"]
    testing = config["testing"]
    U_test, F_data_test, A_test, aBC_test, test_path = load_darcy_h5(
        data_config["testing_path"],
        data_config.get("test_file"),
        data_config.get("a_key", "a"),
        data_config.get("f_key", "f"),
        data_config.get("u_key", "pp"),
        data_config.get("aBC_key", "aBc"),
    )
    test_u, test_f, test_a, aBC_test, test_dataset, test_loader = make_loader(
        U_test, F_data_test, A_test, aBC_test, testing["batch_size"], shuffle=False, device=device, include_aBC=False
    )
    return test_u, test_f, test_a, aBC_test, test_dataset, test_loader, test_path


def relative_l2_error(u_pred, u_true):
    return torch.mean(
        torch.linalg.vector_norm(u_pred - u_true, dim=(1,2,3)) /
        (torch.linalg.vector_norm(u_true, dim=(1,2,3)) + 1e-12)
    )
