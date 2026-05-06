import json
import random
from pathlib import Path

import numpy as np
import scipy.io
import torch


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


def load_nonlinear_hz_data(config):
    data_config = config["data"]
    mat_path = find_mat_file(data_config["testing_path"], data_config.get("test_file"))
    data = scipy.io.loadmat(mat_path)
    f_data = data[data_config.get("f_key", "f_data")]
    u_data = data[data_config.get("u_key", "psi_data")]

    sample_start = data_config.get("sample_start", 0)
    sample_end = data_config.get("sample_end")
    f_data = f_data[sample_start:sample_end]
    u_data = u_data[sample_start:sample_end]

    return f_data, u_data, mat_path
