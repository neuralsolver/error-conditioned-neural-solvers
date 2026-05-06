import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data_utils import get_device, load_config, load_nonlinear_hz_data, set_seed
from initializations import build_initialization
from models import FNO_CNN
from numerical_methods import solve_numerical
from pde_losses import PDELossHZ


def build_model(model_config, device):
    return FNO_CNN(
        in_channels=model_config["in_channels"],
        out_channels=model_config["out_channels"],
        hidden_channels=model_config["hidden_channels"],
        n_modes=tuple(model_config["n_modes"]),
        n_layers=model_config["n_layers"],
    ).to(device)


def load_models(config, device):
    checkpoint_path = config["model"].get("checkpoint")
    if checkpoint_path in [None, "", "none", "None"]:
        return None, None

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model0 = build_model(config["model"]["model0"], device)
    model1 = build_model(config["model"]["model1"], device)
    model0.load_state_dict(checkpoint["model0_state_dict"])
    model1.load_state_dict(checkpoint["model1_state_dict"])
    model0.eval()
    model1.eval()
    return model0, model1


def run_ours(f_tensor, u_tensor, init_data, model1, pde_loss_grid, pde_config, ours_config, verbose=True):
    if model1 is None:
        raise ValueError("Ours requires model1 checkpoint.")

    T = ours_config["T"]
    correction_step = ours_config["correction_step"]
    k = pde_config["k"]
    lamb = pde_config["lamb"]

    rel_l2_all = []
    pde_all = []
    recon_mse_all = []
    time_all = []
    pred_all = []

    for i in range(f_tensor.shape[0]):
        print(f"\n===== Ours sample {i+1}/{f_tensor.shape[0]} =====")
        u_pred = torch.from_numpy(init_data[i]).float().to(f_tensor.device)[None, None, :, :]
        solve_time = 0.0
        sample_rel_l2 = []
        sample_pde = []
        sample_recon = []
        sample_time = []

        with torch.no_grad():
            for j in range(T):
                l_pde = pde_loss_grid(u_pred, f_tensor[i:i+1], k, lamb)
                rel_l2 = torch.mean(
                    torch.linalg.vector_norm(u_pred - u_tensor[i:i+1], dim=(1, 2, 3)) /
                    (torch.linalg.vector_norm(u_tensor[i:i+1], dim=(1, 2, 3)) + 1e-12)
                )
                pde_mse = torch.mean(l_pde**2)
                recon_mse = torch.mean((u_pred - u_tensor[i:i+1])**2)

                sample_rel_l2.append(rel_l2.item())
                sample_pde.append(pde_mse.item())
                sample_recon.append(recon_mse.item())
                sample_time.append(solve_time)

                if verbose:
                    print(
                        f"iter {j:03d}: rel L2 = {rel_l2.item():.3e}, "
                        f"PDE MSE = {pde_mse.item():.3e}, time = {solve_time:.4f}s"
                    )

                t0 = time.perf_counter()
                input_test = torch.cat([f_tensor[i:i+1], u_pred, l_pde], dim=1)
                u_correction = model1(input_test)
                u_pred = u_pred + correction_step * u_correction
                if u_pred.is_cuda:
                    torch.cuda.synchronize()
                solve_time += time.perf_counter() - t0

        pred_all.append(u_pred.squeeze(0).squeeze(0).detach().cpu().numpy())
        rel_l2_all.append(sample_rel_l2)
        pde_all.append(sample_pde)
        recon_mse_all.append(sample_recon)
        time_all.append(sample_time)

    return (
        np.array(pred_all),
        np.array(pde_all),
        np.array(recon_mse_all),
        np.array(rel_l2_all),
        np.array(time_all),
    )


def save_result(save_dir, prefix, method, init_type, pred, pde, recon, rel_l2, times):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}_{method}_{init_type}"
    np.save(save_dir / f"{stem}_pred.npy", pred)
    np.save(save_dir / f"{stem}_pde.npy", pde)
    np.save(save_dir / f"{stem}_recon_mse.npy", recon)
    np.save(save_dir / f"{stem}_rel_l2.npy", rel_l2)
    np.save(save_dir / f"{stem}_time.npy", times)
    print(f"Saved results with prefix {save_dir / stem}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/numerical.json")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 33))
    device = get_device(config.get("device", "auto"))

    f_data, u_data, data_path = load_nonlinear_hz_data(config)
    print(f"Loaded test data from {data_path}")
    print(f"f_data shape: {f_data.shape}, u_data shape: {u_data.shape}")

    pde_loss_grid = PDELossHZ()
    model0, model1 = load_models(config, device)

    f_tensor = torch.from_numpy(f_data).float().unsqueeze(1).to(device)
    u_tensor = torch.from_numpy(u_data).float().unsqueeze(1).to(device)

    methods = config["experiment"]["methods"]
    initializations = config["experiment"]["initializations"]
    save_dir = config["experiment"]["save_dir"]
    prefix = config["experiment"].get("save_prefix", "numerical")
    pde_config = config["pde"]
    numerical_config = config["numerical"]

    for init_type in initializations:
        print(f"\n######## Initialization: {init_type} ########")
        init_data = build_initialization(
            init_type,
            f_tensor,
            model0,
            config.get("initialization", {}),
            numerical_config["S"],
        )

        for method in methods:
            print(f"\n======== Method: {method}, initialization: {init_type} ========")
            if method.lower() == "ours":
                pred, pde, recon, rel_l2, times = run_ours(
                    f_tensor,
                    u_tensor,
                    init_data,
                    model1,
                    pde_loss_grid,
                    pde_config,
                    config["ours"],
                )
            else:
                N = f_data.shape[0]
                max_iter = numerical_config["max_iter"]
                pred = np.zeros_like(u_data)
                pde = np.full((N, max_iter), np.nan)
                recon = np.full((N, max_iter), np.nan)
                rel_l2 = np.full((N, max_iter), np.nan)
                times = np.full((N, max_iter), np.nan)

                for i in range(N):
                    print(f"\n===== Sample {i+1}/{N} =====")
                    psi_full, residual_history, recon_history, rel_l2_history, time_history = solve_numerical(
                        f_full=f_data[i],
                        u_gt_full=u_data[i],
                        init_full=init_data[i],
                        pde_loss_grid=pde_loss_grid,
                        device=device,
                        solver_type=method,
                        S=numerical_config["S"],
                        k=pde_config["k"],
                        lamb=pde_config["lamb"],
                        max_iter=max_iter,
                        tol=numerical_config["tol"],
                        lr=numerical_config.get("lr", 1e-6),
                    )
                    pred[i] = psi_full
                    pde[i, :len(residual_history)] = residual_history
                    recon[i, :len(recon_history)] = recon_history
                    rel_l2[i, :len(rel_l2_history)] = rel_l2_history
                    times[i, :len(time_history)] = time_history

            save_result(save_dir, prefix, method, init_type, pred, pde, recon, rel_l2, times)


if __name__ == "__main__":
    main()
