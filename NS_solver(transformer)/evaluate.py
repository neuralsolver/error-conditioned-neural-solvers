from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from data_utils import (
    get_device,
    load_ns_tensors,
    make_forcing,
    make_loader,
    relative_l2,
    set_seed,
    sigma_model0,
    sigma_model1,
)
from ns_model import build_model
from pde_utils import build_pde_loss, residual_map_vorticity2d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ns_kf.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f_cfg:
        cfg = yaml.safe_load(f_cfg)

    set_seed(cfg.get("seed", 33))
    device = get_device(cfg.get("device", "auto"))
    print(f"Using device: {device}")

    data_cfg = cfg["data"]
    eval_cfg = cfg["evaluate"]
    model_cfg = cfg["model"]

    test_u_out, test_u_in = load_ns_tensors(
        data_cfg["test_path"],
        input_key=data_cfg.get("input_key", "a"),
        output_key=data_cfg.get("output_key", "u"),
    )
    print(f"test_u_out: {tuple(test_u_out.shape)}")
    print(f"test_u_in:  {tuple(test_u_in.shape)}")

    test_loader = make_loader(
        test_u_out,
        test_u_in,
        batch_size=eval_cfg.get("batch_size", 1),
        shuffle=False,
        num_workers=0,
    )

    f = make_forcing(
        data_cfg.get("resolution", test_u_in.shape[-1]),
        device=device,
        forcing_cfg=data_cfg.get("forcing"),
    )
    nu = eval_cfg.get("nu", cfg["train"].get("nu", 1e-4))
    dt = data_cfg.get("dt", 0.2)
    pde_loss = build_pde_loss(f=f, nu=nu, dt=dt)

    model0 = build_model(
        model_cfg.get("model0_in_channels", 1),
        model_cfg.get("model0_out_channels", test_u_out.shape[1]),
        model_cfg,
    ).to(device)
    model1 = build_model(
        model_cfg.get("model1_in_channels", 43),
        model_cfg.get("model1_out_channels", test_u_out.shape[1]),
        model_cfg,
    ).to(device)

    checkpoint_path = args.checkpoint or eval_cfg.get("checkpoint_path")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model0.load_state_dict(checkpoint["model0_state_dict"])
    model1.load_state_dict(checkpoint["model1_state_dict"])
    print(f"Loaded checkpoint: {checkpoint_path}")

    loss = nn.MSELoss()
    model0.eval()
    model1.eval()

    T = eval_cfg.get("rollout_steps", 12)
    sigma_T = eval_cfg.get("sigma_T", cfg["train"].get("rollout_steps", 5))
    step_size = eval_cfg.get("step_size", cfg["train"].get("step_size", 0.1))

    pred_list = []
    gt_list = []
    input_list = []
    mse_list = []
    rel_l2_list = []
    pde_list = []

    with torch.no_grad():
        for u_out_batch, u_in_batch in tqdm(test_loader, desc="evaluate"):
            test_u_out_batch = u_out_batch.to(device)
            test_u_in_batch = u_in_batch.to(device)
            B = test_u_in_batch.shape[0]

            u_pred_test = model0(test_u_in_batch, sigma_model0(B, device))

            for j in range(T):
                R_test = residual_map_vorticity2d(pde_loss, u_pred_test, f, nu, dt)
                input_test = torch.cat([test_u_in_batch, u_pred_test, R_test], dim=1)
                sigma = sigma_model1(j, sigma_T, B, device)
                u_correction_test = model1(input_test, sigma)
                u_pred_test = u_pred_test + step_size * u_correction_test

            R_final = residual_map_vorticity2d(pde_loss, u_pred_test, f, nu, dt)
            test_loss = loss(u_pred_test, test_u_out_batch)
            rel_l2 = relative_l2(u_pred_test, test_u_out_batch)
            pde_mse = torch.mean(R_final**2)

            pred_list.append(u_pred_test.detach().cpu())
            gt_list.append(test_u_out_batch.detach().cpu())
            input_list.append(test_u_in_batch.detach().cpu())
            mse_list.append(test_loss.detach().cpu())
            rel_l2_list.append(rel_l2.detach().cpu())
            pde_list.append(pde_mse.detach().cpu())

    u_pred_test = torch.cat(pred_list, dim=0)
    test_u_eval = torch.cat(gt_list, dim=0)
    test_x_eval = torch.cat(input_list, dim=0)
    mse = torch.stack(mse_list).mean()
    rel_l2_eval = torch.stack(rel_l2_list).mean()
    pde_eval = torch.stack(pde_list).mean()

    print(f"MSE loss: {mse.item():.6e}")
    print(f"Relative L2 error: {rel_l2_eval.item():.2e}")
    print(f"PDE residual: {pde_eval.item():.2e}")

    save_path = Path(eval_cfg.get("save_path", "results/ns_kf_eval.pt"))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "u_pred_test": u_pred_test,
            "test_u": test_u_eval,
            "test_u_in": test_x_eval,
            "mse": mse.item(),
            "rel_l2": rel_l2_eval.item(),
            "pde": pde_eval.item(),
            "config": cfg,
            "checkpoint_path": checkpoint_path,
        },
        save_path,
    )
    print(f"Saved results: {save_path}")


if __name__ == "__main__":
    main()
