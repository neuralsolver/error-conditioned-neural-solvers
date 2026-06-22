import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data_utils import (
    get_device,
    load_ns_tensors,
    make_forcing,
    make_loader,
    set_seed,
    sigma_model0,
    sigma_model1,
)
from ns_model import build_model
from pde_utils import build_pde_loss, residual_map_vorticity2d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ns_kf.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f_cfg:
        cfg = yaml.safe_load(f_cfg)

    set_seed(cfg.get("seed", 33))
    device = get_device(cfg.get("device", "auto"))
    print(f"Using device: {device}")

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]

    train_u_out, train_u_in = load_ns_tensors(
        data_cfg["train_path"],
        input_key=data_cfg.get("input_key", "a"),
        output_key=data_cfg.get("output_key", "u"),
    )
    print(f"train_u_out: {tuple(train_u_out.shape)}")
    print(f"train_u_in:  {tuple(train_u_in.shape)}")

    train_loader = make_loader(
        train_u_out,
        train_u_in,
        batch_size=train_cfg.get("batch_size", 30),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 0),
    )

    f = make_forcing(
        data_cfg.get("resolution", train_u_in.shape[-1]),
        device=device,
        forcing_cfg=data_cfg.get("forcing"),
    )
    pde_loss = build_pde_loss(f=f, nu=train_cfg.get("nu", 1e-4), dt=data_cfg.get("dt", 0.2))

    model0 = build_model(
        model_cfg.get("model0_in_channels", 1),
        model_cfg.get("model0_out_channels", train_u_out.shape[1]),
        model_cfg,
    ).to(device)
    model1 = build_model(
        model_cfg.get("model1_in_channels", 43),
        model_cfg.get("model1_out_channels", train_u_out.shape[1]),
        model_cfg,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model0.parameters()) + list(model1.parameters()),
        lr=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    loss = nn.MSELoss()
    start_epoch = 0

    resume_path = train_cfg.get("resume_path")
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model0.load_state_dict(checkpoint["model0_state_dict"])
        model1.load_state_dict(checkpoint["model1_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        print(f"Resumed from {resume_path} at epoch {start_epoch}")

    n_epochs = train_cfg.get("n_epochs", 200)
    T = train_cfg.get("rollout_steps", 5)
    step_size = train_cfg.get("step_size", 0.1)
    nu = train_cfg.get("nu", 1e-4)
    dt = data_cfg.get("dt", 0.2)
    print_every = train_cfg.get("print_every", 10)
    Loss = []

    for i in range(start_epoch, n_epochs):
        model0.train()
        model1.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"epoch {i}", leave=False)
        for u_out_batch, u_in_batch in pbar:
            u_out_batch = u_out_batch.to(device)
            u_in_batch = u_in_batch.to(device)
            B = u_out_batch.shape[0]

            optimizer.zero_grad()

            u_pred = model0(u_in_batch, sigma_model0(B, device))
            data_loss = 0.0

            for j in range(T):
                with torch.no_grad():
                    R = residual_map_vorticity2d(pde_loss, u_pred, f, nu, dt)
                    input_data = torch.cat([u_in_batch, u_pred, R], dim=1)

                u_correction = model1(input_data, sigma_model1(j, T, B, device))
                u_next = u_pred + step_size * u_correction
                data_loss = data_loss + loss(u_next, u_out_batch)
                u_pred = u_next.detach()

            data_loss = data_loss / T
            data_loss.backward()
            optimizer.step()

            epoch_loss += data_loss.item()
            pbar.set_postfix(loss=f"{data_loss.item():.6f}")

        Loss.append(epoch_loss)
        if i % print_every == 0:
            print(f"Epoch {i}: train loss {epoch_loss:.6f}")

    checkpoint_path = Path(train_cfg.get("checkpoint_path", "checkpoints/ns_kf_video_pde.pth"))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": n_epochs - 1,
        "model0_state_dict": model0.state_dict(),
        "model1_state_dict": model1.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "Loss": Loss,
        "config": cfg,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
