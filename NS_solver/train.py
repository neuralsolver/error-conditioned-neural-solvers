import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data_utils import get_device, load_config, load_train_loader, set_seed
from models import FNO_CNN
from ns_utils import make_forcing
from pde_losses import PDELossNS2d


def build_model(config, device):
    model_config = config["model"]
    model0 = FNO_CNN(
        in_channels=model_config.get("in_channels", 1),
        out_channels=model_config["out_channels"],
        hidden_channels=model_config["hidden_channels"],
        n_modes=tuple(model_config["n_modes"]),
        n_layers=model_config["n_layers"],
    ).to(device)
    return model0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/ns.json")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 33))
    device = get_device(config.get("device", "auto"))

    train_u_out, train_u_in, train_dataset, train_loader, train_path = load_train_loader(config, device)
    print(f"Loaded train data from {train_path}")
    print(f"train_u_out shape: {tuple(train_u_out.shape)}, train_u_in shape: {tuple(train_u_in.shape)}")

    loss = nn.MSELoss()
    pde_loss_grid = PDELossNS2d()
    model0 = build_model(config, device)

    training = config["training"]
    pde_config = config["pde"]
    f = make_forcing(train_u_out.shape[-1], device, pde_config.get("forcing", "default"))

    resume_checkpoint = training.get("resume_checkpoint")
    resume_checkpoint = None if resume_checkpoint in [None, "", "none", "None"] else resume_checkpoint
    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        if "model0_state_dict" in checkpoint:
            model0.load_state_dict(checkpoint["model0_state_dict"])
        else:
            model0.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {resume_checkpoint}")
    else:
        print("Training from scratch")

    optimizer = torch.optim.AdamW(
        model0.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )

    if resume_checkpoint and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    Loss = []
    Test_Loss = []
    n_epochs = training["n_epochs"]

    for i in range(n_epochs):
        model0.train()
        epoch_loss = 0.0

        for u_out_batch, u_in_batch in train_loader:
            u_out_batch = u_out_batch.to(device)
            u_in_batch = u_in_batch.to(device)
            optimizer.zero_grad()

            u_pred = model0(u_in_batch)
            data_loss = (
                loss(u_pred, u_out_batch)
                + training["pde_weight"]
                * torch.mean(
                    pde_loss_grid(
                        u_pred,
                        f,
                        training.get("pde_train_nu", pde_config["nu"]),
                        pde_config.get("dt", 0.2),
                    )**2
                )
            )

            data_loss.backward()
            optimizer.step()

            epoch_loss += data_loss.item()

        Loss.append(epoch_loss)

        if i % training["print_every"] == 0:
            print(f"Epoch {i}: train loss {epoch_loss:.6f}")

    save_checkpoint = Path(training["save_checkpoint"])
    save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": i,
        "model0_state_dict": model0.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }
    torch.save(checkpoint, save_checkpoint)
    print(f"Saved checkpoint to {save_checkpoint}")

    save_loss = training.get("save_loss")
    if save_loss:
        save_loss = Path(save_loss)
        save_loss.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_loss, np.array(Loss))
        print(f"Saved training loss to {save_loss}")


if __name__ == "__main__":
    main()
