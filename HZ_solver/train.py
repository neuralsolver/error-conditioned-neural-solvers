import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data_utils import get_device, load_config, load_train_loader, set_seed
from models import FNO_CNN
from pde_losses import PDELossHZ


def build_models(config, device):
    model_config = config["model"]
    model0 = FNO_CNN(
        in_channels=1,
        out_channels=1,
        hidden_channels=model_config["hidden_channels"],
        n_modes=tuple(model_config["n_modes"]),
        n_layers=model_config["n_layers"],
    ).to(device)
    model1 = FNO_CNN(
        in_channels=3,
        out_channels=1,
        hidden_channels=model_config["hidden_channels"],
        n_modes=tuple(model_config["n_modes"]),
        n_layers=model_config["n_layers"],
    ).to(device)
    return model0, model1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/hz.json")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed"))
    device = get_device(config.get("device", "auto"))

    train_u, train_f, train_dataset, train_loader, train_path = load_train_loader(config, device)
    print(f"Loaded train data from {train_path}")
    print(f"train_u shape: {tuple(train_u.shape)}, train_f shape: {tuple(train_f.shape)}")

    loss = nn.MSELoss()
    pde_loss_grid = PDELossHZ()
    model0, model1 = build_models(config, device)

    training = config["training"]
    pde_config = config["pde"]
    k = pde_config["k"]
    lamb = pde_config.get("lamb", 0)

    resume_checkpoint = training.get("resume_checkpoint")
    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model0.load_state_dict(checkpoint["model0_state_dict"])
        model1.load_state_dict(checkpoint["model1_state_dict"])

    optimizer = torch.optim.AdamW(
        list(model0.parameters()) + list(model1.parameters()),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )

    if resume_checkpoint and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    Loss = []
    n_epochs = training["n_epochs"]
    T = training["T"]

    for i in range(n_epochs):
        model0.train()
        model1.train()

        epoch_loss = 0.0

        for u_batch, f_batch in train_loader:
            u_batch = u_batch.to(device)
            f_batch = f_batch.to(device)

            optimizer.zero_grad()

            u_pred = model0(f_batch)
            data_loss = 0.0

            for j in range(T):
                with torch.no_grad():
                    l_pde = pde_loss_grid(u_pred, f_batch, k, lamb)
                    input_data = torch.cat([f_batch, u_pred, l_pde], dim=1)

                u_correction = model1(input_data)
                u_next = u_pred + training["correction_step"] * u_correction
                data_loss += loss(u_pred, u_batch)
                u_pred = u_next.detach()
                   
            data_loss = data_loss / T
            data_loss.backward()
            optimizer.step()

            epoch_loss += data_loss.item()

        Loss.append(epoch_loss)

        if i % training["print_every"] == 0:
            print(f"Epoch {i}: loss {epoch_loss:.6f}")

    save_checkpoint = Path(training["save_checkpoint"])
    save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": i,
        "model0_state_dict": model0.state_dict(),
        "model1_state_dict": model1.state_dict(),
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
