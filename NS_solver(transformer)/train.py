import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data_utils import get_device, load_config, load_train_loader, set_seed, make_forcing, sigma_model0, sigma_model1
from model import build_model
from pde_losses import PDELossNS2d

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/ns.json")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 33))
    device = get_device(config.get("device", "auto"))

    train_u_out, train_u_in, train_dataset, train_loader, train_path = load_train_loader(config)
    print(f"Loaded train data from {train_path}")
    print(f"train_u_out shape: {tuple(train_u_out.shape)}, train_u_in shape: {tuple(train_u_in.shape)}")

    loss = nn.MSELoss()
    pde_loss_grid = PDELossNS2d()
    model0 = build_model(cfg["model0"], device)
    model1 = build_model(cfg["model1"], device)

    training = config["training"]
    pde_config = config["pde"]
    f = make_forcing(train_u_out.shape[-1], device, pde_config.get("forcing", "default"))
    
    resume_checkpoint = training.get("resume_checkpoint")
    resume_checkpoint = None if resume_checkpoint in [None, "", "none", "None"] else resume_checkpoint
    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model0.load_state_dict(checkpoint["model0_state_dict"])
        model1.load_state_dict(checkpoint["model1_state_dict"])
        print(f"Loaded checkpoint from {resume_checkpoint}")
    else:
        print("Training from scratch")

    optimizer = torch.optim.AdamW(
        list(model0.parameters()) + list(model1.parameters()),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )

    if resume_checkpoint and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    Loss = []
    Test_Loss = []
    n_epochs = training["n_epochs"]
    T = training["T"]
    
    for i in range(n_epochs):
        model0.train()
        model1.train()
        epoch_loss = 0.0

        for u_out_batch, u_in_batch in train_loader:
            u_out_batch = u_out_batch.to(device)
            u_in_batch = u_in_batch.to(device)
            optimizer.zero_grad()
            
            u_pred = model0(u_in_batch, sigma_model0(B, device))
            data_loss = 0.0

            for j in range(T):
                with torch.no_grad():
                    R = pde_loss_grid(u_pred, f, pde_config["nu"], pde_config.get("dt", 0.2))
                    input_data = torch.cat([u_in_batch, u_pred, R], dim=1)

                u_correction = model1(input_data, sigma_model1(j, T, B, device))
                u_next = u_pred + training["correction_step"] * u_correction
                data_loss += loss(u_pred, u_out_batch)
                u_pred = u_next.detach()

            data_loss = data_loss / T
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
