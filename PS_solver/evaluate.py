import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data_utils import get_device, load_config, load_test_loader, relative_l2_error, set_seed
from models import FNO_CNN
from pde_losses import PDELossPS


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


def add_structured_noise(x, noise_level=30, smooth_kernel=9):
    """
    x: (B,1,H,W)
    noise_level: overall noise strength
    smooth_kernel: controls frequency split
    """
    torch.manual_seed(0)
    noise = torch.randn_like(x)

    noise_low = F.avg_pool2d(noise, kernel_size=smooth_kernel, stride=1, padding=smooth_kernel//2)

    noise_high = noise - noise_low

    mask = torch.abs(x)
    mask = mask / (mask.amax(dim=(2,3), keepdim=True) + 1e-8)

    x_noisy = x + noise_level * noise_high * (0.3 + 0.7 * mask)

    return x_noisy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/ps.json")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 33))
    device = get_device(config.get("device", "auto"))

    test_u, test_f, test_dataset, test_loader, test_path = load_test_loader(config, device)
    print(f"Loaded test data from {test_path}")
    print(f"test_u shape: {tuple(test_u.shape)}, test_f shape: {tuple(test_f.shape)}")

    loss = nn.MSELoss()

    pde_config = config["pde"]
    loss_type = pde_config.get("loss_type", "ps").lower()
    
    if loss_type in ["hz", "helmholtz"]:
        pde_loss_grid = PDELossHZ()
        k = pde_config["k"]
        lamb = pde_config.get("lamb", 0)
    elif loss_type in ["ps", "poisson"]:
        pde_loss_grid = PDELossPS()
        k = pde_config["k"]
    else:
        raise ValueError(f"Unknown pde type: {loss_type}")
        
    model0, model1 = build_models(config, device)
    testing = config["testing"]
   

    checkpoint_path = args.checkpoint or testing["checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model0_state_dict" in checkpoint:
        model0.load_state_dict(checkpoint["model0_state_dict"])
    else:
        model0.load_state_dict(checkpoint["model_state_dict"])
    if "model1_state_dict" in checkpoint:
        model1.load_state_dict(checkpoint["model1_state_dict"])

    test = []
    pde = []

    model0.eval()
    model1.eval()
    with torch.no_grad():
        if testing.get("use_structured_noise", True):
            test_f_noise = add_structured_noise(
                test_f,
                noise_level=testing.get("noise_level", 30),
                smooth_kernel=testing.get("smooth_kernel", 9),
            )
        else:
            test_f_noise = test_f

        u_pred_test = model0(test_f_noise)

        for j in range(testing["T"]):
            l_pde_test = pde_loss_grid(u_pred_test, test_f_noise, k)
            input_test = torch.cat([test_f_noise, u_pred_test, l_pde_test], dim=1)
            u_correction_test = model1(input_test)

            u_pred_test = u_pred_test + testing["correction_step"] * u_correction_test

            test_loss = loss(u_pred_test, test_u)
            pde_loss = torch.mean(pde_loss_grid(u_pred_test, test_f_noise, k)**2)
            test.append(test_loss.item())
            pde.append(pde_loss.item())

            print(f"T_test {j + 1}: reconstruction loss {test_loss.item():.6e}, pde loss {pde_loss.item():.6e}")

        rel_l2 = relative_l2_error(u_pred_test, test_u)
        print(f"Relative L2 error: {rel_l2.item():.2e}")

    save_reconstruction_loss = testing.get("save_reconstruction_loss")
    if save_reconstruction_loss:
        save_reconstruction_loss = Path(save_reconstruction_loss)
        save_reconstruction_loss.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_reconstruction_loss, np.array(test))
        print(f"Saved reconstruction loss to {save_reconstruction_loss}")

    save_pde_loss = testing.get("save_pde_loss")
    if save_pde_loss:
        save_pde_loss = Path(save_pde_loss)
        save_pde_loss.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_pde_loss, np.array(pde))
        print(f"Saved PDE loss to {save_pde_loss}")


if __name__ == "__main__":
    main()
