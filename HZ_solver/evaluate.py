import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data_utils import get_device, load_config, load_test_loader, relative_l2_error, set_seed
from models import FNO_CNN
from pde_losses import PDELossHZ, PDELossPS


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
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed"))
    device = get_device(config.get("device", "auto"))

    test_u, test_f, test_dataset, test_loader, test_path = load_test_loader(config, device)
    print(f"Loaded test data from {test_path}")
    print(f"test_u shape: {tuple(test_u.shape)}, test_f shape: {tuple(test_f.shape)}")

    loss = nn.MSELoss()

    pde_config = config["pde"]
    loss_type = pde_config.get("loss_type", "hz").lower()
    
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
    model0.load_state_dict(checkpoint["model0_state_dict"])
    model1.load_state_dict(checkpoint["model1_state_dict"])

    test = []
    pde = []

    model0.eval()
    model1.eval()
    with torch.no_grad():
        u_pred_test = model0(test_f)

        for j in range(testing["T"]):
            
            if loss_type in ["hz", "helmholtz"]:
                l_pde_test = pde_loss_grid(u_pred_test, test_f, k, lamb)
            else:
                l_pde_test = pde_loss_grid(u_pred_test, test_f, k)
                
            input_test = torch.cat([test_f, u_pred_test, l_pde_test], dim=1)
            u_correction_test = model1(input_test)

            u_pred_test = u_pred_test + testing["correction_step"] * u_correction_test

            test_loss = loss(u_pred_test, test_u)
            if loss_type in ["hz", "helmholtz"]:
                pde_loss = torch.mean(pde_loss_grid(u_pred_test, test_f, k, lamb)**2)
            else:
                pde_loss = torch.mean(pde_loss_grid(u_pred_test, test_f, k)**2)
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
