import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data_utils import get_device, load_config, load_test_loader, relative_l2_error, set_seed, make_forcing, sigma_model0, sigma_model1
from model import build_model
from pde_losses import PDELossNS2d

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/ns.json")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 33))
    device = get_device(config.get("device", "auto"))

    test_u_out, test_u_in, test_dataset, test_loader, test_path = load_test_loader(config)
    print(f"Loaded test data from {test_path}")
    print(f"test_u_out shape: {tuple(test_u_out.shape)}, test_u_in shape: {tuple(test_u_in.shape)}")

    loss = nn.MSELoss()
    pde_loss_grid = PDELossNS2d()
    model0 = build_model(config["model0"], device)
    model1 = build_model(config["model1"], device)

    training = config["training"]
    testing = config["testing"]
    pde_config = config["pde"]
    f = make_forcing(test_u_out.shape[-1], device, pde_config.get("forcing", "default"))

    checkpoint_path = args.checkpoint or testing["checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model0.load_state_dict(checkpoint["model0_state_dict"])
    model1.load_state_dict(checkpoint["model1_state_dict"])

    test = []
    pde = []

    model0.eval()
    model1.eval()
    
    with torch.no_grad():
        test_u_out = test_u_out.to(device)
        test_u_in = test_u_in.to(device)
        B = test_u_in.shape[0]

        u_pred_test = model0(test_u_in, sigma_model0(B, device))
        
        for j in range(testing["T"]):
            R_test = pde_loss_grid(u_pred_test, f, pde_config["nu"], pde_config.get("dt", 0.2))
            input_test = torch.cat([test_u_in, u_pred_test, R_test], dim=1)
            sigma = sigma_model1(j, training["T"], B, device)
                
            u_correction_test = model1(input_test, sigma)
            u_pred_test = u_pred_test + testing["correction_step"] * u_correction_test
            test_loss = loss(u_pred_test, test_u_out)
            pde_loss = torch.mean(pde_loss_grid(u_pred_test, f, pde_config["nu"], pde_config.get("dt", 0.2))**2)
            test.append(test_loss.item())
            pde.append(pde_loss.item())
                
            print(f"T_test {j + 1}: reconstruction loss {test_loss.item():.6e}, pde loss {pde_loss.item():.6e}")

        rel_l2 = relative_l2_error(u_pred_test, test_u_out)
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
