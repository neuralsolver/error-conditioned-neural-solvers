import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(".").resolve()))

from model import build_model, apply_cnn_ff
from data import load_data, build_forcing
from pde import residual_map_vorticity2d
from utils import device, sigma_model0, sigma_model1


CHECKPOINT_PATH = 'checkpoints/128_5(V-CNN_t40).pth'


def main():
    train_u_in, train_u_out, test_u_in, test_u_out = load_data()
    f = build_forcing(N=128)

    loss = nn.MSELoss()

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model0 = build_model(1, 40).to(device)
    model1 = build_model(79, 40).to(device)
    model0 = apply_cnn_ff(model0)
    model1 = apply_cnn_ff(model1)
    model0.load_state_dict(checkpoint['model0_state_dict'])
    model1.load_state_dict(checkpoint['model1_state_dict'])

    test = []
    pde = []
    T = 5
    with torch.no_grad():
        B = test_u_in.shape[0]
        u_pred_test = model0(test_u_in.to(device), sigma_model0(B))

        for j in range(100):
            R_test = residual_map_vorticity2d(u_pred_test, f, 5e-4)
            input_test = torch.cat([test_u_in.to(device), u_pred_test, R_test], dim=1)

            sigma = sigma_model1(j, T, B)
            u_correction_test = model1(input_test, sigma)
            u_pred_test = u_pred_test + 0.009 * u_correction_test

            test_loss = loss(u_pred_test, test_u_out.to(device))
            R_test = residual_map_vorticity2d(u_pred_test, f, 5e-4)
            test.append(test_loss.item())
            pde.append(torch.mean(R_test ** 2).item())
            print(test_loss)

    plt.plot(pde)
    plt.show()

    u_plot = u_pred_test.detach().cpu().numpy()[15, 19]
    u_true = test_u_out.detach().cpu().numpy()[15, 19]

    vmin = u_true.min()
    vmax = u_true.max()

    plt.imshow(u_true, cmap='viridis', origin='lower')
    plt.colorbar(label="u(x,y)")
    plt.title("True NS")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.show()

    plt.imshow(u_plot, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar(label="u(x,y)")
    plt.title("Predicted NS (PINO)")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.show()

    diff = abs(u_plot - u_true)
    plt.figure(figsize=(6, 5))
    plt.imshow(diff, cmap='viridis', origin='lower')
    plt.colorbar(label="u(x,y)")
    plt.title("Absolute error (64x64)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    relative_loss = np.linalg.norm(u_plot - u_true) / np.linalg.norm(u_true)
    print(relative_loss)

    return u_pred_test, test_u_out, test, pde


if __name__ == "__main__":
    main()
