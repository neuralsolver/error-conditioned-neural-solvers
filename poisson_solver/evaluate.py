"""
Iterative evaluation of a trained two-model Poisson solver.

At test time:
  u_pred = model0(f)
  for t in range(T_test):
      l_pde  = PDELossPS(u_pred, f, k_test)
      delta  = model1([f, u_pred, l_pde])
      u_pred = u_pred + step_size * delta

Usage:
    python evaluate.py --config configs/poisson.yaml
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import yaml

from models import FNO_CNN, UNet
from pde_losses import PDELossPS


def load_test_data(cfg, device):
    data = scipy.io.loadmat(cfg['data']['test_path'])
    n = cfg['data']['n_test']
    U = torch.from_numpy(data['psi_data'][:n]).float().unsqueeze(1).to(device)
    F = torch.from_numpy(data['f_data'][:n]).float().unsqueeze(1).to(device)
    return U, F


def build_models(cfg, device):
    mc = cfg['model']
    if mc['type'] == 'FNO_CNN':
        def make():
            return FNO_CNN(
                in_channels=mc['in_channels'],
                out_channels=mc['out_channels'],
                hidden_channels=mc['hidden_channels'],
                n_modes=tuple(mc['n_modes']),
                n_layers=mc['n_layers'],
            ).to(device)
        return make(), make()
    elif mc['type'] == 'UNet':
        def make():
            return UNet(n_channels=mc['in_channels'], n_classes=mc['out_channels']).to(device)
        return make(), make()
    else:
        raise ValueError(f"Unknown model type: {mc['type']}")


def evaluate(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ic = cfg['inference']

    model0, model1 = build_models(cfg, device)
    checkpoint = torch.load(ic['checkpoint'], map_location=device, weights_only=False)
    model0.load_state_dict(checkpoint['model0_state_dict'])
    model1.load_state_dict(checkpoint['model1_state_dict'])
    model0.eval()
    model1.eval()

    test_u, test_f = load_test_data(cfg, device)
    criterion = nn.MSELoss()
    pde_loss_fn = PDELossHZ()

    k = ic['k']
    step_size = ic['step_size']
    T = ic['T_test']

    test_losses = []
    pde_residuals = []

    with torch.no_grad():
        # Initial prediction
        u_pred = model0(test_f)

        for t in range(T):
            l_pde = pde_loss_fn(u_pred, test_f, k)
            u_correction = model1(torch.cat([test_f, u_pred, l_pde], dim=1))
            u_pred = u_pred + step_size * u_correction

            l_pde = pde_loss_fn(u_pred, test_f, k)
            test_losses.append(criterion(u_pred, test_u).item())
            pde_residuals.append(torch.mean(l_pde ** 2).cpu().item())
            print(f"T={t+1:3d} | recon loss: {test_losses[-1]:.6f} | PDE residual: {pde_residuals[-1]:.6f}")

    # Save results
    os.makedirs(ic['results_dir'], exist_ok=True)
    np.save(os.path.join(ic['results_dir'], 'test_loss.npy'), test_losses)
    np.save(os.path.join(ic['results_dir'], 'pde_residual.npy'), pde_residuals)

    # --- Plot 1: convergence curves ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(test_losses)
    axes[0].set_xlabel('T_test')
    axes[0].set_ylabel('Reconstruction Loss')
    axes[0].set_title(f'Helmholtz (train k={cfg["pde"]["k"]}, test k={k})')
    axes[0].grid()

    axes[1].plot(np.log(pde_residuals))
    axes[1].set_xlabel('T_test')
    axes[1].set_ylabel('log(PDE Residual)')
    axes[1].set_title(f'Helmholtz (train k={cfg["pde"]["k"]}, test k={k})')
    axes[1].grid()

    plt.tight_layout()
    plt.savefig(os.path.join(ic['results_dir'], 'convergence.png'), dpi=150)
    plt.show()

    # --- Plot 2: u_pred / u_true / absolute error / PDE residual (sample idx 13) ---
    idx = ic.get('vis_sample_idx', 13)

    u_plot = u_pred.detach().cpu().numpy()[idx, 0]
    u_true = test_u.detach().cpu().numpy()[idx, 0]
    vmin, vmax = u_true.min(), u_true.max()

    l_pde_vis = pde_loss_fn(u_pred, test_f, k)
    l_pde_vis = l_pde_vis.detach().cpu().numpy()[idx, 0]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    im0 = axes[0].imshow(u_plot, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    fig.colorbar(im0, ax=axes[0], label='u(x,y)')
    axes[0].set_title('Predicted u')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')

    im1 = axes[1].imshow(u_true, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    fig.colorbar(im1, ax=axes[1], label='u(x,y)')
    axes[1].set_title('True u')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')

    diff = np.abs(u_plot - u_true)
    im2 = axes[2].imshow(diff, cmap='viridis', origin='lower')
    fig.colorbar(im2, ax=axes[2], label='|error|')
    axes[2].set_title('Absolute Error')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')

    im3 = axes[3].imshow(l_pde_vis, cmap='viridis', origin='lower')
    fig.colorbar(im3, ax=axes[3], label='residual')
    axes[3].set_title(f'PDE Residual (k={k})')
    axes[3].set_xlabel('x')
    axes[3].set_ylabel('y')

    relative_error = np.linalg.norm(u_plot - u_true) / np.linalg.norm(u_true)
    fig.suptitle(f'Sample {idx} — Relative Error: {relative_error:.4e}', fontsize=13)

    plt.tight_layout()
    plt.savefig(os.path.join(ic['results_dir'], 'visualization.png'), dpi=150)
    plt.show()
    print(f"Results saved to {ic['results_dir']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/poisson.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    evaluate(cfg)
