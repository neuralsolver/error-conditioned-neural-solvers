"""
Architecture:
  model0: initial predictor  -- input: f  (1 channel)
  model1: iterative corrector -- input: [f, u_pred, l_pde] (3 channels)

Training loop (T steps per sample):
  u_pred = model0(f)
  for t in range(T):
      l_pde  = PDELossHZ(u_pred, f, k)
      delta  = model1([f, u_pred, l_pde])
      u_pred = u_pred + alpha * delta
      loss  += MSE(u_pred, u_true)
  loss /= T

Usage:
    python -m helmholtz_solver.train --config helmholtz_solver/configs/helmholtz.yaml
"""

import argparse
import os
import random

import numpy as np
import scipy.io
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from helmholtz_solver.models import FNO_CNN, UNet
from pde_losses import PDELossHZ


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_data(cfg):
    data = scipy.io.loadmat(cfg['data']['train_path'])
    U = torch.from_numpy(data['psi_data']).float().unsqueeze(1)
    F = torch.from_numpy(data['f_data']).float().unsqueeze(1)
    loader = DataLoader(
        TensorDataset(U, F),
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
    )
    return loader


def build_model(mc, device):
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


def train(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(cfg['training']['seed'])

    loader = load_data(cfg)
    model0 = build_model(cfg['model0'], device)
    model1 = build_model(cfg['model1'], device)

    optimizer = torch.optim.AdamW(
        list(model0.parameters()) + list(model1.parameters()),
        lr=cfg['training']['lr'],
        weight_decay=1e-4)
  
    criterion = nn.MSELoss()
    pde_loss_fn = PDELossHZ()

    T = cfg['training']['T']
    alpha = cfg['training']['alpha']
    k = cfg['pde']['k']
    n_epochs = cfg['training']['n_epochs']
    log_interval = cfg['training']['log_interval']

    os.makedirs(os.path.dirname(cfg['training']['save_path']), exist_ok=True)
    loss_history = []

    for epoch in range(n_epochs):
        model0.train()
        model1.train()
        epoch_loss = 0.0

        for u_batch, f_batch in loader:
            u_batch = u_batch.to(device)   # [B, 1, S, S]
            f_batch = f_batch.to(device)

            optimizer.zero_grad()

            # --- initial prediction ---
            #zeros = torch.zeros_like(u_batch)
            u_pred = model0(f_batch)

            # --- iterative correction ---
            data_loss = 0.0
            for _ in range(T):
                with torch.no_grad():
                    l_pde = pde_loss_fn(u_pred, f_batch, k)
                    input_data = torch.cat([f_batch, u_pred, l_pde], dim=1)
                  
                u_correction = model1(input_data)
                u_pred = u_pred + alpha * u_correction
                data_loss += criterion(u_pred, u_batch)

            data_loss = data_loss / T
            data_loss.backward()
            optimizer.step()
            epoch_loss += data_loss.item()

        loss_history.append(epoch_loss)
        if epoch % log_interval == 0:
            print(f"Epoch {epoch:5d} | loss: {epoch_loss:.6f}")

    checkpoint = {
        'epoch': epoch,
        'model0_state_dict': model0.state_dict(),
        'model1_state_dict': model1.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_history': loss_history,
    }
    torch.save(checkpoint, cfg['training']['save_path'])
    print(f"Checkpoint saved to {cfg['training']['save_path']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/helmholtz.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)
