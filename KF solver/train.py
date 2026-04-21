import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(".").resolve()))

from model import build_model, apply_cnn_ff
from data import load_data, build_forcing, build_train_loader
from pde_loss import residual_map_vorticity2d
from utils import device, sigma_model0, sigma_model1, set_seed


CHECKPOINT_PATH = 'checkpoints/128_5(V-CNN_t40).pth'


def main():
    print(f"Using device: {device}")

    train_u_in, train_u_out, test_u_in, test_u_out = load_data()
    f = build_forcing(N=128)
    train_loader = build_train_loader(train_u_out, train_u_in, batch_size=24)

    loss = nn.MSELoss()

    set_seed(33)

    #checkpoint = torch.load(CHECKPOINT_PATH,  weights_only=False)
    model0 = build_model(1, 40)
    #model0 = FNO_CNN(in_channels=1, out_channels=20, hidden_channels=64, n_modes=(20, 20), n_layers=4).to(device)
    #model1 = FNO_CNN(in_channels=39, out_channels=20, hidden_channels=64, n_modes=(20, 20), n_layers=4).to(device)
    model1 = build_model(79, 40)

    model0 = apply_cnn_ff(model0).to(device)
    model1 = apply_cnn_ff(model1).to(device)

    #model0.load_state_dict(checkpoint['model0_state_dict'])
    #model1.load_state_dict(checkpoint['model1_state_dict'])
    print(next(model0.parameters()).view(-1)[:5])
    print(next(model1.parameters()).view(-1)[:5])

    #optimizer = torch.optim.AdamW([{"params": model0.parameters(), "lr": 1e-4}])
    optimizer = torch.optim.AdamW(
        list(model0.parameters()) + list(model1.parameters()),
        lr=1e-4,
        weight_decay=1e-4,
    )
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    Loss = []
    Test_Loss = []
    n_epochs = 20000
    for i in range(n_epochs):
        model0.train()
        model1.train()
        epoch_loss = 0.0

        for u_out_batch, u_in_batch in train_loader:
            u_out_batch = u_out_batch.to(device)
            u_in_batch = u_in_batch.to(device)
            B = u_in_batch.shape[0]

            optimizer.zero_grad()

            u_pred = model0(u_in_batch, sigma_model0(B))

            data_loss = 0.0
            T = 5

            for j in range(T):
                with torch.no_grad():
                    R = residual_map_vorticity2d(u_pred, f, 5e-4)
                    input_data = torch.cat([u_in_batch, u_pred, R], dim=1)

                sigma = sigma_model1(j, T, B)
                u_correction = model1(input_data, sigma)
                u_pred = u_pred + 0.07 * u_correction
                data_loss += loss(u_pred, u_out_batch)

            data_loss = data_loss / T

            data_loss.backward()
            optimizer.step()

            epoch_loss += data_loss.item()

        if i % 10 == 0:
            print(f"Epoch {i}: train loss {epoch_loss:.6f}")

    checkpoint = {
        'epoch': i,
        'model0_state_dict': model0.state_dict(),
        'model1_state_dict': model1.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    torch.save(checkpoint, CHECKPOINT_PATH)


if __name__ == "__main__":
    main()
