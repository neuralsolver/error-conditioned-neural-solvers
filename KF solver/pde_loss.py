import math

import torch


def build_kolmogorov_forcing(
    resolution: int,
    wavenumber: int = 4,
    amplitude: float = -4.0,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
):
    grid = torch.linspace(0, 1, resolution + 1, device=device, dtype=dtype)[:-1]
    x, y = torch.meshgrid(grid, grid, indexing="ij")
    del x
    return amplitude * torch.cos(2 * math.pi * wavenumber * y)

def residual_map_vorticity2d(
    w: torch.Tensor,
    forcing: torch.Tensor,
    viscosity: float,
    dt: float = 0.1,
):
    if w.ndim != 4:
        raise ValueError(f"Expected w to have shape (B, T, H, W), got {tuple(w.shape)}")

    batch, time_steps, height, width = w.shape
    if height != width:
        raise ValueError("Only square spatial grids are supported")
    if time_steps < 3:
        raise ValueError("At least 3 time steps are required to form centered differences")

    device = w.device
    dtype = w.dtype
    forcing = forcing.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)

    k_max = math.floor(size / 2.0)
    base = torch.cat(
        (
            torch.arange(0, k_max, device=device, dtype=dtype),
            torch.arange(-k_max, 0, device=device, dtype=dtype),
        ),
        dim=0,
    )
    k_y = base.repeat(size, 1)
    k_x = k_y.transpose(0, 1)
    laplacian = 4 * (math.pi**2) * (k_x**2 + k_y**2)
    laplacian[0, 0] = 1.0

    wt = torch.empty_like(w)
    wt[:, 0] = (w[:, 1] - w[:, 0]) / dt
    wt[:, -1] = (w[:, -1] - w[:, -2]) / dt
    wt[:, 1:-1] = (w[:, 2:] - w[:, :-2]) / (2 * dt)

    wh = torch.fft.fft2(w)
    psi_h = wh / laplacian
    u = torch.fft.ifft2((1j * 2 * math.pi * k_y) * psi_h).real
    v = torch.fft.ifft2((-1j * 2 * math.pi * k_x) * psi_h).real
    wx = torch.fft.ifft2((1j * 2 * math.pi * k_x) * wh).real
    wy = torch.fft.ifft2((1j * 2 * math.pi * k_y) * wh).real
    lap_w = torch.fft.ifft2(-laplacian * wh).real

    residual = wt + (u * wx + v * wy) - viscosity * lap_w - forcing
    return residual[:, 1:-1]


def residual_mse(w: torch.Tensor, forcing: torch.Tensor, viscosity: float, dt: float = 0.1,):
    residual = residual_map_vorticity2d(w=w, forcing=forcing, viscosity=viscosity, dt=dt)
    return torch.mean(residual**2)
