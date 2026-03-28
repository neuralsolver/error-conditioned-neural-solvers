"""
PDE residual loss modules for physics-informed training and evaluation.
"""

import math
import torch
import torch.nn as nn


class PDELossHZ(nn.Module):
    """Helmholtz equation residual: u_xx + u_yy + k^2 * u - f = 0."""

    def __init__(self):
        super().__init__()

    def forward(self, u, f, k):
        n = u.shape[3]
        h = 1 / (n - 1)
        residual = torch.zeros_like(u)
        u_xx = (u[:, :, 1:-1, 2:] - 2 * u[:, :, 1:-1, 1:-1] + u[:, :, 1:-1, :-2]) / (h * h)
        u_yy = (u[:, :, 2:, 1:-1] - 2 * u[:, :, 1:-1, 1:-1] + u[:, :, :-2, 1:-1]) / (h * h)
        residual[:, :, 1:-1, 1:-1] = (
            u_xx + u_yy + (k ** 2) * u[:, :, 1:-1, 1:-1] - f[:, :, 1:-1, 1:-1]
        )
        return residual


class PDELossPS(nn.Module):
    """Poisson equation residual: u_xx + u_yy - f = 0."""

    def __init__(self):
        super().__init__()

    def forward(self, u, f, k):
        n = u.shape[3]
        h = 1 / (n - 1)
        residual = torch.zeros_like(u)
        u_xx = (u[:, :, 1:-1, 2:] - 2 * u[:, :, 1:-1, 1:-1] + u[:, :, 1:-1, :-2]) / (h * h)
        u_yy = (u[:, :, 2:, 1:-1] - 2 * u[:, :, 1:-1, 1:-1] + u[:, :, :-2, 1:-1]) / (h * h)
        residual[:, :, 1:-1, 1:-1] = u_xx + u_yy - k * f[:, :, 1:-1, 1:-1]
        return residual


class PDELossDarcy(nn.Module):
    """Darcy flow residual: -div(a * grad(u)) = f."""

    def __init__(self):
        super().__init__()

    def forward(self, u, f, a):
        n = u.shape[3]
        h = 1 / (n - 1)

        u_x = (u[:, :, :, 2:] - u[:, :, :, :-2]) / (2 * h)
        u_y = (u[:, :, 2:, :] - u[:, :, :-2, :]) / (2 * h)
        u_x = nn.functional.pad(u_x, (1, 1, 0, 0), mode='reflect')
        u_y = nn.functional.pad(u_y, (0, 0, 1, 1), mode='reflect')

        u_xx = (u[:, :, :, 2:] - 2 * u[:, :, :, 1:-1] + u[:, :, :, :-2]) / (h ** 2)
        u_yy = (u[:, :, 2:, :] - 2 * u[:, :, 1:-1, :] + u[:, :, :-2, :]) / (h ** 2)
        u_xx = nn.functional.pad(u_xx, (1, 1, 0, 0), mode='reflect')
        u_yy = nn.functional.pad(u_yy, (0, 0, 1, 1), mode='reflect')

        a_x = (a[:, :, :, 2:] - a[:, :, :, :-2]) / (2 * h)
        a_y = (a[:, :, 2:, :] - a[:, :, :-2, :]) / (2 * h)
        a_x = nn.functional.pad(a_x, (1, 1, 0, 0), mode='reflect')
        a_y = nn.functional.pad(a_y, (0, 0, 1, 1), mode='reflect')

        residual = a * (u_xx + u_yy) + (a_x * u_x) + (a_y * u_y) + f
        return residual


class PDELossBurgerSpec(nn.Module):
    """Burgers equation residual via spectral (FFT) differentiation."""

    def __init__(self):
        super().__init__()

    def forward(self, u, nu=1 / 100):
        B, C, Nt, Nx = u.shape
        device = u.device
        dt = 1.0 / (Nt - 1)

        k = 2 * torch.pi * torch.fft.fftfreq(Nx, d=1.0 / Nx).to(device)
        u_fft = torch.fft.fft(u, dim=-1)

        u_xx = torch.fft.ifft(-(k ** 2) * u_fft, dim=-1).real
        u_sq_x = torch.fft.ifft(1j * k * torch.fft.fft(0.5 * u * u, dim=-1), dim=-1).real

        u_t = torch.zeros_like(u)
        u_t[:, :, 2:-2, :] = (
            -u[:, :, 4:, :] + 8 * u[:, :, 3:-1, :] - 8 * u[:, :, 1:-3, :] + u[:, :, :-4, :]
        ) / (12 * dt)
        u_t[..., 0, :] = (-25*u[...,0,:] + 48*u[...,1,:] - 36*u[...,2,:] + 16*u[...,3,:] - 3*u[...,4,:]) / (12*dt)
        u_t[..., 1, :] = (-3*u[...,0,:] - 10*u[...,1,:] + 18*u[...,2,:] - 6*u[...,3,:] + u[...,4,:]) / (12*dt)
        u_t[..., -2, :] = (-u[...,-5,:] + 6*u[...,-4,:] - 18*u[...,-3,:] + 10*u[...,-2,:] + 3*u[...,-1,:]) / (12*dt)
        u_t[..., -1, :] = (25*u[...,-1,:] - 48*u[...,-2,:] + 36*u[...,-3,:] - 16*u[...,-4,:] + 3*u[...,-5,:]) / (12*dt)

        return u_t + u_sq_x - nu * u_xx


class PDELossBurgerFDM(nn.Module):
    """Burgers equation residual via finite difference method."""

    def __init__(self):
        super().__init__()

    def forward(self, u, nu=1 / 100):
        B, C, Nt, Nx = u.shape
        device = u.device
        dx = 1.0 / (Nx - 1)
        dt = 1.0 / (Nt - 1)

        u_xx = (torch.roll(u, -1, -1) - 2 * u + torch.roll(u, 1, -1)) / (dx * dx)
        u_sq_x = (torch.roll(0.5 * u ** 2, -1, -1) - torch.roll(0.5 * u ** 2, 1, -1)) / (2 * dx)

        u_t = torch.zeros_like(u)
        u_t[:, :, 2:-2, :] = (
            -u[:, :, 4:, :] + 8 * u[:, :, 3:-1, :] - 8 * u[:, :, 1:-3, :] + u[:, :, :-4, :]
        ) / (12 * dt)
        u_t[..., 0, :] = (-25*u[...,0,:] + 48*u[...,1,:] - 36*u[...,2,:] + 16*u[...,3,:] - 3*u[...,4,:]) / (12*dt)
        u_t[..., 1, :] = (-3*u[...,0,:] - 10*u[...,1,:] + 18*u[...,2,:] - 6*u[...,3,:] + u[...,4,:]) / (12*dt)
        u_t[..., -2, :] = (-u[...,-5,:] + 6*u[...,-4,:] - 18*u[...,-3,:] + 10*u[...,-2,:] + 3*u[...,-1,:]) / (12*dt)
        u_t[..., -1, :] = (25*u[...,-1,:] - 48*u[...,-2,:] + 36*u[...,-3,:] - 16*u[...,-4,:] + 3*u[...,-5,:]) / (12*dt)

        return u_t + u_sq_x - nu * u_xx


class PDELossNS3d(nn.Module):
    """3D Navier-Stokes vorticity residual (spectral)."""

    def __init__(self):
        super().__init__()

    def forward(self, w, f, nu, dt=0.0476):
        assert w.ndim == 5
        B, _, N, N2, T = w.shape
        assert N == N2
        device = w.device
        dtype = w.dtype

        f = f.to(device=device, dtype=dtype)

        k_max = math.floor(N / 2.0)
        k_y = torch.cat(
            (torch.arange(0, k_max, device=device), torch.arange(-k_max, 0, device=device)), 0
        ).repeat(N, 1)
        k_x = k_y.transpose(0, 1)

        lap_pos = 4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2)
        lap_pos[0, 0] = 1.0

        R = torch.empty((B, N, N, T), device=device, dtype=dtype)
        wt = torch.empty((B, N, N, T), device=device, dtype=dtype)
        wt[:, :, :, 0] = (w[:, 0, :, :, 1] - w[:, 0, :, :, 0]) / dt
        wt[:, :, :, -1] = (w[:, 0, :, :, -1] - w[:, 0, :, :, -2]) / dt
        for n in range(1, T - 1):
            wt[:, :, :, n] = (w[:, 0, :, :, n + 1] - w[:, 0, :, :, n - 1]) / (2 * dt)

        for n in range(T):
            wn = w[:, 0, :, :, n]
            wh = torch.fft.fft2(wn)
            psi_h = wh / lap_pos
            u_vel = torch.fft.ifft2(1j * 2 * math.pi * k_y * psi_h).real
            v_vel = torch.fft.ifft2(-1j * 2 * math.pi * k_x * psi_h).real
            wx = torch.fft.ifft2(1j * 2 * math.pi * k_x * wh).real
            wy = torch.fft.ifft2(1j * 2 * math.pi * k_y * wh).real
            lap_w = torch.fft.ifft2(-lap_pos * wh).real
            R[:, :, :, n] = wt[:, :, :, n] + u_vel * wx + v_vel * wy - nu * lap_w - f

        return R.unsqueeze(1)


class PDELossNS2d(nn.Module):
    """2D Navier-Stokes vorticity residual (spectral)."""

    def __init__(self):
        super().__init__()

    def forward(self, w, f, nu, dt):
        assert w.ndim == 4
        B, T, N, N2 = w.shape
        assert N == N2
        device = w.device
        dtype = w.dtype

        f = f.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)

        k_max = math.floor(N / 2.0)
        k_y = torch.cat(
            (torch.arange(0, k_max, device=device), torch.arange(-k_max, 0, device=device)), 0
        ).repeat(N, 1)
        k_x = k_y.transpose(0, 1)

        lap = 4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2)
        lap[0, 0] = 1.0

        wt = torch.empty_like(w)
        wt[:, 0] = (w[:, 1] - w[:, 0]) / dt
        wt[:, -1] = (w[:, -1] - w[:, -2]) / dt
        wt[:, 1:-1] = (w[:, 2:] - w[:, :-2]) / (2 * dt)

        wh = torch.fft.fft2(w)
        psi_h = wh / lap
        u_vel = torch.fft.ifft2((1j * 2 * math.pi * k_y) * psi_h).real
        v_vel = torch.fft.ifft2((-1j * 2 * math.pi * k_x) * psi_h).real
        wx = torch.fft.ifft2((1j * 2 * math.pi * k_x) * wh).real
        wy = torch.fft.ifft2((1j * 2 * math.pi * k_y) * wh).real
        lap_w = torch.fft.ifft2(-lap * wh).real

        return wt + u_vel * wx + v_vel * wy - nu * lap_w - f
