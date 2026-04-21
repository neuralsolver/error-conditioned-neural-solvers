import math

import torch


def residual_map_vorticity2d(w, f, nu, dt=0.1):

    assert w.ndim == 4
    B, T, N, N2 = w.shape
    assert N == N2

    device = w.device
    dtype  = w.dtype

    f = f.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)

    k_max = math.floor(N / 2.0)
    k_y = torch.cat((
        torch.arange(0, k_max, device=device),
        torch.arange(-k_max, 0, device=device)
    ), 0).repeat(N, 1)
    k_x = k_y.transpose(0, 1)

    lap = 4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2)
    lap[0, 0] = 1.0

    wt = torch.empty_like(w)
    wt[:, 0] = (w[:, 1] - w[:, 0]) / dt
    wt[:, -1] = (w[:, -1] - w[:, -2]) / dt
    wt[:, 1:-1] = (w[:, 2:] - w[:, :-2]) / (2 * dt)
    wh = torch.fft.fft2(w)

    psi_h = wh / lap
    u = torch.fft.ifft2((1j * 2 * math.pi * k_y) * psi_h).real
    v = torch.fft.ifft2((-1j * 2 * math.pi * k_x) * psi_h).real
    wx = torch.fft.ifft2((1j * 2 * math.pi * k_x) * wh).real
    wy = torch.fft.ifft2((1j * 2 * math.pi * k_y) * wh).real
    lap_w = torch.fft.ifft2(-lap * wh).real

    adv = u * wx + v * wy
    diff = nu * lap_w
    res = wt + adv - diff - f

    #den = (torch.abs(adv[:, 1:-1]) + torch.abs(diff[:, 1:-1]) + torch.abs(wt[:, 1:-1])  + torch.abs(f) + 1e-6)
    den = 1
    return res[:, 1:-1, :, :]


def _setup_wavenumbers(N, device):
    k_max = math.floor(N / 2.0)
    k_y = torch.cat((
        torch.arange(0, k_max, device=device),
        torch.arange(-k_max, 0, device=device)
    ), 0).repeat(N, 1).float()
    k_x = k_y.transpose(0, 1)
    lap = 4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2)
    lap[0, 0] = 1.0
    return k_x, k_y, lap


def _rhs(w, f, nu, k_x, k_y, lap):
    wh = torch.fft.fft2(w)
    psi_h = wh / lap
    u  =  torch.fft.ifft2((1j * 2 * math.pi * k_y) * psi_h).real
    v  =  torch.fft.ifft2((-1j * 2 * math.pi * k_x) * psi_h).real
    wx =  torch.fft.ifft2((1j * 2 * math.pi * k_x) * wh).real
    wy =  torch.fft.ifft2((1j * 2 * math.pi * k_y) * wh).real
    lap_w = torch.fft.ifft2(-lap * wh).real
    return -u * wx - v * wy + nu * lap_w + f


def _rk4_integrate(w, f, nu, k_x, k_y, lap, dt_fine, N_substeps):
    for _ in range(N_substeps):
        s1 = _rhs(w,                    f, nu, k_x, k_y, lap)
        s2 = _rhs(w + 0.5 * dt_fine * s1, f, nu, k_x, k_y, lap)
        s3 = _rhs(w + 0.5 * dt_fine * s2, f, nu, k_x, k_y, lap)
        s4 = _rhs(w + dt_fine * s3,       f, nu, k_x, k_y, lap)
        w = w + (dt_fine / 6) * (s1 + 2 * s2 + 2 * s3 + s4)
    return w


def residual_map_vorticity2d1(w, f, nu, dt_coarse=0.1, N_substeps=25):
    assert w.ndim == 4
    B, T, N, N2 = w.shape
    assert N == N2

    device = w.device
    dtype  = w.dtype
    f = f.to(device=device, dtype=dtype)

    k_x, k_y, lap = _setup_wavenumbers(N, device)
    dt_fine = dt_coarse / N_substeps

    residuals = []
    for k in range(T - 1):
        w_integrated = _rk4_integrate(
            w[:, k], f, nu, k_x, k_y, lap, dt_fine, N_substeps
        )
        res = w_integrated - w[:, k + 1]   # shape (B, N, N)
        residuals.append(res)

    return torch.stack(residuals, dim=1)
