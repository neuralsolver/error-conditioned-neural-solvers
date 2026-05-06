import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import uniform_filter


def generate_smooth_boundary_noise(S, scale=6.0, smooth_size=9):
    n = S - 2
    noise = np.random.randn(n, n)
    smooth = uniform_filter(noise, size=smooth_size)

    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    R = np.sqrt(X**2 + Y**2)
    mask = np.exp(- (R / 0.8)**4)

    psi = scale * (mask * noise + (1 - mask) * smooth)
    psi_full = np.zeros((S, S), dtype=np.float32)
    psi_full[1:-1, 1:-1] = psi
    return psi_full


def model0_noise_initialization(model_out, beta=2.0, alpha=1.0, kernel=9, mask_radius=0.6, seed=0):
    torch.manual_seed(seed)
    noise = torch.randn_like(model_out)
    padding = kernel // 2
    noise_smooth = F.avg_pool2d(noise, kernel_size=kernel, stride=1, padding=padding)
    noise_high = noise - noise_smooth

    _, _, H, W = model_out.shape
    x = torch.linspace(-1, 1, H, device=model_out.device)
    y = torch.linspace(-1, 1, W, device=model_out.device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    R = torch.sqrt(X**2 + Y**2)
    mask = torch.exp(- (R / mask_radius)**4).unsqueeze(0).unsqueeze(0)

    shape_mask = torch.abs(model_out)
    shape_mask = shape_mask / (shape_mask.amax(dim=(2,3), keepdim=True) + 1e-8)
    perturb = noise_high * (0.3 + 0.7 * shape_mask) * mask
    return alpha * model_out + beta * perturb


def build_initialization(init_type, f_tensor, model0, init_config, S):
    init_type = init_type.lower()
    if init_type in ["zero", "0"]:
        return np.zeros((f_tensor.shape[0], S, S), dtype=np.float32)

    if init_type in ["gaussian", "gauss_noise", "noise"]:
        return np.stack([
            generate_smooth_boundary_noise(
                S,
                scale=init_config.get("gaussian_scale", 6.0),
                smooth_size=init_config.get("gaussian_smooth_size", 9),
            )
            for _ in range(f_tensor.shape[0])
        ], axis=0)

    if init_type in ["model0", "fno"]:
        if model0 is None:
            raise ValueError("model0 initialization requires model checkpoint.")
        with torch.no_grad():
            return model0(f_tensor).squeeze(1).detach().cpu().numpy()

    if init_type in ["model0_noise", "model0_then_noise", "fno_noise"]:
        if model0 is None:
            raise ValueError("model0_noise initialization requires model checkpoint.")
        with torch.no_grad():
            model_out = model0(f_tensor)
            u_init = model0_noise_initialization(
                model_out,
                alpha=init_config.get("model0_alpha", 1.0),
                beta=init_config.get("model0_noise_beta", 2.0),
                kernel=init_config.get("model0_noise_kernel", 9),
                mask_radius=init_config.get("model0_noise_mask_radius", 0.6),
                seed=init_config.get("model0_noise_seed", 0),
            )
        return u_init.squeeze(1).detach().cpu().numpy()

    raise ValueError(f"Unknown initialization type: {init_type}")
