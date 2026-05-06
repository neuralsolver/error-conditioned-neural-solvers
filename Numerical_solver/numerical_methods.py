import time

import numpy as np
import torch
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import spsolve


def lap(S):
    h = 1.0 / (S - 1)
    n = S - 2

    e = np.ones(n)
    L1 = diags([e, -2 * e, e], offsets=[-1, 0, 1], shape=(n, n), format='csr') / h**2
    L_full = kron(eye(n, format='csr'), L1) + kron(L1, eye(n, format='csr'))
    return L_full.tocsr()


def compute_metrics(psi_full, f_full, u_gt_full, pde_loss_grid, device, k, lamb):
    pred_vec = psi_full[1:-1, 1:-1].reshape(-1)
    u_gt_vec = u_gt_full[1:-1, 1:-1].reshape(-1)
    mse = np.mean((pred_vec - u_gt_vec)**2)
    rel_l2 = np.linalg.norm(pred_vec - u_gt_vec) / (np.linalg.norm(u_gt_vec) + 1e-12)

    psi_torch = torch.from_numpy(psi_full).float().to(device)[None, None, :, :]
    f_torch = torch.from_numpy(f_full).float().to(device)[None, None, :, :]
    with torch.no_grad():
        l_pde = pde_loss_grid(psi_torch, f_torch, k, lamb)
        pde = torch.mean(l_pde ** 2).item()
    return pde, mse, rel_l2


def solve_numerical(
    f_full,
    u_gt_full,
    init_full,
    pde_loss_grid,
    device,
    solver_type="Newton",
    S=128,
    k=2.0,
    lamb=1.0,
    max_iter=50,
    tol=1e-5,
    lr=1e-6,
    verbose=True,
):
    n = S - 2
    L_full = lap(S)
    A = L_full + (k**2) * eye(n * n, format='csr')

    f_vec = f_full[1:-1, 1:-1].reshape(-1)

    if init_full is None:
        psi_vec = np.zeros_like(f_vec)
    else:
        psi_vec = init_full[1:-1, 1:-1].reshape(-1).copy()

    residual_history = []
    recon_mse_history = []
    rel_l2_history = []
    solve_time = 0.0
    time_history = []
    solver_type = solver_type.lower()
    N = psi_vec.size

    for it in range(max_iter):
        psi_full = np.zeros((S, S), dtype=np.float64)
        psi_full[1:-1, 1:-1] = psi_vec.reshape(n, n)

        pde, mse, rel_l2 = compute_metrics(psi_full, f_full, u_gt_full, pde_loss_grid, device, k, lamb)
        F_vec = A @ psi_vec + lamb * (psi_vec**3) - f_vec

        residual_history.append(pde)
        recon_mse_history.append(mse)
        rel_l2_history.append(rel_l2)
        time_history.append(solve_time)

        if verbose:
            print(
                f"iter {it:03d}: PDE MSE = {pde:.3e}, "
                f"recon MSE = {mse:.3e}, rel L2 = {rel_l2:.3e}, "
                f"time = {solve_time:.3e}"
            )

        if pde < tol:
            break

        t0 = time.perf_counter()
        J = A + diags(3.0 * lamb * (psi_vec**2), offsets=0, format='csr')

        if solver_type in ["newton", "n"]:
            delta = spsolve(J, -F_vec)
            psi_vec = psi_vec + delta
        elif solver_type in ["gaussnewton", "gn", "gauss-newton"]:
            JTJ = J.T @ J
            JTF = J.T @ F_vec
            delta = spsolve(JTJ, -JTF)
            psi_vec = psi_vec + delta
        elif solver_type in ["gradientdescent", "gd", "gradient-descent"]:
            grad = (2.0 / N) * (J.T @ F_vec)
            psi_vec = psi_vec - lr * grad
        else:
            raise ValueError(f"Unknown numerical solver_type: {solver_type}")

        solve_time += time.perf_counter() - t0

    psi_full = np.zeros((S, S), dtype=np.float64)
    psi_full[1:-1, 1:-1] = psi_vec.reshape(n, n)
    return psi_full, residual_history, recon_mse_history, rel_l2_history, time_history
