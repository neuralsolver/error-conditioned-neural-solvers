import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pde import residual_map_vorticity2d


def create_animation(data, title, vmin=None, vmax=None, cmap="viridis"):

    fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(data[0], cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
    fig.colorbar(im)

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    def update(frame):
        im.set_data(data[frame])
        ax.set_title(f"{title} (t={frame})")
        return [im]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(data),
        interval=400
    )

    plt.close(fig)
    return ani


def save_field_animations(u_pred_test, test_u_out, sample_idx=15, num_frames=40):
    u_true = test_u_out.detach().cpu().numpy()[sample_idx][:num_frames]
    u_pred = u_pred_test.detach().cpu().numpy()[sample_idx][:num_frames]
    diff = np.abs(u_pred - u_true)

    vmin = u_true.min()
    vmax = u_true.max()

    ani_true = create_animation(u_true, "True NS", vmin, vmax)
    ani_pred = create_animation(u_pred, "Predicted KF", vmin, vmax)
    ani_err  = create_animation(diff, "Absolute Error", cmap="inferno", vmax=0.84)

    ani_true.save("true_NS.gif", writer="pillow", fps=2)
    ani_pred.save("pred_NS.gif", writer="pillow", fps=2)
    ani_err.save("error_NS.gif", writer="pillow", fps=2)

    print("Animations saved.")


def save_residual_animation(u_pred_test, f, sample_idx=20, num_frames=18):
    R_test = residual_map_vorticity2d(u_pred_test, f, 5e-4)
    R_np = R_test.detach().cpu().numpy()[sample_idx][:num_frames]

    vmin = R_np.min()
    vmax = R_np.max()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(R_np[0], origin="lower", vmax=4.9, vmin=-4.5)
    fig.colorbar(im, ax=ax, label="Residual")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("PDE Residual (t=0)")

    def update(frame):
        im.set_data(R_np[frame])
        ax.set_title(f"PDE Residual (t={frame})")
        return [im]

    ani_R = animation.FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=400,
        blit=False
    )

    plt.close(fig)

    ani_R.save("residual_NS.gif", writer="pillow", fps=2)


def plot_stepsize_comparison(save_dir="checkpoints"):
    #reconstruction loss (step-size for extrapolation)
    rl_7 = np.load(f"{save_dir}/rl_0.07.npy")
    rl_5 = np.load(f"{save_dir}/rl_0.05.npy")
    rl_4 = np.load(f"{save_dir}/rl_0.04.npy")
    rl_1 = np.load(f"{save_dir}/rl_0.01.npy")
    plt.figure()
    plt.plot(rl_7, label='step-size = 0.07')
    plt.plot(rl_5, label='step-size = 0.05')
    plt.plot(rl_4, label='step-size = 0.04')
    plt.plot(rl_1, label='step-size = 0.01')
    plt.xlabel("T_test")
    plt.ylabel("Reconstruction Loss")
    plt.legend()
    plt.grid()
    plt.show()

    #pde loss (step-size for extrapolation)
    pde_7 = np.load(f"{save_dir}/pde_0.07.npy")
    pde_5 = np.load(f"{save_dir}/pde_0.05.npy")
    pde_4 = np.load(f"{save_dir}/pde_0.04.npy")
    pde_1 = np.load(f"{save_dir}/pde_0.01.npy")
    plt.figure()
    plt.plot(np.log(pde_7), label='step-size = 0.07')
    plt.plot(np.log(pde_5), label='step-size = 0.05')
    plt.plot(np.log(pde_4), label='step-size = 0.04')
    plt.plot(np.log(pde_1), label='step-size = 0.01')
    plt.xlabel("T_test")
    plt.ylabel("PDE residual")
    plt.legend()
    plt.grid()
    plt.show()
