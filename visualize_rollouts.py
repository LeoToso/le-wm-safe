"""Visualize rollouts from the dataset as frame grids and GIFs."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import argparse

from safe_lewm.dataset import load_trajectories

DATA_PATH = "/mnt/t7shield/safety_point_goal.pkl"
OUT_DIR = Path("rollout_viz")
OUT_DIR.mkdir(exist_ok=True)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def denorm(frame):
    """(3, H, W) normalized -> (H, W, 3) uint8"""
    f = frame * IMAGENET_STD[:, None, None] + IMAGENET_MEAN[:, None, None]
    return (f.clip(0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)


def plot_rollout_grid(traj, traj_idx, n_frames=12, every_n=None):
    """Plot evenly-spaced frames from a trajectory as a grid with cost overlay."""
    T = len(traj["obs"])
    if every_n is None:
        idxs = np.linspace(0, T - 1, n_frames, dtype=int)
    else:
        idxs = np.arange(0, T, every_n)[:n_frames]

    cols = min(6, len(idxs))
    rows = (len(idxs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.8))
    axes = np.array(axes).flatten()

    for i, t in enumerate(idxs):
        frame = denorm(traj["obs"][t, :3])  # first stacked frame, RGB
        axes[i].imshow(frame)
        cost  = traj["cost"][t]
        rew   = traj["reward"][t]
        color = "red" if cost > 0 else "white"
        axes[i].set_title(f"t={t}\nr={rew:.2f} c={int(cost)}", fontsize=7, color=color)
        axes[i].axis("off")
        if cost > 0:
            for spine in axes[i].spines.values():
                spine.set_edgecolor("red"); spine.set_linewidth(3)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    ep_cost = traj["cost"].sum()
    ep_rew  = traj["reward"].sum()
    fig.suptitle(
        f"Trajectory {traj_idx} — total reward={ep_rew:.2f}, total cost={int(ep_cost)}, length={T}",
        fontsize=10, fontweight="bold"
    )
    plt.tight_layout()
    out = OUT_DIR / f"rollout_{traj_idx:04d}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def make_gif(traj, traj_idx, fps=8, max_frames=100):
    """Save a GIF of the rollout (requires imageio)."""
    try:
        import imageio
    except ImportError:
        print("  imageio not found, skipping GIF (pip install imageio)")
        return None

    T = min(len(traj["obs"]), max_frames)
    frames = []
    for t in range(T):
        frame = denorm(traj["obs"][t, :3])  # (H, W, 3) uint8
        # Red border on cost steps
        if traj["cost"][t] > 0:
            frame[:4, :] = [255, 0, 0]
            frame[-4:, :] = [255, 0, 0]
            frame[:, :4] = [255, 0, 0]
            frame[:, -4:] = [255, 0, 0]
        frames.append(frame)

    out = OUT_DIR / f"rollout_{traj_idx:04d}.gif"
    imageio.mimsave(str(out), frames, fps=fps)
    return out


def main():
    print(f"Loading trajectories from {DATA_PATH}...")
    trajs = load_trajectories(DATA_PATH)
    print(f"Loaded {len(trajs)} trajectories")

    # Pick interesting trajectories to visualize:
    # - highest cost, lowest cost, median cost, random sample
    costs = [(t["cost"].sum(), i) for i, t in enumerate(trajs)]
    costs_sorted = sorted(costs, key=lambda x: x[0])

    selected = {
        "highest_cost": costs_sorted[-1][1],
        "second_highest": costs_sorted[-2][1],
        "median": costs_sorted[len(costs_sorted) // 2][1],
        "lowest_cost": costs_sorted[0][1],
        "random_1": np.random.randint(len(trajs)),
        "random_2": np.random.randint(len(trajs)),
    }

    print(f"\nVisualizing {len(selected)} trajectories...")
    for label, idx in selected.items():
        traj = trajs[idx]
        ep_cost = traj["cost"].sum()
        print(f"  [{label}] traj {idx}: reward={traj['reward'].sum():.2f}, cost={int(ep_cost)}")
        out = plot_rollout_grid(traj, idx, n_frames=12)
        print(f"    -> {out}")
        gif = make_gif(traj, idx, fps=8)
        if gif:
            print(f"    -> {gif}")

    # Also make a summary figure: one row per selected trajectory, 6 frames each
    print("\nMaking summary figure...")
    fig, axes = plt.subplots(len(selected), 6, figsize=(15, 2.5 * len(selected)))
    for row, (label, idx) in enumerate(selected.items()):
        traj = trajs[idx]
        T = len(traj["obs"])
        frame_idxs = np.linspace(0, T - 1, 6, dtype=int)
        for col, t in enumerate(frame_idxs):
            ax = axes[row, col]
            frame = denorm(traj["obs"][t, :3])
            ax.imshow(frame)
            cost = traj["cost"][t]
            if cost > 0:
                ax.set_title(f"t={t} ⚠", fontsize=7, color="red")
            else:
                ax.set_title(f"t={t}", fontsize=7)
            ax.axis("off")
        ep_cost = traj["cost"].sum()
        axes[row, 0].set_ylabel(f"{label}\ncost={int(ep_cost)}", fontsize=7, rotation=0,
                                 labelpad=60, va="center")

    plt.suptitle("Rollout Summary — red ⚠ = cost step", fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = OUT_DIR / "rollout_summary.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")
    print(f"\nAll outputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
