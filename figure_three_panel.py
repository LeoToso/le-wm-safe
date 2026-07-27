"""
Three-panel summary figure:
  Panel 1: t-SNE of ρ=1 (with regularization) latent space — safe vs unsafe
  Panel 2: t-SNE of ρ=0 (no regularization) latent space — safe vs unsafe
  Panel 3: Score std on OOD layouts (ρ=1 vs ρ=0) — from evaluate_ood_generalization

Usage:
    MUJOCO_GL=osmesa python figure_three_panel.py

    # If OOD records already saved:
    OOD_RECORDS=/path/to/ood_records.npz python figure_three_panel.py
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, ".")
from safe_lewm.config import Config
from safe_lewm.model import SafeJEPA
from safe_lewm.classifier import ObstacleMLP
from safe_lewm.dataset import load_trajectories

# ── Config ─────────────────────────────────────────────────────────────────────
CHECKPOINT     = os.environ.get("CHECKPOINT",      "/mnt/t7shield/safe_lewm_rho1.pt")
CHECKPOINT_NR  = os.environ.get("CHECKPOINT_NOREG","/mnt/t7shield/safe_lewm_rho0.pt")
CLF_PATH       = os.environ.get("CLF_PATH",        "/mnt/t7shield/classifier.pt")
CLF_NR_PATH    = os.environ.get("CLF_NOREG_PATH",  "/mnt/t7shield/classifier_rho0.pt")
DATA_PATH      = os.environ.get("DATA_PATH",       "/mnt/t7shield/safety_point_goal.pkl")
OOD_RECORDS    = os.environ.get("OOD_RECORDS",     "eval_wm_viz/ood_records.npz")
ENV_NAME       = os.environ.get("ENV_NAME",        "SafetyPointGoal1-v0")
OUT_PATH       = os.environ.get("OUT_PATH",        "figure_three_panel.png")
MAX_SAMPLES    = int(os.environ.get("MAX_SAMPLES", "5000"))
N_TSNE         = int(os.environ.get("N_TSNE",      "2000"))
# OOD collection params (used only if OOD_RECORDS file is missing)
N_EPISODES     = int(os.environ.get("N_EPISODES",  "50"))
MAX_STEPS      = int(os.environ.get("MAX_STEPS",   "400"))
IN_DIST_START  = int(os.environ.get("IN_DIST_START","0"))
OOD_START      = int(os.environ.get("OOD_START",   "5000"))
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────────────────────────

C_SAFE   = "#4C9BE8"   # blue
C_UNSAFE = "#E84C4C"   # red
C_R      = "#2166ac"   # dark blue  — ρ=1
C_N      = "#d73027"   # red        — ρ=0


def load_encoder(ckpt_path):
    cfg = Config()
    m = SafeJEPA(cfg).to(DEVICE)
    m.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False))
    m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m.encoder, cfg


def load_clf(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    c = ObstacleMLP(z_dim=ckpt["z_dim"], hidden_dim=ckpt["hidden_dim"],
                    depth=ckpt["depth"], dropout=ckpt.get("dropout", 0.1)).to(DEVICE)
    c.load_state_dict(ckpt["classifier_state"])
    c.eval()
    return c, ckpt["z_mean"].to(DEVICE), ckpt["z_std"].to(DEVICE)


def encode_dataset(encoder, trajs):
    """Encode observations from trajectories, return (Z, labels)."""
    obs_list, cost_list = [], []
    for traj in trajs:
        for t in range(len(traj["obs"])):
            obs_list.append(traj["obs"][t])
            cost_list.append(traj["cost"][t])
        if len(obs_list) >= MAX_SAMPLES * 4:
            break

    obs_arr  = np.array(obs_list,  dtype=np.float32)
    cost_arr = np.array(cost_list, dtype=np.float32)

    safe_idx   = np.where(cost_arr == 0)[0]
    unsafe_idx = np.where(cost_arr == 1)[0]
    rng = np.random.default_rng(42)
    n_unsafe = min(len(unsafe_idx), MAX_SAMPLES // 2)
    n_safe   = min(len(safe_idx),   MAX_SAMPLES // 2)
    sel_idx  = np.concatenate([rng.choice(safe_idx,   n_safe,   replace=False),
                                rng.choice(unsafe_idx, n_unsafe, replace=False)])
    rng.shuffle(sel_idx)

    obs_sel  = obs_arr[sel_idx]
    labels   = cost_arr[sel_idx].astype(int)

    BATCH = 256
    z_list = []
    with torch.no_grad():
        for i in range(0, len(obs_sel), BATCH):
            batch = torch.tensor(obs_sel[i:i+BATCH]).to(DEVICE)
            z_list.append(encoder(batch).cpu().numpy())
    Z = np.concatenate(z_list, axis=0)
    return Z, labels


def compute_tsne(Z, labels, n_tsne):
    rng = np.random.default_rng(42)
    from sklearn.preprocessing import StandardScaler
    Z_scaled = StandardScaler().fit_transform(Z)
    Z_pca    = PCA(n_components=min(50, Z.shape[1])).fit_transform(Z_scaled)
    idx      = rng.choice(len(Z_pca), min(n_tsne, len(Z_pca)), replace=False)
    Z2d      = TSNE(n_components=2, perplexity=40, random_state=42, n_iter=1000)\
                   .fit_transform(Z_pca[idx])
    return Z2d, labels[idx]


def plot_tsne(ax, Z2d, lab_tsne, title):
    for lab, color, marker, name in [(0, C_SAFE, "o", "safe"), (1, C_UNSAFE, "^", "unsafe")]:
        mask = lab_tsne == lab
        ax.scatter(Z2d[mask, 0], Z2d[mask, 1],
                   c=color, marker=marker, s=14, alpha=0.6, label=name)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.axis("off")


def collect_ood_records():
    """Collect (dist, score_reg, score_noreg) records for in-dist and OOD seeds."""
    from safe_lewm.env_utils import SafetyGymWrapper

    def hazard_dist(env):
        try:
            u = env.env.unwrapped.task
            agent_pos = np.array(u.agent.pos[:2])
            dists = [np.linalg.norm(agent_pos - np.array(h[:2])) for h in u.hazards.pos]
            return float(np.min(dists)) if dists else float('inf')
        except Exception:
            return float('inf')

    def score_both(obs_np):
        x = torch.tensor(obs_np[None], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            z_r = enc_reg(x);   sr = clf_reg((z_r - zm_r) / zs_r).item()
            z_n = enc_noreg(x); sn = clf_noreg((z_n - zm_n) / zs_n).item()
        return sr, sn

    def collect(seed_start, label):
        records = []
        cfg = Config()
        for ep in range(N_EPISODES):
            env = SafetyGymWrapper(ENV_NAME, cfg.image_size, cfg.frame_stack, cfg.frame_skip,
                                   seed=seed_start + ep)
            obs = env.reset()
            for _ in range(MAX_STEPS):
                d = hazard_dist(env)
                sr, sn = score_both(obs)
                records.append((d, sr, sn))
                obs, _, _, done, _ = env.step(env.env.action_space.sample())
                if done: break
            env.env.close()
            if (ep + 1) % 10 == 0:
                print(f"  [{label}] {ep+1}/{N_EPISODES} episodes", flush=True)
        return np.array(records)

    print("Collecting in-dist records...")
    rec_in  = collect(IN_DIST_START, "in-dist")
    print("Collecting OOD records...")
    rec_ood = collect(OOD_START,     "OOD")
    return rec_in, rec_ood


def plot_ood_std(ax, rec_ood):
    MAX_DIST = 3.0
    bins = np.concatenate([np.linspace(0, 1.0, 15),
                            np.linspace(1.0, MAX_DIST, 10)[1:]])
    bc   = 0.5 * (bins[:-1] + bins[1:])

    def bin_std(records, col):
        stds = []
        for i in range(len(bins)-1):
            mask = (records[:, 0] >= bins[i]) & (records[:, 0] < bins[i+1])
            s = records[mask, col]
            stds.append(s.std() if len(s) >= 3 else np.nan)
        return np.array(stds)

    sr_ood = bin_std(rec_ood, 1)
    sn_ood = bin_std(rec_ood, 2)

    v = ~(np.isnan(sr_ood) | np.isnan(sn_ood))
    ax.plot(bc[v], sr_ood[v], "-o", color=C_R, markersize=5, linewidth=2, label="ρ=1 (with reg)")
    ax.plot(bc[v], sn_ood[v], "-o", color=C_N, markersize=5, linewidth=2, label="ρ=0 (no reg)")
    ax.fill_between(bc[v], sr_ood[v], sn_ood[v],
                    where=(sn_ood[v] > sr_ood[v]),
                    alpha=0.2, color="purple", label="ρ=0 excess uncertainty")

    close = bc < 0.8
    if close.any() and not np.isnan(sr_ood[close]).all():
        ratio = np.nanmean(sn_ood[close]) / max(np.nanmean(sr_ood[close]), 1e-6)
        ax.text(0.05, 0.92, f"ρ=0 std / ρ=1 std at <0.8 m:\n  ×{ratio:.2f} higher uncertainty",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    ax.set_title("Score Std on OOD Layouts\n(lower = more reliable warning signal)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Distance to nearest hazard (m)", fontsize=10)
    ax.set_ylabel("Score std (uncertainty)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    ax.set_ylim(bottom=0)


# ── Load models ────────────────────────────────────────────────────────────────
print("Loading encoders...")
enc_reg,  cfg = load_encoder(CHECKPOINT)
enc_noreg, _  = load_encoder(CHECKPOINT_NR)

print("Loading classifiers...")
clf_reg,   zm_r, zs_r = load_clf(CLF_PATH)
clf_noreg, zm_n, zs_n = load_clf(CLF_NR_PATH)

# ── t-SNE ──────────────────────────────────────────────────────────────────────
print("Loading dataset for t-SNE...")
trajs = load_trajectories(DATA_PATH)

print("Encoding with ρ=1...")
Z_r, lab_r = encode_dataset(enc_reg,   trajs)
print("Encoding with ρ=0...")
Z_n, lab_n = encode_dataset(enc_noreg, trajs)

print("Computing t-SNE for ρ=1...")
Z2d_r, lab2d_r = compute_tsne(Z_r, lab_r, N_TSNE)
print("Computing t-SNE for ρ=0...")
Z2d_n, lab2d_n = compute_tsne(Z_n, lab_n, N_TSNE)

# ── OOD records ────────────────────────────────────────────────────────────────
if Path(OOD_RECORDS).exists():
    print(f"Loading OOD records from {OOD_RECORDS}...")
    d = np.load(OOD_RECORDS)
    rec_ood = d["rec_ood"]
else:
    print("OOD records not found, collecting now...")
    rec_in, rec_ood = collect_ood_records()
    Path(OOD_RECORDS).parent.mkdir(exist_ok=True)
    np.savez(OOD_RECORDS, rec_in=rec_in, rec_ood=rec_ood)
    print(f"Saved to {OOD_RECORDS}")

# ── Compose three-panel figure ─────────────────────────────────────────────────
print("Composing figure...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

plot_tsne(axes[0], Z2d_r, lab2d_r,
          "t-SNE — ρ=1 (with regularization)\nSafe vs Unsafe")
plot_tsne(axes[1], Z2d_n, lab2d_n,
          "t-SNE — ρ=0 (no regularization)\nSafe vs Unsafe")
plot_ood_std(axes[2], rec_ood)

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"\nSaved: {OUT_PATH}")
