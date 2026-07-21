"""
Test classifier generalization to novel hazard configurations (OOD).

The training dataset was collected under a fixed set of environment seeds.
This script runs episodes under:
  - IN-DISTRIBUTION seeds  (seeds 0–49,   similar to training)
  - OUT-OF-DISTRIBUTION seeds (seeds 5000–5049, novel hazard placements)

For each seed/step, we record the distance to the nearest hazard and both
classifiers' scores. We then compare:
  1. Score-vs-distance curves: does the score still drop as hazard approaches?
  2. Variance at each distance bin: ρ=1 (geometry-based) should stay consistent
     across novel layouts; ρ=0 (memorized boundary) should become erratic.

This directly demonstrates why the t-SNE clustering matters for planning:
a geometrically structured latent space generalizes; a scattered one doesn't.

Usage:
    MUJOCO_GL=osmesa python evaluate_ood_generalization.py
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, ".")
from safe_lewm.config import Config
from safe_lewm.model import SafeJEPA
from safe_lewm.classifier import ObstacleMLP
from safe_lewm.env_utils import SafetyGymWrapper

# ── Config ─────────────────────────────────────────────────────────────────────
CHECKPOINT     = os.environ.get("CHECKPOINT",     "/mnt/t7shield/safe_lewm_rho1.pt")
CHECKPOINT_NR  = os.environ.get("CHECKPOINT_NOREG", "/mnt/t7shield/safe_lewm_rho0.pt")
CLF_PATH       = os.environ.get("CLF_PATH",       "/mnt/t7shield/classifier.pt")
CLF_NR_PATH    = os.environ.get("CLF_NOREG_PATH", "/mnt/t7shield/classifier_rho0.pt")
ENV_NAME       = os.environ.get("ENV_NAME",       "SafetyPointGoal1-v0")
OUT_DIR        = os.environ.get("OUT_DIR",        "eval_wm_viz")
N_EPISODES     = int(os.environ.get("N_EPISODES", "10"))   # per condition
MAX_STEPS      = int(os.environ.get("MAX_STEPS",  "150"))
NUM_HAZARDS    = int(os.environ.get("NUM_HAZARDS", "4"))
IN_DIST_START  = int(os.environ.get("IN_DIST_START",  "0"))
OOD_START      = int(os.environ.get("OOD_START",      "5000"))
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────────────────────────

Path(OUT_DIR).mkdir(exist_ok=True)
print(f"Device: {DEVICE} | ENV={ENV_NAME} | N_EPISODES={N_EPISODES} per condition")
print(f"IN-DIST seeds: {IN_DIST_START}–{IN_DIST_START+N_EPISODES-1}")
print(f"OOD seeds:     {OOD_START}–{OOD_START+N_EPISODES-1}")


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


print("\nLoading models...")
enc_reg,  cfg = load_encoder(CHECKPOINT)
enc_noreg, _  = load_encoder(CHECKPOINT_NR)
clf_reg,  zm_reg,  zs_reg  = load_clf(CLF_PATH)
clf_noreg, zm_noreg, zs_noreg = load_clf(CLF_NR_PATH)


def score_both(obs_np):
    """Score with both classifiers in one GPU round-trip."""
    x = torch.tensor(obs_np[None], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        z_r = enc_reg(x);   sr = clf_reg((z_r - zm_reg) / zs_reg).item()
        z_n = enc_noreg(x); sn = clf_noreg((z_n - zm_noreg) / zs_noreg).item()
    return sr, sn


def hazard_dist(env):
    try:
        u = env.env.unwrapped.task
        agent_pos = np.array(u.agent.pos[:2])
        dists = [np.linalg.norm(agent_pos - np.array(h[:2])) for h in u.hazards.pos]
        return float(np.min(dists)) if dists else float('inf')
    except Exception:
        try:
            u = env.env.unwrapped.task
            model, data = u.model, u.data
            agent_pos = np.array(data.xpos[model.body('agent').id][:2])
            dists = [np.linalg.norm(agent_pos - data.xpos[i][:2])
                     for i in range(model.nbody) if model.body(i).name.startswith('hazard')]
            return float(np.min(dists)) if dists else float('inf')
        except Exception:
            return float('inf')


def collect_records(seed_start, label):
    records = []  # (dist, score_reg, score_noreg)
    for ep in range(N_EPISODES):
        seed = seed_start + ep
        env = SafetyGymWrapper(ENV_NAME, cfg.image_size, cfg.frame_stack, cfg.frame_skip, seed=seed)
        try:
            env.env.unwrapped.task.hazards.num = NUM_HAZARDS
        except Exception:
            pass
        obs = env.reset()
        for _ in range(MAX_STEPS):
            d = hazard_dist(env)
            sr, sn = score_both(obs)
            records.append((d, sr, sn))
            obs, _, _, done, _ = env.step(env.env.action_space.sample())
            if done:
                break
        env.env.close()
        if (ep + 1) % 10 == 0:
            print(f"  [{label}] {ep+1}/{N_EPISODES} episodes, {len(records)} steps")
    return np.array(records)


print("\nCollecting IN-DISTRIBUTION data...")
rec_in = collect_records(IN_DIST_START, "in-dist")

print("\nCollecting OUT-OF-DISTRIBUTION data...")
rec_ood = collect_records(OOD_START, "OOD")

print(f"\nIn-dist steps:  {len(rec_in)}")
print(f"OOD steps:      {len(rec_ood)}")


# ── Bin by distance and compute mean ± std ────────────────────────────────────
MAX_DIST = 3.0
bins = np.linspace(0, MAX_DIST, 20)
bc   = 0.5 * (bins[:-1] + bins[1:])


def bin_scores(records, col):
    means, stds = [], []
    for i in range(len(bins)-1):
        mask = (records[:, 0] >= bins[i]) & (records[:, 0] < bins[i+1])
        s = records[mask, col]
        means.append(s.mean() if len(s) >= 3 else np.nan)
        stds.append(s.std()  if len(s) >= 3 else np.nan)
    return np.array(means), np.array(stds)


# Col 1 = score_reg, col 2 = score_noreg
mr_in,  sr_in  = bin_scores(rec_in,  1)
mn_in,  sn_in  = bin_scores(rec_in,  2)
mr_ood, sr_ood = bin_scores(rec_ood, 1)
mn_ood, sn_ood = bin_scores(rec_ood, 2)


# ── Quantify generalization gap ───────────────────────────────────────────────
close_mask_in  = rec_in[:,  0] < 0.8
close_mask_ood = rec_ood[:, 0] < 0.8

def generalization_gap(rec_in, rec_ood, col, label):
    s_in  = rec_in[rec_in[:,0]  < 0.8, col]
    s_ood = rec_ood[rec_ood[:,0] < 0.8, col]
    print(f"\n  {label} (close range <0.8m):")
    print(f"    In-dist  — mean={s_in.mean():.3f}  std={s_in.std():.3f}  n={len(s_in)}")
    print(f"    OOD      — mean={s_ood.mean():.3f}  std={s_ood.std():.3f}  n={len(s_ood)}")
    print(f"    Mean shift (OOD−in): {s_ood.mean()-s_in.mean():+.3f}   "
          f"Std ratio (OOD/in): {s_ood.std()/max(s_in.std(),1e-6):.2f}×")

print("\n── Generalization gap at close range (<0.8 m from hazard) ───────────────")
generalization_gap(rec_in, rec_ood, 1, "ρ=1 (regularized)")
generalization_gap(rec_in, rec_ood, 2, "ρ=0 (no reg)")


# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("OOD Generalization: Score vs Hazard Distance\n"
             "In-distribution (solid) vs Novel hazard layouts (dashed)", fontsize=12)

# --- Panel 1: ρ=1 in-dist vs OOD ---
ax = axes[0]
v = ~np.isnan(mr_in)
ax.plot(bc[v], mr_in[v],  "b-o",  markersize=4, label="ρ=1 in-dist")
ax.fill_between(bc[v], mr_in[v]-sr_in[v], mr_in[v]+sr_in[v], alpha=0.2, color="blue")
v2 = ~np.isnan(mr_ood)
ax.plot(bc[v2], mr_ood[v2], "b--o", markersize=4, label="ρ=1 OOD")
ax.fill_between(bc[v2], mr_ood[v2]-sr_ood[v2], mr_ood[v2]+sr_ood[v2], alpha=0.15, color="blue")
ax.axhline(0, color="k", linestyle="--", linewidth=1)
ax.set_title("ρ=1 (with regularization)\nIn-dist vs OOD")
ax.set_xlabel("Distance to nearest hazard (m)")
ax.set_ylabel("Safety score")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.invert_xaxis()

# --- Panel 2: ρ=0 in-dist vs OOD ---
ax = axes[1]
v = ~np.isnan(mn_in)
ax.plot(bc[v], mn_in[v],  "r-o",  markersize=4, label="ρ=0 in-dist")
ax.fill_between(bc[v], mn_in[v]-sn_in[v], mn_in[v]+sn_in[v], alpha=0.2, color="red")
v2 = ~np.isnan(mn_ood)
ax.plot(bc[v2], mn_ood[v2], "r--o", markersize=4, label="ρ=0 OOD")
ax.fill_between(bc[v2], mn_ood[v2]-sn_ood[v2], mn_ood[v2]+sn_ood[v2], alpha=0.15, color="red")
ax.axhline(0, color="k", linestyle="--", linewidth=1)
ax.set_title("ρ=0 (no regularization)\nIn-dist vs OOD")
ax.set_xlabel("Distance to nearest hazard (m)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.invert_xaxis()

# --- Panel 3: OOD score std comparison (key metric) ---
ax = axes[2]
v = ~(np.isnan(sr_ood) | np.isnan(sn_ood))
ax.plot(bc[v], sr_ood[v], "b-o", markersize=4, label="ρ=1 std (OOD)")
ax.plot(bc[v], sn_ood[v], "r-o", markersize=4, label="ρ=0 std (OOD)")
ax.set_title("Score Std on OOD Layouts\n(lower = more reliable warning signal)")
ax.set_xlabel("Distance to nearest hazard (m)")
ax.set_ylabel("Score std (uncertainty)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.invert_xaxis()

plt.tight_layout()
out = f"{OUT_DIR}/ood_generalization.png"
plt.savefig(out, dpi=150)
print(f"\nPlot saved: {out}")
