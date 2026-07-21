"""
Show how safety score varies with distance to nearest hazard for both classifiers.

The argument: if ρ=1 gives lower (more negative) scores at larger hazard distances,
the MPPI planner has a wider buffer — it starts penalizing unsafe trajectories
from further away, leading to earlier and smoother avoidance.

Run a set of episodes, record (distance_to_nearest_hazard, safety_score) at each
step for both classifiers, then plot score vs distance with confidence bands.

Usage:
    python evaluate_safety_margin_vs_distance.py
"""
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, ".")
from safe_lewm.config import Config
from safe_lewm.model import SafeJEPA
from safe_lewm.classifier import ObstacleMLP
from safe_lewm.env_utils import SafetyGymWrapper

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT       = os.environ.get("CHECKPOINT",       "/mnt/t7shield/safe_lewm_rho1.pt")
CHECKPOINT_NOREG = os.environ.get("CHECKPOINT_NOREG", "/mnt/t7shield/safe_lewm_rho0.pt")
CLF_PATH         = os.environ.get("CLF_PATH",         "/mnt/t7shield/classifier.pt")
CLF_NOREG_PATH   = os.environ.get("CLF_NOREG_PATH",   "/mnt/t7shield/classifier_rho0.pt")
ENV_NAME         = os.environ.get("ENV_NAME",         "SafetyPointGoal1-v0")
OUT_DIR          = os.environ.get("OUT_DIR",          "eval_wm_viz")
N_EPISODES       = int(os.environ.get("N_EPISODES",   "20"))
MAX_STEPS        = int(os.environ.get("MAX_STEPS",    "200"))
NUM_HAZARDS      = int(os.environ.get("NUM_HAZARDS",  "4"))
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────────────────────────

Path(OUT_DIR).mkdir(exist_ok=True)
print(f"Device: {DEVICE} | ENV={ENV_NAME} | N_EPISODES={N_EPISODES}")


def load_encoder(ckpt_path):
    cfg = Config()
    m = SafeJEPA(cfg).to(DEVICE)
    m.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False))
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m.encoder, cfg


def load_clf(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    c = ObstacleMLP(z_dim=ckpt["z_dim"], hidden_dim=ckpt["hidden_dim"],
                    depth=ckpt["depth"], dropout=ckpt.get("dropout", 0.1)).to(DEVICE)
    c.load_state_dict(ckpt["classifier_state"])
    c.eval()
    zm = ckpt["z_mean"].to(DEVICE)
    zs = ckpt["z_std"].to(DEVICE)
    return c, zm, zs


print("Loading models and classifiers...")
enc_reg,   cfg = load_encoder(CHECKPOINT)
enc_noreg, _   = load_encoder(CHECKPOINT_NOREG)
clf_reg,   zm_reg,   zs_reg   = load_clf(CLF_PATH)[:3]
clf_noreg, zm_noreg, zs_noreg = load_clf(CLF_NOREG_PATH)[:3]


def get_score(obs_np, encoder, clf, zm, zs):
    x = torch.tensor(obs_np[None], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        z = encoder(x)
        z_norm = (z - zm) / zs
        return clf(z_norm).item()


def get_hazard_distance(env):
    """Return distance to the nearest hazard from the agent's current position."""
    try:
        u = env.env.unwrapped.task
        agent_pos = np.array(u.agent.pos[:2], dtype=np.float64)
        # u.hazards.pos is a list of (x, y, z) positions for each hazard
        hazard_positions = u.hazards.pos
        if len(hazard_positions) == 0:
            return float('inf')
        dists = [np.linalg.norm(agent_pos - np.array(h[:2])) for h in hazard_positions]
        return float(np.min(dists))
    except Exception:
        # Fallback via MuJoCo model body names
        try:
            u = env.env.unwrapped.task
            model, data = u.model, u.data
            agent_pos = np.array(data.xpos[model.body('agent').id][:2])
            dists = []
            for i in range(model.nbody):
                if model.body(i).name.startswith('hazard'):
                    dists.append(np.linalg.norm(agent_pos - data.xpos[i][:2]))
            return float(np.min(dists)) if dists else float('inf')
        except Exception:
            return float('inf')


# ── Collect data ───────────────────────────────────────────────────────────────
print(f"\nRunning {N_EPISODES} episodes with random policy...")
records = []  # list of (dist, score_reg, score_noreg)

for ep in range(N_EPISODES):
    env = SafetyGymWrapper(ENV_NAME, cfg.image_size, cfg.frame_stack, cfg.frame_skip, seed=ep)
    try:
        env.env.unwrapped.task.hazards.num = NUM_HAZARDS
    except Exception:
        pass
    obs = env.reset()

    for step in range(MAX_STEPS):
        dist = get_hazard_distance(env)
        score_reg   = get_score(obs, enc_reg,   clf_reg,   zm_reg,   zs_reg)
        score_noreg = get_score(obs, enc_noreg, clf_noreg, zm_noreg, zs_noreg)
        records.append((dist, score_reg, score_noreg))

        action = env.env.action_space.sample()
        obs, _, _, done, _ = env.step(action)
        terminated = truncated = done
        if terminated or truncated:
            break

    env.close()
    if (ep + 1) % 5 == 0:
        print(f"  Episode {ep+1}/{N_EPISODES} done  ({len(records)} samples collected)")

records = np.array(records)  # (N, 3): dist, score_reg, score_noreg
dist_all       = records[:, 0]
score_reg_all  = records[:, 1]
score_noreg_all= records[:, 2]

print(f"\nTotal samples: {len(records)}")

# ── Bin by distance ────────────────────────────────────────────────────────────
MAX_DIST = 3.5
bins = np.linspace(0, MAX_DIST, 25)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

mean_reg, std_reg, mean_noreg, std_noreg = [], [], [], []
counts = []

for i in range(len(bins) - 1):
    mask = (dist_all >= bins[i]) & (dist_all < bins[i+1])
    counts.append(mask.sum())
    if mask.sum() >= 3:
        mean_reg.append(score_reg_all[mask].mean())
        std_reg.append(score_reg_all[mask].std())
        mean_noreg.append(score_noreg_all[mask].mean())
        std_noreg.append(score_noreg_all[mask].std())
    else:
        mean_reg.append(np.nan); std_reg.append(np.nan)
        mean_noreg.append(np.nan); std_noreg.append(np.nan)

mean_reg   = np.array(mean_reg);   std_reg   = np.array(std_reg)
mean_noreg = np.array(mean_noreg); std_noreg = np.array(std_noreg)

# ── Print table ────────────────────────────────────────────────────────────────
print("\n── Safety score vs hazard distance ────────────────────────────────────")
print(f"{'Dist':>6}  {'ρ=1 score':>10}  {'ρ=0 score':>10}  {'n':>5}")
for i, bc in enumerate(bin_centers):
    if not np.isnan(mean_reg[i]):
        print(f"{bc:>6.2f}  {mean_reg[i]:>+10.3f}  {mean_noreg[i]:>+10.3f}  {counts[i]:>5}")

# Find the distance where each classifier crosses threshold=0 on average
def crossover_dist(bin_centers, mean_scores):
    for i in range(len(bin_centers) - 1):
        if not (np.isnan(mean_scores[i]) or np.isnan(mean_scores[i+1])):
            if mean_scores[i] < 0 < mean_scores[i+1] or mean_scores[i+1] < 0 < mean_scores[i]:
                # linear interpolate
                t = -mean_scores[i] / (mean_scores[i+1] - mean_scores[i])
                return bin_centers[i] + t * (bin_centers[i+1] - bin_centers[i])
    return None

cross_reg   = crossover_dist(bin_centers, mean_reg)
cross_noreg = crossover_dist(bin_centers, mean_noreg)

print("\n── Score=0 crossover distance (planning buffer) ───────────────────────")
if cross_reg:
    print(f"  ρ=1: mean score crosses 0 at ~{cross_reg:.2f}m from hazard")
if cross_noreg:
    print(f"  ρ=0: mean score crosses 0 at ~{cross_noreg:.2f}m from hazard")
if cross_reg and cross_noreg:
    print(f"  ρ=1 gives {cross_reg - cross_noreg:+.2f}m additional warning distance")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Safety Score vs Distance to Nearest Hazard", fontsize=13)

# Left: score vs distance curves
ax = axes[0]
valid = ~np.isnan(mean_reg)
ax.plot(bin_centers[valid], mean_reg[valid],   "b-o", markersize=4, label="ρ=1 (with reg)")
ax.fill_between(bin_centers[valid],
                mean_reg[valid] - std_reg[valid],
                mean_reg[valid] + std_reg[valid], alpha=0.2, color="blue")
ax.plot(bin_centers[valid], mean_noreg[valid], "r-o", markersize=4, label="ρ=0 (no reg)")
ax.fill_between(bin_centers[valid],
                mean_noreg[valid] - std_noreg[valid],
                mean_noreg[valid] + std_noreg[valid], alpha=0.2, color="red")
ax.axhline(0, color="k", linestyle="--", linewidth=1.2, label="Threshold=0 (planner trigger)")
if cross_reg:
    ax.axvline(cross_reg,   color="blue", linestyle=":", alpha=0.7, label=f"ρ=1 buffer ≈{cross_reg:.2f}m")
if cross_noreg:
    ax.axvline(cross_noreg, color="red",  linestyle=":", alpha=0.7, label=f"ρ=0 buffer ≈{cross_noreg:.2f}m")
ax.set_xlabel("Distance to nearest hazard (m)")
ax.set_ylabel("Safety score  (+ = safe, − = unsafe)")
ax.set_title("Score vs Hazard Distance\n(higher buffer = more planning lead time)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.invert_xaxis()   # left = closer to hazard

# Right: histogram of score at close range (< 1.0m) vs far (> 2.0m)
ax = axes[1]
close = dist_all < 1.0
far   = dist_all > 2.0
bins_s = np.linspace(-4, 4, 40)
ax.hist(score_reg_all[close],   bins=bins_s, alpha=0.5, color="blue",       density=True, label="ρ=1 close (<1m)")
ax.hist(score_reg_all[far],     bins=bins_s, alpha=0.5, color="steelblue",  density=True, label="ρ=1 far (>2m)", linestyle="--", histtype="step", linewidth=1.5)
ax.hist(score_noreg_all[close], bins=bins_s, alpha=0.5, color="red",        density=True, label="ρ=0 close (<1m)")
ax.hist(score_noreg_all[far],   bins=bins_s, alpha=0.5, color="salmon",     density=True, label="ρ=0 far (>2m)",  linestyle="--", histtype="step", linewidth=1.5)
ax.axvline(0, color="k", linestyle="--", linewidth=1.2)
ax.set_xlabel("Safety score")
ax.set_ylabel("Density")
ax.set_title("Score Distribution: Close vs Far from Hazard")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = f"{OUT_DIR}/score_vs_hazard_distance.png"
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved: {out_path}")
