"""
Direct evaluation of world model quality via multi-step prediction accuracy.

Compares two models:
  - ρ=1  (with regularization)   : CHECKPOINT (default safe_lewm_rho1.pt)
  - ρ=0  (without regularization): CHECKPOINT_NOREG (default safe_lewm_rho0.pt)

For each held-out trajectory segment of length HIST_LEN + EVAL_HORIZON:
  1. Encode the first HIST_LEN frames to get latent history.
  2. Roll out the transition model for EVAL_HORIZON steps using ground-truth actions.
  3. Encode the true next observations (teacher-forced targets).
  4. Measure cosine similarity and L2 error between predicted and true latents
     at each horizon step.

Usage:
    python evaluate_world_model.py
    CHECKPOINT=/mnt/t7shield/safe_lewm_rho1.pt \\
    CHECKPOINT_NOREG=/mnt/t7shield/safe_lewm_rho0.pt \\
    DATA_PATH=/mnt/t7shield/safety_point_goal.pkl \\
    python evaluate_world_model.py
"""
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, ".")
from safe_lewm.config import Config
from safe_lewm.model import SafeJEPA
from safe_lewm.dataset import load_trajectories

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT       = os.environ.get("CHECKPOINT",       "/mnt/t7shield/safe_lewm_rho1.pt")
CHECKPOINT_NOREG = os.environ.get("CHECKPOINT_NOREG", "/mnt/t7shield/safe_lewm_rho0.pt")
DATA_PATH        = os.environ.get("DATA_PATH",        "/mnt/t7shield/safety_point_goal.pkl")
OUT_DIR          = os.environ.get("OUT_DIR",          "eval_wm_viz")
HIST_LEN         = int(os.environ.get("HIST_LEN",    "4"))   # frames of context
EVAL_HORIZON     = int(os.environ.get("EVAL_HORIZON", "15"))  # steps to roll out
N_SEGS           = int(os.environ.get("N_SEGS",      "2000")) # trajectory segments
BATCH            = int(os.environ.get("BATCH",        "128"))
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────────────────────────

Path(OUT_DIR).mkdir(exist_ok=True)
print(f"Device: {DEVICE}")
print(f"Encoder ρ=1:  {CHECKPOINT}")
print(f"Encoder ρ=0:  {CHECKPOINT_NOREG}")
print(f"HIST_LEN={HIST_LEN}  EVAL_HORIZON={EVAL_HORIZON}  N_SEGS={N_SEGS}")


def load_model(ckpt_path):
    cfg = Config()
    m = SafeJEPA(cfg).to(DEVICE)
    m.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False))
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, cfg


print("\nLoading models...")
model_reg,   cfg = load_model(CHECKPOINT)
model_noreg, _   = load_model(CHECKPOINT_NOREG)

# ── Build evaluation segments ──────────────────────────────────────────────────
print("\nLoading trajectories...")
trajs = load_trajectories(DATA_PATH)

SEG_LEN = HIST_LEN + EVAL_HORIZON
rng = np.random.default_rng(0)

# Collect valid segments from the held-out portion of trajectories
# Use the last 15% of each trajectory to approximate a held-out set
obs_segs    = []   # (N_SEGS, SEG_LEN, C, H, W)
action_segs = []   # (N_SEGS, SEG_LEN, A)

for traj in rng.permutation(len(trajs)):
    t = trajs[traj]
    T = len(t["obs"])
    start_heldout = int(T * 0.85)
    for start in range(start_heldout, T - SEG_LEN):
        obs_segs.append(t["obs"][start:start + SEG_LEN])
        action_segs.append(t["action"][start:start + SEG_LEN])
        if len(obs_segs) >= N_SEGS:
            break
    if len(obs_segs) >= N_SEGS:
        break

print(f"Collected {len(obs_segs)} evaluation segments")
obs_segs    = np.array(obs_segs,    dtype=np.float32)  # (N, SEG_LEN, C, H, W)
action_segs = np.array(action_segs, dtype=np.float32)  # (N, SEG_LEN, A)


def evaluate_model(model, obs_segs, action_segs):
    """
    Returns:
        l2_by_step:    (EVAL_HORIZON,) mean L2 error at each horizon step
        cos_by_step:   (EVAL_HORIZON,) mean cosine similarity at each horizon step
    """
    N = len(obs_segs)
    l2_accum  = np.zeros(EVAL_HORIZON)
    cos_accum = np.zeros(EVAL_HORIZON)
    count = 0

    for i in range(0, N, BATCH):
        obs_b   = torch.tensor(obs_segs[i:i+BATCH]).to(DEVICE)    # (B, SEG_LEN, C, H, W)
        act_b   = torch.tensor(action_segs[i:i+BATCH]).to(DEVICE) # (B, SEG_LEN, A)
        B = obs_b.shape[0]

        # Encode all frames to get ground-truth latents
        flat = obs_b.view(B * SEG_LEN, *obs_b.shape[2:])
        z_all = model.encoder(flat).view(B, SEG_LEN, -1)  # (B, SEG_LEN, z_dim)

        # Context: first HIST_LEN frames
        z_hist = z_all[:, :HIST_LEN, :]         # (B, HIST_LEN, z_dim)
        act_future = act_b[:, HIST_LEN:, :]     # (B, EVAL_HORIZON, A)

        # Roll out transition model
        z_rolled = model.rollout(z_hist, act_future, history_size=HIST_LEN)
        # z_rolled: (B, HIST_LEN + EVAL_HORIZON, z_dim)
        z_pred = z_rolled[:, HIST_LEN:, :]     # (B, EVAL_HORIZON, z_dim)
        z_true = z_all[:, HIST_LEN:, :]        # (B, EVAL_HORIZON, z_dim)

        # L2 error per step
        l2 = (z_pred - z_true).pow(2).sum(-1).sqrt()  # (B, EVAL_HORIZON)
        l2_accum += l2.cpu().numpy().sum(0)

        # Cosine similarity per step
        p_norm = z_pred / (z_pred.norm(dim=-1, keepdim=True) + 1e-8)
        t_norm = z_true / (z_true.norm(dim=-1, keepdim=True) + 1e-8)
        cos = (p_norm * t_norm).sum(-1)  # (B, EVAL_HORIZON)
        cos_accum += cos.cpu().numpy().sum(0)

        count += B

    return l2_accum / count, cos_accum / count


print("\nEvaluating ρ=1 (with regularization)...")
l2_reg, cos_reg = evaluate_model(model_reg, obs_segs, action_segs)

print("Evaluating ρ=0 (without regularization)...")
l2_noreg, cos_noreg = evaluate_model(model_noreg, obs_segs, action_segs)

# ── Print results ──────────────────────────────────────────────────────────────
steps = np.arange(1, EVAL_HORIZON + 1)
print("\n── Multi-step prediction error (L2 in latent space) ──────────────────")
print(f"{'Step':>4}  {'ρ=1 L2':>10}  {'ρ=0 L2':>10}  {'Δ (ρ=0−ρ=1)':>12}")
for s, lr, ln in zip(steps, l2_reg, l2_noreg):
    print(f"{s:>4}  {lr:>10.4f}  {ln:>10.4f}  {ln-lr:>+12.4f}")

print("\n── Cosine similarity ─────────────────────────────────────────────────")
print(f"{'Step':>4}  {'ρ=1 cos':>10}  {'ρ=0 cos':>10}  {'Δ (ρ=1−ρ=0)':>12}")
for s, cr, cn in zip(steps, cos_reg, cos_noreg):
    print(f"{s:>4}  {cr:>10.4f}  {cn:>10.4f}  {cr-cn:>+12.4f}")

print(f"\nSummary (mean over {EVAL_HORIZON} steps):")
print(f"  ρ=1  avg L2={l2_reg.mean():.4f}  avg cos={cos_reg.mean():.4f}")
print(f"  ρ=0  avg L2={l2_noreg.mean():.4f}  avg cos={cos_noreg.mean():.4f}")
print(f"  L2 improvement  (ρ=1 vs ρ=0): {(l2_noreg.mean()-l2_reg.mean())/l2_noreg.mean()*100:+.1f}%")
print(f"  Cos improvement (ρ=1 vs ρ=0): {(cos_reg.mean()-cos_noreg.mean())/abs(cos_noreg.mean())*100:+.1f}%")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("World Model Prediction Quality: ρ=1 vs ρ=0", fontsize=13)

ax = axes[0]
ax.plot(steps, l2_reg,   "b-o", label="ρ=1 (with reg)", markersize=4)
ax.plot(steps, l2_noreg, "r-o", label="ρ=0 (no reg)",   markersize=4)
ax.set_xlabel("Prediction horizon (steps)")
ax.set_ylabel("L2 error in latent space")
ax.set_title("Multi-step L2 Prediction Error")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(steps, cos_reg,   "b-o", label="ρ=1 (with reg)", markersize=4)
ax.plot(steps, cos_noreg, "r-o", label="ρ=0 (no reg)",   markersize=4)
ax.set_xlabel("Prediction horizon (steps)")
ax.set_ylabel("Cosine similarity")
ax.set_title("Multi-step Cosine Similarity")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([-0.1, 1.05])

plt.tight_layout()
out_path = f"{OUT_DIR}/wm_prediction_accuracy.png"
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved: {out_path}")

# Save raw numbers
np.savez(f"{OUT_DIR}/wm_eval_results.npz",
         steps=steps, l2_reg=l2_reg, l2_noreg=l2_noreg,
         cos_reg=cos_reg, cos_noreg=cos_noreg)
print(f"Raw data saved: {OUT_DIR}/wm_eval_results.npz")
