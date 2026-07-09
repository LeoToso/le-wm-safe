"""
Offline evaluation of the conformal safety classifier.

No environment rendering needed — runs on the pre-collected dataset.

Measures:
  1. Classifier precision/recall on safe vs unsafe latents
  2. MPPI safety filter: given a trajectory from the dataset, how often does
     the planner's safety penalty correctly flag the unsafe steps?
  3. Safety score distribution: safe vs unsafe observations
  4. Conformal threshold coverage: fraction of unsafe obs correctly flagged

Usage:
    python evaluate_offline.py
    CHECKPOINT=/mnt/t7shield/safe_lewm_rho1.pt python evaluate_offline.py
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, ".")
from safe_lewm.config import Config
from safe_lewm.model import SafeJEPA
from safe_lewm.classifier import ObstacleMLP
from safe_lewm.dataset import load_trajectories

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT = os.environ.get("CHECKPOINT", "/mnt/t7shield/safe_lewm_rho1.pt")
CLF_PATH   = os.environ.get("CLF_PATH",   "/mnt/t7shield/classifier.pt")
DATA_PATH  = os.environ.get("DATA_PATH",  "/mnt/t7shield/safety_point_goal.pkl")
OUT_DIR    = Path(os.environ.get("OUT_DIR", "eval_offline"))
MAX_OBS    = 20000
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────────────────────────

OUT_DIR.mkdir(exist_ok=True)
print(f"Device: {DEVICE}")

# ── Load encoder ──────────────────────────────────────────────────────────────
print("Loading encoder...")
cfg = Config()
model = SafeJEPA(cfg).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False))
model.eval()
encoder = model.encoder
for p in encoder.parameters():
    p.requires_grad_(False)

# ── Load classifier ───────────────────────────────────────────────────────────
print("Loading classifier...")
ckpt = torch.load(CLF_PATH, map_location=DEVICE, weights_only=False)
clf = ObstacleMLP(
    z_dim=ckpt["z_dim"], hidden_dim=ckpt["hidden_dim"],
    depth=ckpt["depth"], dropout=ckpt.get("dropout", 0.1),
).to(DEVICE)
clf.load_state_dict(ckpt["classifier_state"])
clf.eval()
z_mean = ckpt["z_mean"].to(DEVICE)
z_std  = ckpt["z_std"].to(DEVICE)
safe_threshold = ckpt["safe_threshold"]
print(f"  Conformal threshold: {safe_threshold:.4f}")

# ── Load data & encode ────────────────────────────────────────────────────────
print("Loading trajectories...")
trajs = load_trajectories(DATA_PATH)

print("Encoding observations...")
obs_list, cost_list = [], []
for traj in trajs:
    for t in range(len(traj["obs"])):
        obs_list.append(traj["obs"][t])
        cost_list.append(traj["cost"][t])
    if len(obs_list) >= MAX_OBS:
        break

obs_arr  = np.array(obs_list[:MAX_OBS], dtype=np.float32)
cost_arr = np.array(cost_list[:MAX_OBS], dtype=np.float32)
labels   = cost_arr.astype(int)

BATCH = 512
z_list = []
with torch.no_grad():
    for i in range(0, len(obs_arr), BATCH):
        x = torch.tensor(obs_arr[i:i+BATCH]).to(DEVICE)
        z = encoder(x).cpu()
        z_list.append(z)
Z = torch.cat(z_list, dim=0)  # (N, z_dim)

print(f"Encoded {len(Z)} observations | safe={int((labels==0).sum())} | unsafe={int((labels==1).sum())}")

# Normalize
Z_norm = ((Z.to(DEVICE) - z_mean) / z_std)

# ── Classifier scores ─────────────────────────────────────────────────────────
print("\nComputing classifier scores...")
with torch.no_grad():
    scores = clf(Z_norm).cpu().numpy()  # (N,) positive=safe, negative=unsafe

safe_scores   = scores[labels == 0]
unsafe_scores = scores[labels == 1]

print(f"\n── Score statistics ──")
print(f"  Safe   scores: mean={safe_scores.mean():.4f}, std={safe_scores.std():.4f}, "
      f"min={safe_scores.min():.4f}, max={safe_scores.max():.4f}")
print(f"  Unsafe scores: mean={unsafe_scores.mean():.4f}, std={unsafe_scores.std():.4f}, "
      f"min={unsafe_scores.min():.4f}, max={unsafe_scores.max():.4f}")

# ── Classifier accuracy at threshold ─────────────────────────────────────────
print(f"\n── Classifier accuracy (threshold={safe_threshold:.4f}) ──")
preds = (scores >= safe_threshold).astype(int)   # 1 = predicted safe
true_safe   = labels == 0
true_unsafe = labels == 1

tp = int(( (preds == 1) & true_safe  ).sum())   # correctly predicted safe
tn = int(( (preds == 0) & true_unsafe).sum())   # correctly predicted unsafe
fp = int(( (preds == 1) & true_unsafe).sum())   # unsafe predicted as safe (dangerous!)
fn = int(( (preds == 0) & true_safe  ).sum())   # safe predicted as unsafe (conservative)

precision = tp / (tp + fp + 1e-8)
recall    = tp / (tp + fn + 1e-8)
unsafe_recall = tn / (tn + fp + 1e-8)  # recall on unsafe class (= 1 - false negative rate)

print(f"  True  positives (safe→safe):     {tp}")
print(f"  True  negatives (unsafe→unsafe): {tn}")
print(f"  False positives (unsafe→safe):   {fp}  ← dangerous misses")
print(f"  False negatives (safe→unsafe):   {fn}  ← conservative false alarms")
print(f"  Safe precision:   {precision*100:.1f}%")
print(f"  Safe recall:      {recall*100:.1f}%")
print(f"  Unsafe recall:    {unsafe_recall*100:.1f}%  (conformal guarantee: ≥{(1-ckpt['delta'])*100:.0f}%)")

# ── Conformal coverage check ──────────────────────────────────────────────────
print(f"\n── Conformal coverage ──")
# The guarantee: fraction of unsafe obs where score <= threshold should be >= 1-delta
nc_scores = np.maximum(0.0, scores[labels == 1])   # nonconformity scores on unsafe
coverage  = (nc_scores <= safe_threshold).mean()
print(f"  Unsafe obs with score <= threshold: {coverage*100:.1f}%  (target: ≥{(1-ckpt['delta'])*100:.0f}%)")
print(f"  Nonconformity score range: [{nc_scores.min():.4f}, {nc_scores.max():.4f}]")

# ── Trajectory-level analysis ─────────────────────────────────────────────────
print(f"\n── Trajectory-level safety analysis ──")
# For each trajectory, check how many unsafe steps the classifier would flag
flagged_unsafe_steps = 0
total_unsafe_steps   = 0
flagged_safe_steps   = 0
total_safe_steps     = 0

for traj in trajs[:100]:   # first 100 trajs
    T = len(traj["obs"])
    with torch.no_grad():
        x = torch.tensor(traj["obs"], dtype=torch.float32).to(DEVICE)
        z = encoder(x)
        z_n = (z - z_mean) / z_std
        s = clf(z_n).cpu().numpy()

    costs = traj["cost"]
    predicted_unsafe = s < safe_threshold

    unsafe_mask = costs == 1
    safe_mask   = costs == 0
    flagged_unsafe_steps += int(predicted_unsafe[unsafe_mask].sum())
    total_unsafe_steps   += int(unsafe_mask.sum())
    flagged_safe_steps   += int(predicted_unsafe[safe_mask].sum())
    total_safe_steps     += int(safe_mask.sum())

print(f"  Unsafe steps flagged: {flagged_unsafe_steps}/{total_unsafe_steps} "
      f"({flagged_unsafe_steps/max(total_unsafe_steps,1)*100:.1f}%)")
print(f"  Safe steps flagged:   {flagged_safe_steps}/{total_safe_steps} "
      f"({flagged_safe_steps/max(total_safe_steps,1)*100:.1f}%) ← false alarm rate")

# ── Plots ─────────────────────────────────────────────────────────────────────
print("\nSaving plots...")

# Score distribution
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(safe_scores,   bins=60, alpha=0.6, color="#4C9BE8", label=f"safe (n={len(safe_scores)})",   density=True)
ax.hist(unsafe_scores, bins=30, alpha=0.7, color="#E84C4C", label=f"unsafe (n={len(unsafe_scores)})", density=True)
ax.axvline(safe_threshold, color="black", linestyle="--", linewidth=2,
           label=f"conformal threshold={safe_threshold:.3f}")
ax.set_xlabel("Classifier score (positive = safe, negative = unsafe)")
ax.set_ylabel("Density")
ax.set_title("Safety Classifier Score Distribution — Safe vs Unsafe Observations")
ax.legend()
plt.tight_layout()
fig.savefig(OUT_DIR / "score_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {OUT_DIR}/score_distribution.png")

# Score over a high-cost trajectory
costly = sorted(trajs, key=lambda t: t["cost"].sum(), reverse=True)[0]
with torch.no_grad():
    x = torch.tensor(costly["obs"], dtype=torch.float32).to(DEVICE)
    z = encoder(x)
    z_n = (z - z_mean) / z_std
    traj_scores = clf(z_n).cpu().numpy()

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
axes[0].plot(traj_scores, color="#4C9BE8", linewidth=0.8)
axes[0].axhline(safe_threshold, color="black", linestyle="--", label=f"threshold={safe_threshold:.3f}")
axes[0].fill_between(range(len(traj_scores)),
                      traj_scores, safe_threshold,
                      where=np.array(traj_scores) < safe_threshold,
                      alpha=0.3, color="#E84C4C", label="predicted unsafe")
axes[0].set_ylabel("Safety score"); axes[0].legend(fontsize=8)
axes[0].set_title("Classifier safety score over highest-cost trajectory")

axes[1].fill_between(range(len(costly["cost"])), costly["cost"], alpha=0.7, color="#E84C4C")
axes[1].set_ylabel("True cost")

axes[2].plot(costly["reward"], color="#4C9BE8", linewidth=0.8)
axes[2].set_ylabel("Reward"); axes[2].set_xlabel("Step")

plt.tight_layout()
fig.savefig(OUT_DIR / "trajectory_scores.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {OUT_DIR}/trajectory_scores.png")

print(f"\nAll outputs saved to {OUT_DIR}/")
