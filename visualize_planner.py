"""
Visualize MPPI planner trajectories vs random policy.

Records pixel frames from the live environment and saves:
  - side_by_side.gif : random (left) vs MPPI-safe (right)
  - random_traj.gif  : random policy alone
  - mppi_traj.gif    : MPPI planner alone
  - comparison.png   : cost & safety score over time

Speed tip: use small N_SAMPLES (64) and short HORIZON (5) — planning quality
           matters less than showing avoidance behaviour.

Usage:
    MUJOCO_GL=osmesa python visualize_planner.py
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
from safe_lewm.env_utils import SafetyGymWrapper

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT    = os.environ.get("CHECKPOINT",  "/mnt/t7shield/safe_lewm_rho1.pt")
CLF_PATH      = os.environ.get("CLF_PATH",    "/mnt/t7shield/classifier.pt")
OUT_DIR       = Path(os.environ.get("OUT_DIR", "planner_viz"))
MAX_STEPS     = int(os.environ.get("MAX_STEPS",  "200"))
N_SAMPLES     = int(os.environ.get("N_SAMPLES",  "64"))
HORIZON       = int(os.environ.get("HORIZON",    "5"))
TEMPERATURE   = float(os.environ.get("TEMPERATURE", "0.05"))
SAFETY_W      = float(os.environ.get("SAFETY_W",    "20.0"))
SAFETY_MARGIN = float(os.environ.get("SAFETY_MARGIN","0.5"))
HISTORY       = int(os.environ.get("HISTORY",    "3"))
SEED          = int(os.environ.get("SEED",       "0"))
N_SEARCH      = int(os.environ.get("N_SEARCH",   "10"))   # episodes to search for a costly one
FPS           = int(os.environ.get("FPS",        "8"))
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────────────────────────

OUT_DIR.mkdir(exist_ok=True)
print(f"Device: {DEVICE} | N_SAMPLES={N_SAMPLES} | HORIZON={HORIZON} | MAX_STEPS={MAX_STEPS}")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def denorm(frame_chw):
    """(3, H, W) normalized -> (H, W, 3) uint8"""
    f = frame_chw * IMAGENET_STD[:, None, None] + IMAGENET_MEAN[:, None, None]
    return (f.clip(0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)

def add_overlay(frame_hwc, cost, score, threshold, step):
    """Draw cost/score text and red border if unsafe."""
    import cv2
    frame = frame_hwc.copy()
    if cost > 0:
        cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (255, 0, 0), 3)
    safe_str = "SAFE" if score >= threshold else "UNSAFE"
    color = (0, 200, 0) if score >= threshold else (255, 50, 50)
    cv2.putText(frame, f"t={step}", (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
    cv2.putText(frame, safe_str,   (2, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    return frame

# ── Load model ─────────────────────────────────────────────────────────────────
print("Loading world model...")
cfg = Config()
model = SafeJEPA(cfg).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False))
model.eval()
for p in model.parameters():
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


def get_z(obs_np):
    """Encode a single observation (C, H, W) numpy -> (D,) tensor."""
    x = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return model.encoder(x).squeeze(0)

def get_safety_score(z):
    """(D,) -> scalar safety score."""
    z_n = (z.unsqueeze(0) - z_mean) / z_std
    with torch.no_grad():
        return clf(z_n).item()

def mppi_step(z_hist_deque, U_warm):
    """
    One MPPI planning step.
    z_hist_deque: deque of (D,) tensors, length HISTORY
    U_warm: (HORIZON, A) warm-start action tensor
    Returns: (A,) action, updated U_warm
    """
    z_hist = torch.stack(list(z_hist_deque), dim=0).unsqueeze(0).expand(N_SAMPLES, -1, -1)
    # Goal: center of safe latent distribution (zero in normalized space ≈ mean latent)
    # We use the safe cluster center — pull toward low-norm latents to stay safe
    z_goal = torch.zeros(1, model.cfg.z_dim, device=DEVICE)

    noise = torch.randn(N_SAMPLES, HORIZON, cfg.action_dim, device=DEVICE)
    U_b   = (U_warm.unsqueeze(0) + noise).clamp(-1, 1)

    z_seq    = model.rollout(z_hist, U_b, HISTORY)
    z_rolled = z_seq[:, -HORIZON:, :]   # (N, H, D)

    # Goal cost
    goal_cost = (z_rolled[:, -1, :] - z_goal).pow(2).sum(-1)

    # Safety cost
    z_flat   = z_rolled.reshape(N_SAMPLES * HORIZON, -1)
    z_flat_n = (z_flat - z_mean) / z_std
    scores   = clf(z_flat_n)                                   # (N*H,)
    pen      = F.softplus(safe_threshold + SAFETY_MARGIN - scores)
    safety_cost = pen.reshape(N_SAMPLES, HORIZON).sum(-1)

    total = goal_cost + SAFETY_W * safety_cost
    beta    = total.min()
    weights = torch.exp(-(total - beta) / TEMPERATURE)
    weights = weights / (weights.sum() + 1e-8)

    U_warm = (weights[:, None, None] * U_b).sum(0).clamp(-1, 1)
    action = U_warm[0].clone()
    U_warm = torch.roll(U_warm, -1, dims=0)
    U_warm[-1] = 0.0
    return action, U_warm


def run_episode(env, use_planner=False, seed=SEED):
    """Run one episode, return frames, costs, safety scores."""
    obs = env.reset()
    frames, costs, scores = [], [], []

    from collections import deque
    z_hist_deque = deque(maxlen=HISTORY)
    U_warm = torch.zeros(HORIZON, cfg.action_dim, device=DEVICE)

    # Seed initial latent history
    z0 = get_z(obs)
    for _ in range(HISTORY):
        z_hist_deque.append(z0)

    for step in range(MAX_STEPS):
        # Current safety score
        z_cur = get_z(obs)
        score = get_safety_score(z_cur)

        # Record pixel frame with overlay
        raw = denorm(obs[:3])
        try:
            import cv2
            frame = add_overlay(raw, 0, score, safe_threshold, step)
        except ImportError:
            frame = raw
        frames.append(frame)
        scores.append(score)

        # Action
        if use_planner:
            with torch.no_grad():
                action, U_warm = mppi_step(z_hist_deque, U_warm)
            action_np = action.cpu().numpy()
        else:
            action_np = env.action_space.sample()

        next_obs, r, c, done, _ = env.step(action_np)

        # Overlay cost on frame after step (we now know if we hit something)
        costs.append(c)
        if c > 0:
            # Re-draw with red border
            try:
                frame = add_overlay(raw, c, score, safe_threshold, step)
                frames[-1] = frame
            except Exception:
                pass

        # Update latent history
        z_next = get_z(next_obs)
        z_hist_deque.append(z_next)
        obs = next_obs

        if step % 20 == 0:
            tag = "MPPI" if use_planner else "Random"
            print(f"  [{tag}] step {step:3d} | score={score:+.3f} | cost={int(c)} | cumcost={int(sum(costs))}")

        if done:
            break

    return frames, np.array(costs), np.array(scores)


# ── Find a seed where random policy hits hazards ───────────────────────────────
# Hazards are sparse (~0.9% of steps) so we search across seeds.
print(f"\nSearching for a seed where random policy hits hazards (up to {N_SEARCH} tries)...")
best_seed = SEED
best_cost = 0
best_rnd  = None

for trial_seed in range(SEED, SEED + N_SEARCH):
    env_trial = SafetyGymWrapper(cfg.env_name, cfg.image_size, cfg.frame_stack,
                                  cfg.frame_skip, seed=trial_seed)
    trial_frames, trial_costs, trial_scores = run_episode(env_trial, use_planner=False)
    total = int(trial_costs.sum())
    print(f"  seed={trial_seed}: random cost={total}")
    if total > best_cost:
        best_cost  = total
        best_seed  = trial_seed
        best_rnd   = (trial_frames, trial_costs, trial_scores)
    if best_cost >= 3:   # good enough — stop searching
        break

if best_rnd is None:
    # Use whatever we have from last trial
    best_rnd = (trial_frames, trial_costs, trial_scores)

print(f"\nBest seed={best_seed} with random cost={best_cost}")
rnd_frames, rnd_costs, rnd_scores = best_rnd

# ── Run episodes ───────────────────────────────────────────────────────────────
env = SafetyGymWrapper(cfg.env_name, cfg.image_size, cfg.frame_stack, cfg.frame_skip, seed=best_seed)

print(f"\n{'='*50}")
print(f"Random policy (seed={best_seed}) — already recorded above")
print(f"  Total cost: {int(rnd_costs.sum())} | Steps: {len(rnd_costs)}")

print(f"\n{'='*50}")
print(f"Running MPPI-SAFE planner (same seed={best_seed})...")
mppi_frames, mppi_costs, mppi_scores = run_episode(env, use_planner=True)
print(f"  Total cost: {int(mppi_costs.sum())} | Steps: {len(mppi_costs)}")

# ── Save GIFs ─────────────────────────────────────────────────────────────────
try:
    import imageio
    print("\nSaving GIFs...")

    imageio.mimsave(str(OUT_DIR / "random_traj.gif"),  rnd_frames,  fps=FPS)
    imageio.mimsave(str(OUT_DIR / "mppi_traj.gif"),    mppi_frames, fps=FPS)
    print(f"  Saved {OUT_DIR}/random_traj.gif")
    print(f"  Saved {OUT_DIR}/mppi_traj.gif")

    # Side-by-side GIF (pad to same length)
    n = min(len(rnd_frames), len(mppi_frames))
    label_h = 20
    import cv2
    side_frames = []
    for i in range(n):
        rnd_f   = rnd_frames[i]
        mppi_f  = mppi_frames[i]
        H, W, C = rnd_f.shape
        # Label bars
        rnd_label  = np.zeros((label_h, W, 3), dtype=np.uint8)
        mppi_label = np.zeros((label_h, W, 3), dtype=np.uint8)
        try:
            cv2.putText(rnd_label,  "RANDOM",    (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
            cv2.putText(mppi_label, "MPPI-SAFE", (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100,220,100), 1)
        except Exception:
            pass
        rnd_col  = np.vstack([rnd_label,  rnd_f])
        mppi_col = np.vstack([mppi_label, mppi_f])
        divider  = np.ones((H + label_h, 4, 3), dtype=np.uint8) * 200
        combined = np.hstack([rnd_col, divider, mppi_col])
        side_frames.append(combined)

    imageio.mimsave(str(OUT_DIR / "side_by_side.gif"), side_frames, fps=FPS)
    print(f"  Saved {OUT_DIR}/side_by_side.gif")

except ImportError:
    print("  imageio not found — skipping GIFs (pip install imageio)")

# ── Comparison plot ────────────────────────────────────────────────────────────
print("Saving comparison plot...")
n_rnd  = len(rnd_costs)
n_mppi = len(mppi_costs)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Cumulative cost
ax = axes[0, 0]
ax.plot(np.cumsum(rnd_costs),  color="#E84C4C", label=f"Random  (total={int(rnd_costs.sum())})")
ax.plot(np.cumsum(mppi_costs), color="#4C9BE8", label=f"MPPI-safe (total={int(mppi_costs.sum())})")
ax.set_title("Cumulative Cost"); ax.set_xlabel("Step"); ax.set_ylabel("Cumulative cost")
ax.legend()

# Safety score over time
ax = axes[0, 1]
ax.plot(rnd_scores,  color="#E84C4C", linewidth=0.7, alpha=0.8, label="Random")
ax.plot(mppi_scores, color="#4C9BE8", linewidth=0.7, alpha=0.8, label="MPPI-safe")
ax.axhline(safe_threshold, color="black", linestyle="--", label=f"threshold={safe_threshold:.2f}")
ax.set_title("Safety Score Over Time"); ax.set_xlabel("Step"); ax.set_ylabel("Safety score")
ax.legend()

# Cost per step
ax = axes[1, 0]
ax.fill_between(range(n_rnd),  rnd_costs,  alpha=0.5, color="#E84C4C", label="Random")
ax.fill_between(range(n_mppi), mppi_costs, alpha=0.5, color="#4C9BE8", label="MPPI-safe")
ax.set_title("Cost Per Step"); ax.set_xlabel("Step"); ax.set_ylabel("Cost")
ax.legend()

# Score distribution comparison
ax = axes[1, 1]
ax.hist(rnd_scores,  bins=40, alpha=0.6, color="#E84C4C", density=True, label="Random")
ax.hist(mppi_scores, bins=40, alpha=0.6, color="#4C9BE8", density=True, label="MPPI-safe")
ax.axvline(safe_threshold, color="black", linestyle="--", label="threshold")
ax.set_title("Safety Score Distribution"); ax.set_xlabel("Score"); ax.set_ylabel("Density")
ax.legend()

plt.suptitle(
    f"Random vs MPPI-Safe | Random cost={int(rnd_costs.sum())}  MPPI cost={int(mppi_costs.sum())}",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
fig.savefig(OUT_DIR / "comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {OUT_DIR}/comparison.png")

print(f"\n{'='*50}")
print(f"Random:    total_cost={int(rnd_costs.sum())}, steps={n_rnd}")
print(f"MPPI-safe: total_cost={int(mppi_costs.sum())}, steps={n_mppi}")
cost_red = (rnd_costs.sum() - mppi_costs.sum()) / (rnd_costs.sum() + 1e-8) * 100
print(f"Cost reduction: {cost_red:.1f}%")
print(f"\nAll outputs saved to {OUT_DIR}/")
