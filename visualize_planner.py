"""
Visualize MPPI planner trajectories vs random policy.

Records pixel frames from the live environment and saves:
  - side_by_side.gif : random (left) vs MPPI-safe (right)
  - random_traj.gif  : random policy alone
  - mppi_traj.gif    : MPPI planner alone
  - comparison.png   : cost & safety score over time

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
from collections import deque

sys.path.insert(0, ".")
from safe_lewm.config import Config
from safe_lewm.model import SafeJEPA
from safe_lewm.classifier import ObstacleMLP
from safe_lewm.env_utils import SafetyGymWrapper

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT    = os.environ.get("CHECKPOINT",  "/mnt/t7shield/safe_lewm_rho1.pt")
CLF_PATH      = os.environ.get("CLF_PATH",    "/mnt/t7shield/classifier.pt")
OUT_DIR       = Path(os.environ.get("OUT_DIR", "planner_viz"))
MAX_STEPS     = int(os.environ.get("MAX_STEPS",  "500"))
N_SAMPLES     = int(os.environ.get("N_SAMPLES",  "64"))
HORIZON       = int(os.environ.get("HORIZON",    "5"))
TEMPERATURE   = float(os.environ.get("TEMPERATURE", "0.05"))
SAFETY_W      = float(os.environ.get("SAFETY_W",    "20.0"))
SAFETY_MARGIN = float(os.environ.get("SAFETY_MARGIN","1.5"))
HISTORY       = int(os.environ.get("HISTORY",    "3"))
SEED          = int(os.environ.get("SEED",       "0"))
N_SEARCH      = int(os.environ.get("N_SEARCH",   "15"))
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


# ── Camera projection for "fixedfar" (pos="0 -5 5", zaxis="0 -1 1", fovy=45) ──
# Camera axes in world frame:
#   X = (1, 0, 0)              right
#   Y = (0, 1/√2, 1/√2)       up-in-camera = up-right in world
#   Z = (0, -1/√2, 1/√2)      camera "forward-back" axis (looks along -Z)
_S  = 1.0 / np.sqrt(2.0)
_FIXEDFAR_CAM_MAT = np.array([[1.0, 0.0, 0.0],
                               [0.0,  _S,  _S],
                               [0.0, -_S,  _S]], dtype=np.float64)
_FIXEDFAR_CAM_POS = np.array([0.0, -5.0, 5.0], dtype=np.float64)
_FIXEDFAR_FOVY    = 45.0   # degrees

def world_to_pixel(world_xy, width=256, height=256):
    """Project world XY (z=0 ground plane) to pixel for the fixedfar camera."""
    world_xyz = np.array([world_xy[0], world_xy[1], 0.0], dtype=np.float64)
    f   = (height / 2.0) / np.tan(np.radians(_FIXEDFAR_FOVY / 2.0))
    dp  = world_xyz - _FIXEDFAR_CAM_POS
    p_cam = _FIXEDFAR_CAM_MAT @ dp   # camera-space coords
    if p_cam[2] >= 0:                 # point behind camera — fall back to centre
        return (width // 2, height // 2)
    px = int( f * p_cam[0] / (-p_cam[2]) + width  / 2)
    py = int(-f * p_cam[1] / (-p_cam[2]) + height / 2)
    return px, py


def draw_marker(frame, px, py, color, label, radius=8):
    """Filled circle + white outline + label text."""
    import cv2
    H, W = frame.shape[:2]
    if 0 <= px < W and 0 <= py < H:
        cv2.circle(frame, (px, py), radius,     color,         -1)
        cv2.circle(frame, (px, py), radius + 1, (255, 255, 255), 1)
        cv2.putText(frame, label, (px + radius + 2, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
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
    """Encode (C, H, W) numpy obs → (D,) latent tensor."""
    x = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return model.encoder(x).squeeze(0)


def get_safety_score(z):
    """(D,) → scalar safety score (higher = safer)."""
    z_n = (z.unsqueeze(0) - z_mean) / z_std
    with torch.no_grad():
        return clf(z_n).item()


def get_agent_goal_pos(raw_env):
    """Return (agent_xy, goal_xy, goal_dist) from safety_gymnasium Builder."""
    try:
        u = raw_env.unwrapped
        a = np.array(u.task.agent.pos[:2], dtype=np.float64)
        g = np.array(u.task.goal.pos[:2],  dtype=np.float64)
        dist = float(np.linalg.norm(g - a)) + 1e-8
        return a, g, dist
    except Exception:
        return np.zeros(2), np.zeros(2), 999.0


def get_goal_direction(raw_env, wrapper=None):
    """(dx, dy) unit vector toward goal from agent, and goal distance in metres."""
    agent_xy, goal_xy, dist = get_agent_goal_pos(raw_env)
    if dist < 990:
        return (goal_xy - agent_xy) / dist, dist
    return np.zeros(2), 999.0


def safe_goal_step(z_hist_deque, U_warm, goal_dir_action):
    """
    SLS²-style safe-goal controller:
      1. Predict safety of next state under the goal-directed action.
      2. If predicted-safe: take the goal-directed action directly (no MPPI overhead).
      3. If predicted-unsafe: run MPPI (safety cost) with goal-direction bias so the
         agent navigates AROUND the hazard while still aiming for the goal.
    """
    goal_t = torch.tensor(goal_dir_action, dtype=torch.float32, device=DEVICE)
    z_hist = torch.stack(list(z_hist_deque), dim=0).unsqueeze(0)  # (1, H, D)

    # ── Step 1: full-horizon look-ahead safety check ──────────────────────────
    # Repeat goal action for entire HORIZON and check that all predicted states are safe
    a_goal_seq = goal_t.clamp(-1, 1).unsqueeze(0).unsqueeze(0).expand(1, HORIZON, -1)
    z_pred_seq = model.rollout(z_hist, a_goal_seq, HISTORY)[:, -HORIZON:, :]  # (1, H, D)
    z_pred_n   = (z_pred_seq.squeeze(0) - z_mean) / z_std  # (H, D)
    min_score  = clf(z_pred_n).min().item()

    if min_score >= safe_threshold + SAFETY_MARGIN:
        # Predicted safe: go straight to goal — fast, no MPPI needed
        return goal_t.clamp(-1, 1), U_warm

    # ── Step 2: MPPI with safety cost + goal-direction bias ───────────────────
    N = N_SAMPLES
    z_h   = z_hist.expand(N, -1, -1)
    noise = torch.randn(N, HORIZON, cfg.action_dim, device=DEVICE) * 0.5
    U_b   = (U_warm.unsqueeze(0) + noise).clamp(-1, 1)
    # Half the samples are initialised toward goal so MPPI explores around it
    U_b[:N // 2] = (goal_t.unsqueeze(0).unsqueeze(0) + noise[:N // 2]).clamp(-1, 1)

    z_seq    = model.rollout(z_h, U_b, HISTORY)
    z_rolled = z_seq[:, -HORIZON:, :].reshape(N * HORIZON, -1)
    z_flat_n = (z_rolled - z_mean) / z_std
    scores   = clf(z_flat_n)

    pen         = F.softplus(safe_threshold + SAFETY_MARGIN - scores)
    safety_cost = pen.reshape(N, HORIZON).sum(-1)

    beta    = safety_cost.min()
    weights = torch.exp(-(safety_cost - beta) / TEMPERATURE)
    weights = weights / (weights.sum() + 1e-8)

    U_warm = (weights[:, None, None] * U_b).sum(0).clamp(-1, 1)
    action = U_warm[0].clone()
    U_warm = torch.roll(U_warm, -1, dims=0)
    U_warm[-1] = goal_t   # seed next warm-start with goal direction
    return action, U_warm


def make_viz_env(seed):
    """Separate env with fixedfar overhead camera for recording GIFs."""
    import safety_gymnasium
    viz = safety_gymnasium.make(
        cfg.env_name,
        render_mode="rgb_array",
        camera_name="fixedfar",
        width=256,
        height=256,
    )
    viz.reset(seed=seed)
    return viz


def run_episode(env, use_planner=False, seed=SEED):
    """Run one episode; return (frames, costs, scores)."""
    obs = env.reset()
    viz_env = make_viz_env(seed)

    frames, costs, scores = [], [], []

    z_hist_deque = deque(maxlen=HISTORY)
    U_warm = torch.zeros(HORIZON, cfg.action_dim, device=DEVICE)
    z0 = get_z(obs)
    for _ in range(HISTORY):
        z_hist_deque.append(z0)

    # Record starting positions for markers
    agent_xy, goal_xy, goal_dist_start = get_agent_goal_pos(env.env)
    if goal_dist_start > 990:
        # Builder API failed; try vector obs fallback for initial goal dir (no absolute pos available)
        _, goal_dist_start = get_goal_direction_from_vec_obs(env)
        goal_xy = np.zeros(2)   # unknown; markers will be inaccurate but functional
    start_xy = agent_xy.copy()
    sp = world_to_pixel(start_xy)
    gp = world_to_pixel(goal_xy)
    print(f"  Start: {start_xy}  px={sp}")
    print(f"  Goal:  {goal_xy}   px={gp}  dist={goal_dist_start:.2f}m")

    for step in range(MAX_STEPS):
        z_cur = get_z(obs)
        score = get_safety_score(z_cur)
        goal_dir, goal_dist = get_goal_direction(env.env)

        # High-res overhead frame
        raw_viz = viz_env.render()   # (256, 256, 3) uint8
        try:
            import cv2
            frame = raw_viz.copy()

            # Live goal and agent positions for markers
            live_agent_xy, live_goal_xy, _ = get_agent_goal_pos(viz_env)
            cur_gp = world_to_pixel(live_goal_xy)
            cur_ap = world_to_pixel(live_agent_xy)

            # START marker (fixed, cyan) — where agent spawned
            draw_marker(frame, sp[0], sp[1], (0, 220, 255), "S", radius=6)
            # GOAL marker (bright green, slightly larger)
            draw_marker(frame, cur_gp[0], cur_gp[1], (0, 210, 0), "G", radius=9)
            # Current agent position (yellow dot, small)
            draw_marker(frame, cur_ap[0], cur_ap[1], (50, 200, 255), "", radius=4)

            safe_str = "SAFE" if score >= safe_threshold else "UNSAFE"
            color    = (0, 220, 0) if score >= safe_threshold else (255, 50, 50)
            cv2.putText(frame, f"t={step:3d}",          (6, 18),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, safe_str,                 (6, 36),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,           2)
            cv2.putText(frame, f"goal:{goal_dist:.1f}m", (6, 54),  cv2.FONT_HERSHEY_SIMPLEX, 0.45,(220, 220, 80),  1)
        except ImportError:
            frame = raw_viz

        frames.append(frame)
        scores.append(score)

        # Choose action
        if use_planner:
            with torch.no_grad():
                action, U_warm = safe_goal_step(z_hist_deque, U_warm, goal_dir)
            action_np = action.cpu().numpy()
        else:
            action_np = env.action_space.sample()

        next_obs, r, c, done, _ = env.step(action_np)
        viz_env.step(action_np)

        costs.append(c)
        if c > 0:
            try:
                import cv2
                cv2.rectangle(frame, (0, 0), (255, 255), (220, 0, 0), 5)
                frames[-1] = frame
            except Exception:
                pass

        z_hist_deque.append(get_z(next_obs))
        obs = next_obs

        if step % 20 == 0:
            tag = "MPPI" if use_planner else "Random"
            print(f"  [{tag}] step {step:3d} | score={score:+.3f} | cost={int(c)} | cumcost={int(sum(costs))}")

        if done:
            break

    return frames, np.array(costs), np.array(scores)


# ── Search for a seed with: random policy hits hazards AND goal starts far away ─
print(f"\nSearching for good seed (cost>0 and initial goal_dist>2m) ...")
best_seed    = SEED
best_cost    = 0
best_rnd     = None
best_goal_dist = 0.0

for trial_seed in range(SEED, SEED + N_SEARCH):
    env_trial = SafetyGymWrapper(cfg.env_name, cfg.image_size, cfg.frame_stack,
                                  cfg.frame_skip, seed=trial_seed)
    # Check initial goal distance and current safety score before running full episode
    env_trial.reset()
    _, _, init_dist = get_agent_goal_pos(env_trial.env)
    z0_check = get_z(env_trial._get_stacked_obs() if hasattr(env_trial, '_get_stacked_obs') else env_trial.reset())
    init_safety = get_safety_score(z0_check)
    print(f"  seed={trial_seed}: goal_dist={init_dist:.2f}m, init_safety={init_safety:.2f}", end="")

    trial_frames, trial_costs, trial_scores = run_episode(env_trial, use_planner=False, seed=trial_seed)
    total = int(trial_costs.sum())
    print(f" -> random cost={total}")

    # Prefer seeds where: random hits hazards, goal starts far, agent doesn't spawn IN a hazard
    starts_safe = init_safety > 1.0    # agent spawns away from hazards
    starts_far  = init_dist  > 2.5     # goal is not immediately next to agent
    rank = total * 2 + starts_far + starts_safe
    best_rank = best_cost * 2 + (best_goal_dist > 2.5) + 1
    if rank > best_rank or best_rnd is None:
        best_cost      = total
        best_seed      = trial_seed
        best_rnd       = (trial_frames, trial_costs, trial_scores)
        best_goal_dist = init_dist

    if best_cost >= 5 and starts_far and starts_safe:
        break

if best_rnd is None:
    best_rnd = (trial_frames, trial_costs, trial_scores)

print(f"\nBest seed={best_seed}: random cost={best_cost}, initial goal_dist={best_goal_dist:.2f}m")
rnd_frames, rnd_costs, rnd_scores = best_rnd

# ── Run MPPI episode with best seed ───────────────────────────────────────────
env_mppi = SafetyGymWrapper(cfg.env_name, cfg.image_size, cfg.frame_stack,
                             cfg.frame_skip, seed=best_seed)

print(f"\n{'='*50}")
print(f"Random policy (seed={best_seed}) — total cost: {int(rnd_costs.sum())} | Steps: {len(rnd_costs)}")

print(f"\n{'='*50}")
print(f"Running MPPI-SAFE planner (seed={best_seed})...")
mppi_frames, mppi_costs, mppi_scores = run_episode(env_mppi, use_planner=True, seed=best_seed)
print(f"  Total cost: {int(mppi_costs.sum())} | Steps: {len(mppi_costs)}")

# ── Save GIFs ─────────────────────────────────────────────────────────────────
try:
    import imageio
    import cv2
    print("\nSaving GIFs...")

    imageio.mimsave(str(OUT_DIR / "random_traj.gif"),  rnd_frames,  fps=FPS)
    imageio.mimsave(str(OUT_DIR / "mppi_traj.gif"),    mppi_frames, fps=FPS)
    print(f"  Saved {OUT_DIR}/random_traj.gif")
    print(f"  Saved {OUT_DIR}/mppi_traj.gif")

    # Side-by-side GIF
    n = min(len(rnd_frames), len(mppi_frames))
    label_h = 22
    side_frames = []
    for i in range(n):
        rnd_f  = rnd_frames[i]
        mppi_f = mppi_frames[i]
        H, W, _ = rnd_f.shape
        rnd_label  = np.zeros((label_h, W, 3), dtype=np.uint8)
        mppi_label = np.zeros((label_h, W, 3), dtype=np.uint8)
        cv2.putText(rnd_label,  "RANDOM",    (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(mppi_label, "MPPI-SAFE", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 100), 1)
        rnd_col  = np.vstack([rnd_label,  rnd_f])
        mppi_col = np.vstack([mppi_label, mppi_f])
        divider  = np.ones((H + label_h, 4, 3), dtype=np.uint8) * 200
        side_frames.append(np.hstack([rnd_col, divider, mppi_col]))

    imageio.mimsave(str(OUT_DIR / "side_by_side.gif"), side_frames, fps=FPS)
    print(f"  Saved {OUT_DIR}/side_by_side.gif")

except ImportError:
    print("  imageio not found — skipping GIFs")

# ── Comparison plot ────────────────────────────────────────────────────────────
print("Saving comparison plot...")
n_rnd  = len(rnd_costs)
n_mppi = len(mppi_costs)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

ax = axes[0, 0]
ax.plot(np.cumsum(rnd_costs),  color="#E84C4C", label=f"Random   (total={int(rnd_costs.sum())})")
ax.plot(np.cumsum(mppi_costs), color="#4C9BE8", label=f"MPPI-safe (total={int(mppi_costs.sum())})")
ax.set_title("Cumulative Cost"); ax.set_xlabel("Step"); ax.set_ylabel("Cost")
ax.legend()

ax = axes[0, 1]
ax.plot(rnd_scores,  color="#E84C4C", linewidth=0.7, alpha=0.8, label="Random")
ax.plot(mppi_scores, color="#4C9BE8", linewidth=0.7, alpha=0.8, label="MPPI-safe")
ax.axhline(safe_threshold, color="black", linestyle="--", label=f"threshold={safe_threshold:.2f}")
ax.set_title("Safety Score Over Time"); ax.set_xlabel("Step"); ax.set_ylabel("Score")
ax.legend()

ax = axes[1, 0]
ax.fill_between(range(n_rnd),  rnd_costs,  alpha=0.5, color="#E84C4C", label="Random")
ax.fill_between(range(n_mppi), mppi_costs, alpha=0.5, color="#4C9BE8", label="MPPI-safe")
ax.set_title("Cost Per Step"); ax.set_xlabel("Step"); ax.set_ylabel("Cost")
ax.legend()

ax = axes[1, 1]
ax.hist(rnd_scores,  bins=40, alpha=0.6, color="#E84C4C", density=True, label="Random")
ax.hist(mppi_scores, bins=40, alpha=0.6, color="#4C9BE8", density=True, label="MPPI-safe")
ax.axvline(safe_threshold, color="black", linestyle="--", label="threshold")
ax.set_title("Safety Score Distribution"); ax.set_xlabel("Score"); ax.set_ylabel("Density")
ax.legend()

plt.suptitle(
    f"Random vs MPPI-Safe | seed={best_seed} | Random cost={int(rnd_costs.sum())}  MPPI cost={int(mppi_costs.sum())}",
    fontsize=12, fontweight="bold",
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
