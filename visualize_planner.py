"""
Visualize MPPI planner: without regularization (rho=0) vs with regularization (rho=1).

Records pixel frames from the live environment and saves:
  - side_by_side.gif   : without reg (left) vs with reg (right)
  - noreg_traj.gif     : without regularization alone
  - reg_traj.gif       : with regularization alone
  - comparison.png     : cost & safety score over time

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
CHECKPOINT       = os.environ.get("CHECKPOINT",       "/mnt/t7shield/safe_lewm_rho1.pt")
CHECKPOINT_NOREG = os.environ.get("CHECKPOINT_NOREG", "/mnt/t7shield/safe_lewm_rho0.pt")
CLF_PATH         = os.environ.get("CLF_PATH",         "/mnt/t7shield/classifier.pt")
CLF_NOREG_PATH   = os.environ.get("CLF_NOREG_PATH",   "")   # optional: separate classifier for rho=0 model
ENV_NAME         = os.environ.get("ENV_NAME",         "SafetyPointGoal2-v0")
OUT_DIR       = Path(os.environ.get("OUT_DIR", "planner_viz"))
MAX_STEPS     = int(os.environ.get("MAX_STEPS",  "500"))
N_SAMPLES     = int(os.environ.get("N_SAMPLES",  "64"))
HORIZON       = int(os.environ.get("HORIZON",    "5"))
TEMPERATURE   = float(os.environ.get("TEMPERATURE", "0.05"))
SAFETY_W      = float(os.environ.get("SAFETY_W",    "20.0"))
SAFETY_MARGIN = float(os.environ.get("SAFETY_MARGIN","-0.5"))
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


# ── Load models ────────────────────────────────────────────────────────────────
cfg = Config()

print("Loading world model (with regularization, rho=1)...")
model = SafeJEPA(cfg).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False))
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

print("Loading world model (without regularization, rho=0)...")
model_noreg = SafeJEPA(cfg).to(DEVICE)
model_noreg.load_state_dict(torch.load(CHECKPOINT_NOREG, map_location=DEVICE, weights_only=False))
model_noreg.eval()
for p in model_noreg.parameters():
    p.requires_grad_(False)

# ── Load classifier(s) ────────────────────────────────────────────────────────
def load_clf(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    c = ObstacleMLP(
        z_dim=ckpt["z_dim"], hidden_dim=ckpt["hidden_dim"],
        depth=ckpt["depth"], dropout=ckpt.get("dropout", 0.1),
    ).to(DEVICE)
    c.load_state_dict(ckpt["classifier_state"])
    c.eval()
    return c, ckpt["z_mean"].to(DEVICE), ckpt["z_std"].to(DEVICE), ckpt["safe_threshold"]

print("Loading classifier (with regularization)...")
clf, z_mean, z_std, safe_threshold = load_clf(CLF_PATH)
print(f"  Conformal threshold: {safe_threshold:.4f}")

if CLF_NOREG_PATH:
    print("Loading classifier (without regularization)...")
    clf_noreg, z_mean_noreg, z_std_noreg, safe_threshold_noreg = load_clf(CLF_NOREG_PATH)
    print(f"  Conformal threshold (noreg): {safe_threshold_noreg:.4f}")
else:
    print("  CLF_NOREG_PATH not set — using shared classifier for both models")
    clf_noreg, z_mean_noreg, z_std_noreg, safe_threshold_noreg = clf, z_mean, z_std, safe_threshold


def get_z(obs_np, world_model=None):
    """Encode (C, H, W) numpy obs → (D,) latent tensor."""
    m = world_model if world_model is not None else model
    x = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return m.encoder(x).squeeze(0)


def get_safety_score(z, classifier=None, zm=None, zs=None):
    """(D,) → scalar safety score (higher = safer)."""
    c  = classifier if classifier is not None else clf
    zm = zm if zm is not None else z_mean
    zs = zs if zs is not None else z_std
    z_n = (z.unsqueeze(0) - zm) / zs
    with torch.no_grad():
        return c(z_n).item()


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


def get_robot_heading(raw_env):
    """
    Return the robot's yaw angle (radians) from its freejoint quaternion.
    qpos layout for a freejoint: [x, y, z, qw, qx, qy, qz, ...].
    """
    try:
        u = raw_env.unwrapped
        qpos = np.array(u.task.data.qpos)
        if len(qpos) >= 7:
            qw, qz = float(qpos[3]), float(qpos[6])
            return 2.0 * np.arctan2(qz, qw)
    except Exception:
        pass
    return 0.0


def get_agent_heading(raw_env):
    """Return the agent's current world-frame heading in radians."""
    try:
        u = raw_env.unwrapped
        agent_id = u.task.model.body('agent').id
        xquat = np.array(u.task.data.xquat[agent_id])  # (qw, qx, qy, qz)
        return float(2.0 * np.arctan2(xquat[3], xquat[0]))
    except Exception:
        return 0.0


def get_goal_direction(raw_env, wrapper=None):
    """
    Unicycle controller for the PointRobot.

    The PointRobot has body-frame actuation:
      - action[0]: forward/backward force in current heading direction
      - action[1]: angular velocity (turns the body)

    Controller:
      - Compute angle to goal (alpha) and heading error (delta = alpha - theta)
      - action[0] = cos(delta): full forward when facing goal, brakes when facing away
      - action[1] = k_turn * sin(delta): turn toward goal
      - Scale by distance to slow down near goal

    This is a pure proportional controller — MuJoCo's joint damping provides
    natural dissipation and prevents infinite oscillation.
    """
    agent_xy, goal_xy, dist = get_agent_goal_pos(raw_env)
    if dist >= 990:
        return np.zeros(2), 999.0

    alpha = float(np.arctan2(goal_xy[1] - agent_xy[1], goal_xy[0] - agent_xy[0]))
    theta = get_agent_heading(raw_env)
    delta = alpha - theta
    # Normalize to [-pi, pi]
    delta = (delta + np.pi) % (2 * np.pi) - np.pi

    k_turn = 2.0   # turning gain (≥1 to turn aggressively toward goal)
    speed  = min(1.0, dist)  # proportional to distance, max 1

    action = np.array([speed * np.cos(delta), k_turn * np.sin(delta)])
    norm = float(np.linalg.norm(action))
    return (action / norm if norm > 1.0 else action), dist


def safe_goal_step(z_hist_deque, U_warm, goal_dir_action, current_score, recent_cost=0, goal_dist=999.0, world_model=None, classifier=None, zm=None, zs=None, thr=None):
    """
    Reactive-predictive safe-goal controller:
      - SAFE: go straight to goal (PD action from get_goal_direction).
      - UNSAFE: MPPI escape biased toward goal.

    MPPI is triggered reactively (actual hazard contact) or when the world
    model is very confident the current state is unsafe (score < -1.5).
    Using actual cost avoids false-positive triggers from world-model calibration
    errors that would send the agent on destructive detours away from the goal.
    """
    goal_t = torch.tensor(goal_dir_action, dtype=torch.float32, device=DEVICE)

    # Trigger MPPI ONLY on actual hazard contact.
    # Score-based triggers fire on world-model false positives far from any real hazard,
    # trapping the robot in long MPPI spirals away from the goal.
    # Don't activate MPPI if the robot is already close to the goal — at that range
    # MPPI escape actions overshoot and push the robot far away.  Trust the PD
    # controller to navigate the last stretch (it will incur some cost but will
    # actually capture the goal, unlike MPPI which deflects away).
    # Don't activate MPPI if the robot is close to the goal — MPPI escape actions
    # at close range push the robot far away.  Trust the PD controller inside 1.2m
    # (the orbit converges below 1.2m and will naturally reach capture distance).
    use_mppi = (recent_cost > 0) and (goal_dist > 1.2)
    if not use_mppi:
        # Safe: go straight to goal at full speed
        return goal_t.clamp(-1, 1), U_warm

    # Unsafe: MPPI escape. Half samples biased toward goal so the agent navigates
    # AROUND the hazard rather than reversing away from it.
    z_hist = torch.stack(list(z_hist_deque), dim=0).unsqueeze(0)
    N   = N_SAMPLES
    z_h = z_hist.expand(N, -1, -1)

    noise = torch.randn(N, HORIZON, cfg.action_dim, device=DEVICE) * 0.5
    U_b  = (U_warm.unsqueeze(0) + noise).clamp(-1, 1)
    U_b[:N // 2] = (goal_t.unsqueeze(0).unsqueeze(0) + noise[:N // 2]).clamp(-1, 1)

    m   = world_model if world_model is not None else model
    c   = classifier if classifier is not None else clf
    _zm = zm if zm is not None else z_mean
    _zs = zs if zs is not None else z_std
    _thr = thr if thr is not None else safe_threshold

    z_seq    = m.rollout(z_h, U_b, HISTORY)
    z_rolled = z_seq[:, -HORIZON:, :].reshape(N * HORIZON, -1)
    z_flat_n = (z_rolled - _zm) / _zs
    scores   = c(z_flat_n)

    pen         = F.softplus(_thr + SAFETY_MARGIN - scores)
    safety_cost = pen.reshape(N, HORIZON).sum(-1)

    # Goal-attraction cost: prefer escape paths that stay near the goal direction.
    GOAL_ALPHA  = 5.0
    goal_expand = goal_t.unsqueeze(0).unsqueeze(0)  # (1, 1, A)
    goal_cost   = ((U_b - goal_expand) ** 2).mean(-1).sum(-1)  # (N,)
    total_cost  = safety_cost + GOAL_ALPHA * goal_cost

    beta    = total_cost.min()
    weights = torch.exp(-(total_cost - beta) / TEMPERATURE)
    weights = weights / (weights.sum() + 1e-8)

    U_warm = (weights[:, None, None] * U_b).sum(0).clamp(-1, 1)
    action = U_warm[0].clone()
    U_warm = torch.roll(U_warm, -1, dims=0)
    U_warm[-1] = goal_t
    return action, U_warm


NUM_HAZARDS = 4   # fewer cost-inducing hazard cylinders (default is 8)
NUM_VASES   = 8   # more pushable obstacle boxes (default is 1)


def make_viz_env(seed):
    """Separate env with fixedfar overhead camera for recording GIFs."""
    import safety_gymnasium
    viz = safety_gymnasium.make(
        ENV_NAME,
        render_mode="rgb_array",
        camera_name="fixedfar",
        width=256,
        height=256,
    )
    viz.unwrapped.task.hazards.num = NUM_HAZARDS
    viz.unwrapped.task.vases.num   = NUM_VASES
    viz.reset(seed=seed)
    return viz


def run_episode(env, use_planner=False, seed=SEED, world_model=None, classifier=None, zm=None, zs=None, thr=None):
    """Run one episode; return (frames, costs, scores)."""
    obs = env.reset()
    viz_env = make_viz_env(seed)

    frames, costs, scores = [], [], []

    z_hist_deque = deque(maxlen=HISTORY)
    U_warm = torch.zeros(HORIZON, cfg.action_dim, device=DEVICE)
    z0 = get_z(obs, world_model)
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

    prev_cost = 0  # cost from previous env.step; used as reactive MPPI trigger
    for step in range(MAX_STEPS):
        z_cur = get_z(obs, world_model)
        score = get_safety_score(z_cur, classifier, zm, zs)
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

            # Current agent position (yellow dot, small)
            draw_marker(frame, cur_ap[0], cur_ap[1], (50, 200, 255), "", radius=4)

            _thr_disp = thr if thr is not None else safe_threshold
            safe_str = "SAFE" if score >= _thr_disp else "UNSAFE"
            color    = (0, 220, 0) if score >= _thr_disp else (255, 50, 50)
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
                action, U_warm = safe_goal_step(z_hist_deque, U_warm, goal_dir, score, prev_cost, goal_dist, world_model, classifier, zm, zs, thr)
            action_np = action.cpu().numpy()
            # Debug every 10 steps
            if step % 5 == 0:
                branch = "MPPI" if prev_cost > 0 else "STRAIGHT"
                a_xy, g_xy, _ = get_agent_goal_pos(env.env)
                heading_deg = float(np.degrees(get_robot_heading(env.env)))
                try:
                    _vel = np.array(env.env.unwrapped.task.data.qvel[:2], dtype=np.float64)
                    vel_str = f"({_vel[0]:+.3f},{_vel[1]:+.3f})"
                except Exception:
                    vel_str = "(?,?)"
                print(f"    [dbg] step={step:3d} | branch={branch} | dist={goal_dist:.3f}m"
                      f" | agent=({a_xy[0]:+.3f},{a_xy[1]:+.3f})"
                      f" | goal=({g_xy[0]:+.3f},{g_xy[1]:+.3f})"
                      f" | hdg={heading_deg:+.0f}° | vel={vel_str}"
                      f" | action=({action_np[0]:+.3f},{action_np[1]:+.3f})")
        else:
            action_np = env.action_space.sample()

        next_obs, r, c, done, _ = env.step(action_np)
        viz_env.step(action_np)
        prev_cost = c

        costs.append(c)
        if c > 0:
            try:
                import cv2
                cv2.rectangle(frame, (0, 0), (255, 255), (220, 0, 0), 5)
                frames[-1] = frame
            except Exception:
                pass

        z_hist_deque.append(get_z(next_obs, world_model))
        obs = next_obs

        if step % 20 == 0:
            tag = "WithReg" if use_planner else "NoReg"
            print(f"  [{tag}] step {step:3d} | score={score:+.3f} | cost={int(c)} | cumcost={int(sum(costs))}")

        if done:
            break

    return frames, np.array(costs), np.array(scores)


# ── Multi-seed evaluation ──────────────────────────────────────────────────────
EVAL_SEEDS = int(os.environ.get("EVAL_SEEDS", "20"))
GIF_SEED   = SEED  # seed used for GIF recording (first one)

def make_env(seed):
    e = SafetyGymWrapper(ENV_NAME, cfg.image_size, cfg.frame_stack, cfg.frame_skip, seed=seed)
    e.env.unwrapped.task.hazards.num = NUM_HAZARDS
    e.env.unwrapped.task.vases.num   = NUM_VASES
    return e

print(f"\nEvaluating {EVAL_SEEDS} seeds (SEED {SEED} … {SEED + EVAL_SEEDS - 1}) ...")
print(f"GIFs will be recorded for seed={GIF_SEED}\n")

noreg_total_costs, reg_total_costs = [], []
noreg_all_scores,  reg_all_scores  = [], []
gif_noreg_frames = gif_noreg_costs = gif_noreg_scores = None
gif_reg_frames   = gif_reg_costs   = gif_reg_scores   = None

for i, seed in enumerate(range(SEED, SEED + EVAL_SEEDS)):
    record_gif = (seed == GIF_SEED)
    print(f"  [{i+1:2d}/{EVAL_SEEDS}] seed={seed}", end="  ", flush=True)

    env_nr = make_env(seed)
    nr_frames, nr_costs, nr_scores = run_episode(env_nr, use_planner=True, seed=seed,
        world_model=model_noreg, classifier=clf_noreg,
        zm=z_mean_noreg, zs=z_std_noreg, thr=safe_threshold_noreg)
    noreg_total_costs.append(int(nr_costs.sum()))
    noreg_all_scores.extend(nr_scores.tolist())

    env_r = make_env(seed)
    r_frames, r_costs, r_scores = run_episode(env_r, use_planner=True, seed=seed,
        world_model=model, classifier=clf,
        zm=z_mean, zs=z_std, thr=safe_threshold)
    reg_total_costs.append(int(r_costs.sum()))
    reg_all_scores.extend(r_scores.tolist())

    print(f"noreg_cost={int(nr_costs.sum()):3d}  reg_cost={int(r_costs.sum()):3d}")

    if record_gif:
        gif_noreg_frames, gif_noreg_costs, gif_noreg_scores = nr_frames, nr_costs, nr_scores
        gif_reg_frames,   gif_reg_costs,   gif_reg_scores   = r_frames,  r_costs,  r_scores

noreg_total_costs = np.array(noreg_total_costs)
reg_total_costs   = np.array(reg_total_costs)
noreg_all_scores  = np.array(noreg_all_scores)
reg_all_scores    = np.array(reg_all_scores)

print(f"\n{'='*50}")
print(f"Without regularization — mean cost: {noreg_total_costs.mean():.1f} ± {noreg_total_costs.std():.1f}")
print(f"With    regularization — mean cost: {reg_total_costs.mean():.1f}   ± {reg_total_costs.std():.1f}")
cost_red = (noreg_total_costs.mean() - reg_total_costs.mean()) / (noreg_total_costs.mean() + 1e-8) * 100
print(f"Cost reduction with regularization: {cost_red:.1f}%")

# ── Save GIFs (for GIF_SEED) ──────────────────────────────────────────────────
if gif_noreg_frames is not None:
    try:
        import imageio
        import cv2
        print("\nSaving GIFs...")

        imageio.mimsave(str(OUT_DIR / "noreg_traj.gif"), gif_noreg_frames, fps=FPS)
        imageio.mimsave(str(OUT_DIR / "reg_traj.gif"),   gif_reg_frames,   fps=FPS)
        print(f"  Saved {OUT_DIR}/noreg_traj.gif")
        print(f"  Saved {OUT_DIR}/reg_traj.gif")

        n = min(len(gif_noreg_frames), len(gif_reg_frames))
        label_h = 22
        side_frames = []
        for i in range(n):
            noreg_f = gif_noreg_frames[i]
            reg_f   = gif_reg_frames[i]
            H, W, _ = noreg_f.shape
            noreg_label = np.zeros((label_h, W, 3), dtype=np.uint8)
            reg_label   = np.zeros((label_h, W, 3), dtype=np.uint8)
            cv2.putText(noreg_label, "WITHOUT REG", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
            cv2.putText(reg_label,   "WITH REG",    (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 100), 1)
            noreg_col = np.vstack([noreg_label, noreg_f])
            reg_col   = np.vstack([reg_label,   reg_f])
            divider   = np.ones((H + label_h, 4, 3), dtype=np.uint8) * 200
            side_frames.append(np.hstack([noreg_col, divider, reg_col]))

        imageio.mimsave(str(OUT_DIR / "side_by_side.gif"), side_frames, fps=FPS)
        print(f"  Saved {OUT_DIR}/side_by_side.gif")
    except ImportError:
        print("  imageio not found — skipping GIFs")

# ── Aggregate comparison plot ──────────────────────────────────────────────────
print("Saving comparison plot...")
seeds_x = np.arange(EVAL_SEEDS)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Per-seed bar chart
ax = axes[0, 0]
w = 0.35
ax.bar(seeds_x - w/2, noreg_total_costs, w, color="#E84C4C", alpha=0.8, label="Without reg")
ax.bar(seeds_x + w/2, reg_total_costs,   w, color="#4C9BE8", alpha=0.8, label="With reg")
ax.axhline(noreg_total_costs.mean(), color="#E84C4C", linestyle="--", linewidth=1.5, label=f"No-reg mean={noreg_total_costs.mean():.1f}")
ax.axhline(reg_total_costs.mean(),   color="#4C9BE8", linestyle="--", linewidth=1.5, label=f"Reg mean={reg_total_costs.mean():.1f}")
ax.set_xticks(seeds_x); ax.set_xticklabels([str(SEED + i) for i in range(EVAL_SEEDS)], rotation=45, fontsize=7)
ax.set_title("Total Cost per Seed"); ax.set_xlabel("Seed"); ax.set_ylabel("Total Cost")
ax.legend(fontsize=8)

# Box plot
ax = axes[0, 1]
bp = ax.boxplot([noreg_total_costs, reg_total_costs],
                labels=["Without reg", "With reg"],
                patch_artist=True,
                medianprops=dict(color="white", linewidth=2))
bp["boxes"][0].set_facecolor("#E84C4C"); bp["boxes"][0].set_alpha(0.7)
bp["boxes"][1].set_facecolor("#4C9BE8"); bp["boxes"][1].set_alpha(0.7)
ax.set_title(f"Cost Distribution ({EVAL_SEEDS} seeds)"); ax.set_ylabel("Total Cost")

# Safety score distributions
ax = axes[1, 0]
ax.hist(noreg_all_scores, bins=60, alpha=0.6, color="#E84C4C", density=True, label="Without reg")
ax.hist(reg_all_scores,   bins=60, alpha=0.6, color="#4C9BE8", density=True, label="With reg")
ax.axvline(safe_threshold_noreg, color="#E84C4C", linestyle="--", alpha=0.7, label=f"thr noreg={safe_threshold_noreg:.2f}")
ax.axvline(safe_threshold,       color="#4C9BE8", linestyle="--", alpha=0.7, label=f"thr reg={safe_threshold:.2f}")
ax.set_title("Safety Score Distribution (all seeds)"); ax.set_xlabel("Score"); ax.set_ylabel("Density")
ax.legend()

# Cumulative cost for GIF seed
ax = axes[1, 1]
if gif_noreg_costs is not None:
    ax.plot(np.cumsum(gif_noreg_costs), color="#E84C4C", label=f"Without reg (seed={GIF_SEED}, total={int(gif_noreg_costs.sum())})")
    ax.plot(np.cumsum(gif_reg_costs),   color="#4C9BE8", label=f"With reg    (seed={GIF_SEED}, total={int(gif_reg_costs.sum())})")
ax.set_title(f"Cumulative Cost (seed={GIF_SEED})"); ax.set_xlabel("Step"); ax.set_ylabel("Cost")
ax.legend()

plt.suptitle(
    f"Without Regularization vs With Regularization | {EVAL_SEEDS} seeds | "
    f"NoReg mean={noreg_total_costs.mean():.1f}  Reg mean={reg_total_costs.mean():.1f}  "
    f"({cost_red:+.1f}%)",
    fontsize=11, fontweight="bold",
)
plt.tight_layout()
fig.savefig(OUT_DIR / "comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {OUT_DIR}/comparison.png")
