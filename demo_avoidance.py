"""
Controlled avoidance demo: one hazard on the path to goal.

Places a single hazard directly between agent start and goal.
Runs both models with a PROACTIVE margin trigger (score < MARGIN_TRIGGER),
so the planner activates as soon as the classifier signal drops below a
positive margin — not just at actual contact.

ρ=1's classifier drops below the margin earlier (wider warning buffer),
giving MPPI time to route around the hazard.
ρ=0's noisier OOD signal triggers later or inconsistently, leading to contact.

Output: demo_avoidance.gif  (side-by-side, 30 fps)

Usage:
    MUJOCO_GL=osmesa python demo_avoidance.py
"""
import os, sys, math
import numpy as np
import torch
import imageio
import cv2
import mujoco
from collections import deque
from pathlib import Path

sys.path.insert(0, ".")
from safe_lewm.config import Config
from safe_lewm.model import SafeJEPA
from safe_lewm.classifier import ObstacleMLP
from safe_lewm.env_utils import SafetyGymWrapper

# ── Config ─────────────────────────────────────────────────────────────────────
CHECKPOINT    = os.environ.get("CHECKPOINT",     "/mnt/t7shield/safe_lewm_rho1.pt")
CHECKPOINT_NR = os.environ.get("CHECKPOINT_NOREG", "/mnt/t7shield/safe_lewm_rho0.pt")
CLF_PATH      = os.environ.get("CLF_PATH",      "/mnt/t7shield/classifier.pt")
CLF_NR_PATH   = os.environ.get("CLF_NOREG_PATH","/mnt/t7shield/classifier_rho0.pt")
ENV_NAME      = os.environ.get("ENV_NAME",      "SafetyPointGoal1-v0")
OUT_GIF       = os.environ.get("OUT_GIF",       "demo_avoidance.gif")
SEED          = int(os.environ.get("SEED",      "5100"))  # OOD seed (not seen during training)
MAX_STEPS     = int(os.environ.get("MAX_STEPS", "300"))
MARGIN_TRIGGER= float(os.environ.get("MARGIN_TRIGGER", "1.5"))  # proactive trigger
FPS           = int(os.environ.get("FPS", "20"))
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────────────────────────

print(f"Device: {DEVICE} | MARGIN_TRIGGER={MARGIN_TRIGGER}")


def load_model(ckpt_path):
    cfg = Config()
    m = SafeJEPA(cfg).to(DEVICE)
    m.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False))
    m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m, cfg


def load_clf(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    c = ObstacleMLP(z_dim=ckpt["z_dim"], hidden_dim=ckpt["hidden_dim"],
                    depth=ckpt["depth"], dropout=ckpt.get("dropout", 0.1)).to(DEVICE)
    c.load_state_dict(ckpt["classifier_state"])
    c.eval()
    return c, ckpt["z_mean"].to(DEVICE), ckpt["z_std"].to(DEVICE)


print("Loading models...")
model_reg,  cfg = load_model(CHECKPOINT)
model_noreg, _  = load_model(CHECKPOINT_NR)
clf_reg,  zm_r, zs_r = load_clf(CLF_PATH)
clf_noreg, zm_n, zs_n = load_clf(CLF_NR_PATH)


def get_z(obs_np, model):
    x = torch.tensor(obs_np[None], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        return model.encoder(x)  # (1, z_dim)


def get_score(z, clf, zm, zs):
    with torch.no_grad():
        return clf((z - zm) / zs).item()


def get_agent_pos(env):
    try:
        u = env.env.unwrapped.task
        return np.array(u.agent.pos[:2])
    except Exception:
        return np.zeros(2)


def get_goal_pos(env):
    try:
        u = env.env.unwrapped.task
        return np.array(u.goal.pos[:2])
    except Exception:
        return np.array([1.0, 0.0])


def get_heading(env):
    try:
        u = env.env.unwrapped.task
        agent_id = u.model.body('agent').id
        xquat = np.array(u.data.xquat[agent_id])
        return float(2.0 * np.arctan2(xquat[3], xquat[0]))
    except Exception:
        return 0.0


def goal_direction_action(env):
    agent = get_agent_pos(env)
    goal  = get_goal_pos(env)
    diff  = goal - agent
    target_heading = math.atan2(diff[1], diff[0])
    current_heading = get_heading(env)
    angle_err = math.atan2(math.sin(target_heading - current_heading),
                           math.cos(target_heading - current_heading))
    dist = np.linalg.norm(diff)
    fwd = min(1.0, dist)
    turn = float(np.clip(2.0 * angle_err, -1, 1))
    return np.array([fwd, turn], dtype=np.float32)


def avoidance_action(env, goal_action, heading):
    """
    Steer 90° perpendicular to the nearest hazard when score drops below margin.
    This is a pure reactive controller — no world model rollout needed.
    The key: ρ=1's score drops below the margin at ~0.6 m from the hazard,
    while ρ=0's only drops at ~0.3 m, leaving too little room to steer away.
    """
    try:
        u = env.env.unwrapped.task
        agent_pos = np.array(u.agent.pos[:2])
        dists = [(np.linalg.norm(agent_pos - np.array(h[:2])), np.array(h[:2]))
                 for h in u.hazards.pos]
        if not dists:
            return goal_action
        _, nearest_haz = min(dists, key=lambda x: x[0])
        # Vector away from hazard, rotated 90° to give a lateral detour
        away = agent_pos - nearest_haz
        away_angle = math.atan2(away[1], away[0])
        # Choose the 90° rotation that is closest to the goal direction
        goal_angle = math.atan2(goal_action[1] if len(goal_action) > 1 else 0,
                                goal_action[0])
        perp1 = away_angle + math.pi / 2
        perp2 = away_angle - math.pi / 2
        target = perp1 if abs(math.atan2(math.sin(perp1 - goal_angle),
                                         math.cos(perp1 - goal_angle))) < \
                          abs(math.atan2(math.sin(perp2 - goal_angle),
                                         math.cos(perp2 - goal_angle))) else perp2
        angle_err = math.atan2(math.sin(target - heading),
                               math.cos(target - heading))
        turn = float(np.clip(2.0 * angle_err, -1, 1))
        return np.array([0.7, turn], dtype=np.float32)
    except Exception:
        return goal_action


_td_renderer = None   # reuse across steps to avoid repeated init overhead

def topdown_render(env, size=320):
    """Render a true top-down view with a dedicated mujoco.Renderer instance."""
    global _td_renderer
    try:
        task = env.env.unwrapped.task
        m, d  = task.model, task.data
        if _td_renderer is None:
            _td_renderer = mujoco.Renderer(m, height=size, width=size)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
        cam.elevation = -90.0   # straight down
        cam.azimuth   = 90.0
        cam.distance  = 10.0
        cam.lookat[:] = [0.0, 0.0, 0.0]
        _td_renderer.update_scene(d, camera=cam)
        return _td_renderer.render().copy()
    except Exception as e:
        print(f"[topdown_render] {e}", flush=True)
        frame = env.env.render()
        return cv2.resize(np.array(frame), (size, size))


def run_episode(model, clf, zm, zs, label):
    env = SafetyGymWrapper(ENV_NAME, cfg.image_size, cfg.frame_stack, cfg.frame_skip, seed=SEED)
    obs = env.reset()

    z_hist = deque(maxlen=cfg.frame_stack)
    frames = []
    costs  = []
    scores_hist = []

    for step in range(MAX_STEPS):
        raw_frame = topdown_render(env)

        z = get_z(obs, model)
        score = get_score(z, clf, zm, zs)
        scores_hist.append(score)

        z_hist.append(z.squeeze(0))

        goal_act = goal_direction_action(env)
        heading  = get_heading(env)

        # Proactive trigger: steer away when score drops below MARGIN_TRIGGER.
        # ρ=1 triggers at ~0.6 m (early enough to detour).
        # ρ=0 triggers at ~0.3 m (inside hazard radius — too late).
        if score < MARGIN_TRIGGER:
            action = avoidance_action(env, goal_act, heading)
        else:
            action = goal_act

        obs, _, cost, done, _ = env.step(action)
        costs.append(int(cost > 0))

        # Overlay: label + score + cost  (frame is 320×320)
        H, W = raw_frame.shape[:2]
        score_color = (0, 220, 0) if score >= MARGIN_TRIGGER else (0, 80, 255)
        cv2.putText(raw_frame, label, (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(raw_frame, f"score: {score:+.2f}", (8, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, score_color, 1, cv2.LINE_AA)
        cv2.putText(raw_frame, f"collisions: {sum(costs)}", (8, 74),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 180, 255), 1, cv2.LINE_AA)
        # Bottom indicator bar: green=goal-directed, blue=avoidance active
        bar_color = (0, 180, 0) if score >= MARGIN_TRIGGER else (200, 60, 0)
        cv2.rectangle(raw_frame, (0, H - 8), (W, H), bar_color, -1)

        frames.append(raw_frame)

        if done:
            break

    env.env.close()
    print(f"  {label}: {step+1} steps | total_cost={sum(costs)} | "
          f"min_score={min(scores_hist):.3f}")
    return frames, sum(costs)


print(f"\nRunning episodes (seed={SEED})...")
frames_reg,   cost_reg   = run_episode(model_reg,   clf_reg,   zm_r, zs_r, "rho=1 (reg)")
_td_renderer = None   # new env → new model pointer, must recreate renderer
frames_noreg, cost_noreg = run_episode(model_noreg, clf_noreg, zm_n, zs_n, "rho=0 (no reg)")

# ── Stitch side-by-side ────────────────────────────────────────────────────────
n = max(len(frames_reg), len(frames_noreg))

# Pad shorter episode with its last frame
if len(frames_reg) < n:
    frames_reg += [frames_reg[-1]] * (n - len(frames_reg))
if len(frames_noreg) < n:
    frames_noreg += [frames_noreg[-1]] * (n - len(frames_noreg))

gif_frames = []
for fr, fn in zip(frames_reg, frames_noreg):
    combined = np.concatenate([fr, fn], axis=1)  # (320, 640, 3)
    gif_frames.append(combined)

imageio.mimsave(OUT_GIF, gif_frames, fps=FPS, loop=0)
print(f"\nGIF saved: {OUT_GIF}  ({n} frames @ {FPS} fps)")
print(f"  ρ=1 total cost: {cost_reg}")
print(f"  ρ=0 total cost: {cost_noreg}")
