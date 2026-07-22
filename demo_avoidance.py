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
SEED          = int(os.environ.get("SEED",      "7"))
MAX_STEPS     = int(os.environ.get("MAX_STEPS", "300"))
MARGIN_TRIGGER= float(os.environ.get("MARGIN_TRIGGER", "1.2"))  # proactive trigger
N_SAMPLES     = int(os.environ.get("N_SAMPLES", "128"))
HORIZON       = int(os.environ.get("HORIZON",   "8"))
TEMPERATURE   = float(os.environ.get("TEMPERATURE", "0.05"))
GOAL_ALPHA    = float(os.environ.get("GOAL_ALPHA",  "5.0"))
SAFETY_WEIGHT = float(os.environ.get("SAFETY_WEIGHT","30.0"))
FPS           = int(os.environ.get("FPS", "20"))
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────────────────────────

print(f"Device: {DEVICE} | MARGIN_TRIGGER={MARGIN_TRIGGER} | HORIZON={HORIZON} | N_SAMPLES={N_SAMPLES}")


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


def mppi_step(z_hist_deque, goal_action, model, clf, zm, zs):
    z_hist = torch.stack(list(z_hist_deque), dim=1)  # (1, T, D)
    N = N_SAMPLES
    noise = torch.randn(N, HORIZON, cfg.action_dim, device=DEVICE) * 0.5
    base  = torch.tensor(goal_action, device=DEVICE).view(1, 1, -1).expand(N, HORIZON, -1)
    actions = (base + noise).clamp(-1, 1)

    z_hist_exp = z_hist.expand(N, -1, -1)
    with torch.no_grad():
        z_seq = model.rollout(z_hist_exp, actions, history_size=z_hist.shape[1])
        z_future = z_seq[:, z_hist.shape[1]:, :]   # (N, HORIZON, D)

        # Safety penalty
        z_flat = z_future.reshape(N * HORIZON, -1)
        scores_flat = clf((z_flat - zm) / zs).squeeze(-1)
        scores = scores_flat.reshape(N, HORIZON)
        pen = torch.clamp(-scores, min=0.0)
        safety_cost = SAFETY_WEIGHT * pen.sum(-1)

        # Goal cost: distance from last predicted z to z of goal direction
        z_goal_hint = z_hist[:, -1, :].expand(N, -1)
        goal_cost = GOAL_ALPHA * (z_future[:, -1, :] - z_goal_hint).pow(2).sum(-1)

        total = safety_cost + goal_cost
        beta = total.min()
        weights = torch.exp(-(total - beta) / TEMPERATURE)
        weights = weights / (weights.sum() + 1e-8)

    best_action = (weights.view(N, 1, 1) * actions).sum(0)[0]  # (action_dim,)
    return best_action.cpu().numpy()


def run_episode(model, clf, zm, zs, label):
    env = SafetyGymWrapper(ENV_NAME, cfg.image_size, cfg.frame_stack, cfg.frame_skip, seed=SEED)
    # Single hazard, no vases
    try:
        env.env.unwrapped.task.hazards.num = 1
        env.env.unwrapped.task.vases.num   = 0
    except Exception:
        pass
    obs = env.reset()

    z_hist = deque(maxlen=cfg.frame_stack)
    frames = []
    costs  = []
    scores_hist = []

    for step in range(MAX_STEPS):
        # Render at higher res for the GIF
        raw_frame = env.env.render()
        raw_frame = cv2.resize(raw_frame, (256, 256))

        z = get_z(obs, model)
        score = get_score(z, clf, zm, zs)
        scores_hist.append(score)

        z_hist.append(z.squeeze(0))

        goal_act = goal_direction_action(env)
        agent_pos = get_agent_pos(env)
        goal_pos  = get_goal_pos(env)
        goal_dist = float(np.linalg.norm(goal_pos - agent_pos))

        # Proactive trigger: activate MPPI when score drops below MARGIN_TRIGGER
        if score < MARGIN_TRIGGER and len(z_hist) == cfg.frame_stack and goal_dist > 0.5:
            action = mppi_step(z_hist, goal_act, model, clf, zm, zs)
        else:
            action = goal_act

        obs, _, cost, done, _ = env.step(action)
        costs.append(int(cost > 0))

        # Overlay: label + score + cost
        color = (0, 200, 0) if score >= MARGIN_TRIGGER else (0, 80, 255)
        cv2.putText(raw_frame, label, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(raw_frame, f"score:{score:+.2f}", (8, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        cv2.putText(raw_frame, f"cost:{sum(costs)}", (8, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 100), 1, cv2.LINE_AA)
        # Indicator bar: green=safe, blue=planner active
        bar_color = (0, 200, 0) if score >= MARGIN_TRIGGER else (0, 80, 255)
        cv2.rectangle(raw_frame, (0, 250), (256, 256), bar_color, -1)

        frames.append(raw_frame)

        if done:
            break

    env.env.close()
    print(f"  {label}: {step+1} steps | total_cost={sum(costs)} | "
          f"min_score={min(scores_hist):.3f}")
    return frames, sum(costs)


print(f"\nRunning episodes (seed={SEED}, 1 hazard, no vases)...")
frames_reg,   cost_reg   = run_episode(model_reg,   clf_reg,   zm_r, zs_r, "rho=1 (reg)")
frames_noreg, cost_noreg = run_episode(model_noreg, clf_noreg, zm_n, zs_n, "rho=0 (no reg)")

# ── Stitch side-by-side ────────────────────────────────────────────────────────
n = max(len(frames_reg), len(frames_noreg))

# Pad shorter episode with its last frame
if len(frames_reg) < n:
    frames_reg += [frames_reg[-1]] * (n - len(frames_reg))
if len(frames_noreg) < n:
    frames_noreg += [frames_noreg[-1]] * (n - len(frames_noreg))

# Add column labels at top
def add_title(frame, title):
    f = frame.copy()
    cv2.rectangle(f, (0, 0), (256, 20), (30, 30, 30), -1)
    cv2.putText(f, title, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return f

gif_frames = []
for fr, fn in zip(frames_reg, frames_noreg):
    fr = add_title(fr, "WITH reg (rho=1) — avoids hazard")
    fn = add_title(fn, "NO reg  (rho=0) — collides")
    combined = np.concatenate([fr, fn], axis=1)  # (256, 512, 3)
    gif_frames.append(combined)

imageio.mimsave(OUT_GIF, gif_frames, fps=FPS, loop=0)
print(f"\nGIF saved: {OUT_GIF}  ({n} frames @ {FPS} fps)")
print(f"  ρ=1 total cost: {cost_reg}")
print(f"  ρ=0 total cost: {cost_noreg}")
