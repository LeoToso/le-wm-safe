"""
Evaluate the MPPI planner with conformal safety constraint.

Runs N episodes in SafetyPointGoal1-v0 and reports:
  - Mean episode reward
  - Mean episode cost (lower = safer)
  - Cost rate per step
  - Comparison: random baseline vs MPPI-safe planner

Usage:
    MUJOCO_GL=osmesa python evaluate_planner.py
    MUJOCO_GL=osmesa CHECKPOINT=/mnt/t7shield/safe_lewm_rho1.pt python evaluate_planner.py
"""
import os, sys
import numpy as np
import torch
from collections import deque
from pathlib import Path

sys.path.insert(0, ".")
from safe_lewm.config import Config
from safe_lewm.model import SafeJEPA
from safe_lewm.env_utils import SafetyGymWrapper
from safe_lewm.planner import load_planner

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT   = os.environ.get("CHECKPOINT",  "/mnt/t7shield/safe_lewm_rho1.pt")
CLF_PATH     = os.environ.get("CLF_PATH",    "/mnt/t7shield/classifier.pt")
N_EPISODES   = int(os.environ.get("N_EPISODES", "20"))
HORIZON      = int(os.environ.get("HORIZON",    "10"))
N_SAMPLES    = int(os.environ.get("N_SAMPLES",  "512"))
TEMPERATURE  = float(os.environ.get("TEMPERATURE", "0.1"))
SAFETY_W     = float(os.environ.get("SAFETY_W",    "10.0"))
SAFETY_MARGIN= float(os.environ.get("SAFETY_MARGIN","0.2"))
HISTORY      = int(os.environ.get("HISTORY",    "3"))
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────────────────────────

print(f"Device: {DEVICE}")
print(f"Checkpoint: {CHECKPOINT}")
print(f"Classifier: {CLF_PATH}")
print(f"Episodes: {N_EPISODES} | Horizon: {HORIZON} | Samples: {N_SAMPLES}")


def run_random_episodes(env, n_episodes):
    """Baseline: random policy."""
    rewards, costs = [], []
    for ep in range(n_episodes):
        obs = env.reset()
        ep_reward, ep_cost = 0.0, 0.0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, r, c, done, _ = env.step(action)
            ep_reward += r
            ep_cost += c
        rewards.append(ep_reward)
        costs.append(ep_cost)
        print(f"  [Random] ep {ep+1}/{n_episodes}: reward={ep_reward:.2f}, cost={int(ep_cost)}")
    return np.array(rewards), np.array(costs)


def run_planner_episodes(env, model, planner, n_episodes):
    """MPPI planner with conformal safety."""
    rewards, costs = [], []

    for ep in range(n_episodes):
        obs = env.reset()        # (C, H, W) normalized
        planner.reset()

        # Build initial latent history by repeating the first observation
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            z0 = model.encoder(obs_t).squeeze(0)  # (D,)
        z_hist = z0.unsqueeze(0).expand(HISTORY, -1).clone()  # (HISTORY, D)

        # Pick a goal: use a fixed random latent (since we don't have a pixel goal here,
        # we use the encoder mean of the first obs as a proxy; replace with real goal if available)
        # For evaluation purposes use a zero vector — the planner will minimize distance to it,
        # which biases toward the center of the safe cluster.
        z_goal = torch.zeros(model.cfg.z_dim, device=DEVICE)

        ep_reward, ep_cost = 0.0, 0.0
        done = False
        step = 0

        while not done:
            # Plan
            action = planner.plan(z_hist, z_goal, n_iterations=1)
            action_np = action.cpu().numpy()

            # Step environment
            next_obs, r, c, done, _ = env.step(action_np)
            ep_reward += r
            ep_cost   += c
            step += 1

            # Update latent history
            with torch.no_grad():
                next_obs_t = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                z_next = model.encoder(next_obs_t).squeeze(0)  # (D,)
            z_hist = torch.cat([z_hist[1:], z_next.unsqueeze(0)], dim=0)  # slide window

        rewards.append(ep_reward)
        costs.append(ep_cost)
        print(f"  [MPPI]   ep {ep+1}/{n_episodes}: reward={ep_reward:.2f}, cost={int(ep_cost)}, steps={step}")

    return np.array(rewards), np.array(costs)


# ── Load model ─────────────────────────────────────────────────────────────────
print("\nLoading world model...")
cfg = Config()
model = SafeJEPA(cfg).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False))
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

# ── Load planner ───────────────────────────────────────────────────────────────
print("Loading planner...")
planner = load_planner(
    model=model,
    clf_path=CLF_PATH,
    device=DEVICE,
    horizon=HORIZON,
    n_samples=N_SAMPLES,
    temperature=TEMPERATURE,
    safety_margin=SAFETY_MARGIN,
    safety_weight=SAFETY_W,
)

# ── Environment ────────────────────────────────────────────────────────────────
env = SafetyGymWrapper(
    cfg.env_name,
    image_size=cfg.image_size,
    frame_stack=cfg.frame_stack,
    frame_skip=cfg.frame_skip,
    seed=42,
)

# ── Random baseline ────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Random baseline ({N_EPISODES} episodes)...")
rnd_rewards, rnd_costs = run_random_episodes(env, N_EPISODES)

# ── MPPI planner ───────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"MPPI + conformal safety ({N_EPISODES} episodes)...")
env2 = SafetyGymWrapper(
    cfg.env_name,
    image_size=cfg.image_size,
    frame_stack=cfg.frame_stack,
    frame_skip=cfg.frame_skip,
    seed=42,
)
plan_rewards, plan_costs = run_planner_episodes(env2, model, planner, N_EPISODES)

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"{'':30s} {'Random':>12s} {'MPPI-Safe':>12s}")
print(f"{'='*54}")
print(f"{'Mean episode reward':30s} {rnd_rewards.mean():>12.2f} {plan_rewards.mean():>12.2f}")
print(f"{'Mean episode cost':30s} {rnd_costs.mean():>12.2f} {plan_costs.mean():>12.2f}")
print(f"{'Cost std':30s} {rnd_costs.std():>12.2f} {plan_costs.std():>12.2f}")
print(f"{'Episodes with any cost':30s} {(rnd_costs>0).sum():>12d} {(plan_costs>0).sum():>12d}")
print(f"{'Cost rate (cost/total)':30s} {(rnd_costs>0).mean():>12.2%} {(plan_costs>0).mean():>12.2%}")
print(f"{'='*54}")
cost_reduction = (rnd_costs.mean() - plan_costs.mean()) / (rnd_costs.mean() + 1e-8) * 100
print(f"Cost reduction: {cost_reduction:.1f}%")
