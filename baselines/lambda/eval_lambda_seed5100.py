"""
Evaluate LAMBDA policy at seed 5100 with hard-coded safety_gymnasium positions.
Bypasses replay buffer shape mismatch by only restoring model weights.
"""
import sys, os
os.environ["MUJOCO_GL"] = "egl"

LAMBDA_DIR = "/home/dz2478/la-mbda"
EXPERIMENTS_DIR = os.path.join(LAMBDA_DIR, "experiments")
sys.path.insert(0, LAMBDA_DIR)
sys.path.insert(0, EXPERIMENTS_DIR)

import numpy as np
import tensorflow as tf
import imageio
from PIL import Image

LOG_DIR = "/home/dz2478/la-mbda/results/lambda_pointgoal1/seed0"
SEED = 5100
SAVE_GIF = "/home/dz2478/le-wm-safe/baselines/lambda/rollout_seed5100.gif"
os.makedirs(os.path.dirname(SAVE_GIF), exist_ok=True)

# ── Inject argv so make_config parses correctly ───────────────────────────────
sys.argv = [
    "eval",
    "--safety",
    "--seed", str(SEED),
    "--environment", "sgymn_SafetyPointGoal1-v0",
    "--observation_type", "rgb_image",
    "--action_repeat", "2",
    "--episode_length", "1000",
    "--total_training_steps", "100000",
    "--log_dir", LOG_DIR,
]

os.chdir(EXPERIMENTS_DIR)
import train_utils
from la_mbda.la_mbda import LAMBDA
import la_mbda.env_wrappers as env_wrappers

config = train_utils.make_config(train_utils.define_config())
print("Config loaded. observation_type:", config.observation_type)

# ── Build env just to get observation/action spaces ───────────────────────────
ref_env = env_wrappers.make_env(
    config.environment, config.episode_length, config.action_repeat, config.seed,
    config.observation_type
)
obs_space = ref_env.observation_space
act_space = ref_env.action_space
ref_env.close()
print(f"obs_space: {obs_space}, act_space: {act_space}")

# ── Build LAMBDA agent ────────────────────────────────────────────────────────
import logging
logger = logging.getLogger("lambda_eval")
agent = LAMBDA(config, logger, obs_space, act_space)

# Restore only model weights, skip replay buffer
ckpt_dir = os.path.join(LOG_DIR, "agent_data")
latest = tf.train.latest_checkpoint(ckpt_dir)
print(f"Latest checkpoint: {latest}")

checkpoint = tf.train.Checkpoint(
    actor=agent.actor,
    model=agent.model,
    critic=agent.critic,
    safety_critic=agent.safety_critic,
)
checkpoint.restore(latest).expect_partial()
print("Checkpoint restored (model weights only, buffer skipped).")

# ── Build rollout env with fixedfar camera and fixed positions ────────────────
import safety_gymnasium
import cv2
from collections import deque

HAZARD_XY = [
    (-0.21911489,  1.22436124),
    (-0.60313723,  0.29073607),
    ( 0.19738684, -0.75398054),
    (-0.68791828,  0.86701657),
    ( 0.15841011,  0.16933630),
    ( 1.13426888,  0.68566826),
    ( 0.84437717,  1.09028997),
    (-0.19297098,  0.28184397),
]
GOAL_XY = (0.33311007, 1.04349929)

IMAGE_SIZE = 64
FRAME_STACK = 4
ACTION_REPEAT = config.action_repeat

raw_env = safety_gymnasium.make(
    "SafetyPointGoal1-v0", render_mode="rgb_array", camera_name="fixedfar"
)

def override_positions(env):
    task = env.unwrapped.task
    try:
        task.goal.pos[:2] = list(GOAL_XY)
    except Exception as e:
        print(f"  [warn] goal override: {e}")
    try:
        for i, (x, y) in enumerate(HAZARD_XY):
            task.hazards.hazards_pos[i][:2] = [x, y]
    except Exception as e:
        print(f"  [warn] hazard override: {e}")

def get_frame(env):
    frame = env.render()
    if frame is None:
        return np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
    frame = frame.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    frame = (frame - mean) / std
    return frame.transpose(2, 0, 1)

# Reset and override
obs_gym, info = raw_env.reset(seed=SEED)
override_positions(raw_env)

frames_deque = deque(maxlen=FRAME_STACK)
frame0 = get_frame(raw_env)
for _ in range(FRAME_STACK):
    frames_deque.append(frame0)

viz_frames = [raw_env.render()]

def stacked():
    return np.concatenate(list(frames_deque), axis=0)  # (12, H, W)

# ── Run rollout ───────────────────────────────────────────────────────────────
total_reward = total_cost = 0.0
done = False
step = 0
max_steps = 500

print("Running rollout...")
# Warm up agent state with blank observations
obs_np = stacked()[np.newaxis].astype(np.float32)  # (1, 12, H, W)

while not done and step < max_steps:
    obs_tf = tf.constant(obs_np)
    try:
        action = agent.policy(obs_tf, training=False)
    except Exception:
        # Try agent.act or agent.__call__
        try:
            action = agent(obs_tf, training=False)
        except Exception as e2:
            print(f"policy call failed: {e2}")
            action = raw_env.action_space.sample()

    if hasattr(action, "numpy"):
        action = action.numpy()
    action = np.array(action).squeeze()

    # Step with action_repeat
    r_sum = c_sum = 0.0
    for _ in range(ACTION_REPEAT):
        result = raw_env.step(action)
        if len(result) == 6:
            _, r, cost, terminated, truncated, info = result
        else:
            _, r, terminated, truncated, info = result
            cost = info.get("cost", 0.0)
        r_sum += r
        c_sum += cost
        if terminated or truncated:
            done = True
            break

    frames_deque.append(get_frame(raw_env))
    obs_np = stacked()[np.newaxis].astype(np.float32)
    total_reward += r_sum
    total_cost += c_sum
    viz_frames.append(raw_env.render())
    step += 1

    if step % 50 == 0:
        print(f"  step {step}: reward={total_reward:.2f}, cost={total_cost:.2f}")

print(f"Done: steps={step}, reward={total_reward:.2f}, cost={total_cost:.2f}")

# ── Save GIF ──────────────────────────────────────────────────────────────────
valid = [f for f in viz_frames if f is not None]
if valid:
    pil_frames = [Image.fromarray(f.astype(np.uint8)) for f in valid]
    pil_frames[0].save(
        SAVE_GIF, save_all=True, append_images=pil_frames[1:], duration=50, loop=0
    )
    print(f"GIF saved: {SAVE_GIF}")
else:
    print("No frames captured.")
