"""
Evaluate LAMBDA policy at seed 5100 with hard-coded safety_gymnasium positions.
Bypasses replay buffer shape mismatch by only restoring model weights.
"""
import sys, os
os.environ["MUJOCO_GL"] = "egl"

# Must run from la-mbda/experiments/ for bare imports to work
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

# ── Inject argv so make_config parses correctly ──────────────────────────────
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
from train_utils import make_config, make_env, make_agent

config = make_config(sys.argv[1:])
print("Config loaded. observation_type:", config.observation_type)

# ── Build env with safety_gymnasium ──────────────────────────────────────────
# We need the fixedfar camera for overhead viz and hard-coded positions
import safety_gymnasium
from collections import deque
import cv2

# Hard-coded positions from safety_gymnasium seed=5100
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

class FixedLayoutEnv:
    """Wraps safety_gymnasium, overrides hazard/goal positions after reset."""
    def __init__(self, env_name, seed, image_size=64, frame_stack=4, action_repeat=2):
        self.env = safety_gymnasium.make(env_name, render_mode="rgb_array",
                                         camera_name="fixedfar")
        self.image_size = image_size
        self.frame_stack = frame_stack
        self.action_repeat = action_repeat
        self.seed = seed
        self.frames = deque(maxlen=frame_stack)

    def _override_positions(self):
        """Force hazard and goal positions to match hard-coded layout."""
        task = self.env.unwrapped.task
        # Override goal
        task.goal.pos[:2] = list(GOAL_XY)
        # Override hazards
        for i, (x, y) in enumerate(HAZARD_XY):
            if i < len(task.hazards.hazards_pos):
                task.hazards.hazards_pos[i][:2] = [x, y]
        # rebuild internal structures
        try:
            self.env.unwrapped._setup_layout()
        except Exception:
            pass

    def _get_frame(self):
        frame = self.env.render()
        if frame is None:
            frame = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        frame = cv2.resize(frame, (self.image_size, self.image_size))
        frame = frame.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        frame = (frame - mean) / std
        return frame.transpose(2, 0, 1)  # (3, H, W)

    def _stacked(self):
        return np.concatenate(list(self.frames), axis=0)  # (12, H, W)

    def reset(self):
        obs, info = self.env.reset(seed=self.seed)
        try:
            self._override_positions()
        except Exception as e:
            print(f"  [warn] position override failed: {e}")
        frame = self._get_frame()
        for _ in range(self.frame_stack):
            self.frames.append(frame)
        return self._stacked(), info

    def step(self, action):
        total_reward = total_cost = 0.0
        for _ in range(self.action_repeat):
            result = self.env.step(action)
            if len(result) == 6:
                obs, r, cost, terminated, truncated, info = result
            else:
                obs, r, terminated, truncated, info = result
                cost = info.get("cost", 0.0)
            total_reward += r
            total_cost += cost
            if terminated or truncated:
                break
        self.frames.append(self._get_frame())
        done = terminated or truncated
        return self._stacked(), total_reward, float(total_cost > 0), done, info

    def render_rgb(self):
        return self.env.render()

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space


env = FixedLayoutEnv("sgymn_SafetyPointGoal1-v0", seed=SEED,
                     image_size=64, frame_stack=4, action_repeat=2)
print("Env created.")

# ── Build agent ───────────────────────────────────────────────────────────────
# Use make_agent but then selectively restore only model weights
agent = make_agent(config)

# Find checkpoint
ckpt_dir = os.path.join(LOG_DIR, "agent_data")
latest = tf.train.latest_checkpoint(ckpt_dir)
print(f"Latest checkpoint: {latest}")

# Restore only model weights (skip replay buffer _experience)
checkpoint = tf.train.Checkpoint(
    actor=agent.actor,
    model=agent.model,
    critic=agent.critic,
    safety_critic=agent.safety_critic,
)
status = checkpoint.restore(latest).expect_partial()
print("Checkpoint restored (model weights only, buffer skipped).")

# ── Run one episode ───────────────────────────────────────────────────────────
obs_stack, info = env.reset()
frames_rgb = [env.render_rgb()]

total_reward = 0.0
total_cost = 0.0
done = False
step = 0
max_steps = 500

# LAMBDA expects (batch, frame_stack, C, H, W) or similar — check agent.act signature
# Typical LAMBDA: agent.act(obs) where obs is (frame_stack*C, H, W)
print("Running rollout...")
while not done and step < max_steps:
    # obs_stack shape: (frame_stack*3, H, W) = (12, 64, 64)
    # LAMBDA actor usually expects (1, frame_stack, C, H, W) or (1, C*frame_stack, H, W)
    obs_input = obs_stack[np.newaxis]  # (1, 12, 64, 64)
    try:
        action = agent.act(obs_input, training=False)
    except Exception:
        # Try reshaping to (1, frame_stack, 3, H, W)
        obs_reshaped = obs_stack.reshape(4, 3, 64, 64)[np.newaxis]
        action = agent.act(obs_reshaped, training=False)

    if hasattr(action, "numpy"):
        action = action.numpy()
    action = action.squeeze()

    obs_stack, reward, cost, done, info = env.step(action)
    total_reward += reward
    total_cost += cost
    frames_rgb.append(env.render_rgb())
    step += 1
    if step % 50 == 0:
        print(f"  step {step}: reward={total_reward:.2f}, cost={total_cost:.2f}")

print(f"Episode done: steps={step}, reward={total_reward:.2f}, cost={total_cost:.2f}")

# ── Save GIF ──────────────────────────────────────────────────────────────────
valid_frames = [f for f in frames_rgb if f is not None]
if valid_frames:
    gif_frames = [Image.fromarray(f.astype(np.uint8)) for f in valid_frames]
    gif_frames[0].save(
        SAVE_GIF,
        save_all=True,
        append_images=gif_frames[1:],
        duration=50,
        loop=0,
    )
    print(f"GIF saved to {SAVE_GIF}")
else:
    print("No frames to save!")
