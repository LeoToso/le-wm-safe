import gymnasium as gym
import safety_gymnasium
import numpy as np
from collections import deque


class SafetyGymWrapper:
    """Wraps Safety Gym env to return stacked 64x64 RGB pixel observations."""

    def __init__(self, env_name, image_size=64, frame_stack=4, frame_skip=2, seed=0):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.image_size = image_size
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self.frames = deque(maxlen=frame_stack)
        self.seed = seed

    def _get_frame(self):
        """Render and resize frame to (3, H, W) normalized."""
        import cv2
        frame = self.env.render()  # (H, W, 3) uint8
        frame = cv2.resize(frame, (self.image_size, self.image_size))
        frame = frame.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        frame = (frame - mean) / std
        return frame.transpose(2, 0, 1)  # (3, H, W)

    def _get_stacked_obs(self):
        """Returns (frame_stack*3, H, W) stacked frames."""
        return np.concatenate(list(self.frames), axis=0)

    def reset(self):
        self.env.reset(seed=self.seed)
        frame = self._get_frame()
        for _ in range(self.frame_stack):
            self.frames.append(frame)
        return self._get_stacked_obs()

    def step(self, action):
        total_reward = 0.0
        total_cost = 0.0
        for _ in range(self.frame_skip):
            result = self.env.step(action)
            # safety-gymnasium returns (obs, reward, cost, terminated, truncated, info)
            # standard gymnasium returns (obs, reward, terminated, truncated, info)
            if len(result) == 6:
                obs, reward, cost, terminated, truncated, info = result
            else:
                obs, reward, terminated, truncated, info = result
                cost = info.get("cost", 0.0)
            total_reward += reward
            total_cost += cost
            if terminated or truncated:
                break
        frame = self._get_frame()
        self.frames.append(frame)
        done = terminated or truncated
        return self._get_stacked_obs(), total_reward, float(total_cost > 0), done, info

    @property
    def action_space(self):
        return self.env.action_space


def collect_dataset(cfg, seed=42):
    """Collect offline dataset using random policy with heuristic goal bias."""
    import pickle
    from pathlib import Path

    env = SafetyGymWrapper(cfg.env_name, cfg.image_size, cfg.frame_stack, cfg.frame_skip, seed)
    trajectories = []

    print(f"Collecting {cfg.num_episodes} episodes...")
    for ep in range(cfg.num_episodes):
        obs = env.reset()
        traj = {"obs": [], "action": [], "reward": [], "cost": [], "next_obs": []}

        for step in range(cfg.episode_length):
            action = env.action_space.sample()
            next_obs, reward, cost, done, info = env.step(action)

            traj["obs"].append(obs.copy())
            traj["action"].append(action.copy())
            traj["reward"].append(reward)
            traj["cost"].append(cost)
            traj["next_obs"].append(next_obs.copy())

            obs = next_obs
            if done:
                break

        for k in traj:
            traj[k] = np.array(traj[k])
        trajectories.append(traj)

        if (ep + 1) % 100 == 0:
            print(f"  Episode {ep+1}/{cfg.num_episodes}")

    Path(cfg.data_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.data_path, "wb") as f:
        pickle.dump(trajectories, f)
    print(f"Saved dataset to {cfg.data_path}")
    return trajectories
