import safety_gymnasium
import numpy as np
from collections import deque


class SafetyGymWrapper:
    """Wraps Safety Gym env to return stacked 64x64 RGB pixel observations."""

    def __init__(self, env_name, image_size=64, frame_stack=4, frame_skip=2, seed=0):
        # Use safety_gymnasium.make directly to avoid gymnasium wrapper conflicts
        # (gymnasium wrappers expect 5-tuple but safety-gymnasium returns 6-tuple)
        self.env = safety_gymnasium.make(env_name, render_mode="rgb_array")
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


def collect_dataset(cfg, seed=42, save_every=100):
    """Collect offline dataset. Saves one file per checkpoint to avoid corruption on crash."""
    import pickle, sys, glob
    from pathlib import Path

    data_dir = Path(cfg.data_path).parent
    data_dir.mkdir(parents=True, exist_ok=True)
    env = SafetyGymWrapper(cfg.env_name, cfg.image_size, cfg.frame_stack, cfg.frame_skip, seed)

    # Find highest valid checkpoint already saved
    ckpt_files = sorted(glob.glob(str(data_dir / "chunk_*.pkl")))
    trajectories = []
    for ckpt in ckpt_files:
        try:
            with open(ckpt, "rb") as f:
                chunk = pickle.load(f)
            trajectories.extend(chunk)
        except Exception:
            print(f"  Warning: could not load {ckpt}, skipping", flush=True)
    start_ep = len(trajectories)
    if start_ep > 0:
        print(f"Resuming from episode {start_ep}/{cfg.num_episodes}", flush=True)

    print(f"Collecting {cfg.num_episodes - start_ep} episodes...", flush=True)

    chunk = []
    for ep in range(start_ep, cfg.num_episodes):
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
        chunk.append(traj)
        trajectories.append(traj)

        if (ep + 1) % save_every == 0:
            # Save this chunk to its own file — never overwrites previous chunks
            chunk_path = data_dir / f"chunk_{ep+1:05d}.pkl"
            with open(chunk_path, "wb") as f:
                pickle.dump(chunk, f)
            chunk = []
            print(f"  Episode {ep+1}/{cfg.num_episodes} — saved {chunk_path.name}", flush=True)
            sys.stdout.flush()

    # Save any remaining episodes
    if chunk:
        chunk_path = data_dir / f"chunk_{cfg.num_episodes:05d}_final.pkl"
        with open(chunk_path, "wb") as f:
            pickle.dump(chunk, f)

    # Merge all chunks into the main data file
    with open(cfg.data_path, "wb") as f:
        pickle.dump(trajectories, f)
    print(f"Done. Saved {len(trajectories)} trajectories to {cfg.data_path}", flush=True)
    return trajectories
