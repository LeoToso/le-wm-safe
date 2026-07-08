import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import glob
from pathlib import Path


class TrajectoryDataset(Dataset):
    """Dataset of trajectory sub-sequences of length seq_len."""

    def __init__(self, trajectories, seq_len=5):
        self.seq_len = seq_len
        self.samples = []

        for traj in trajectories:
            T = len(traj["obs"])
            for start in range(T - seq_len):
                self.samples.append({
                    k: traj[k][start:start + seq_len] for k in traj
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "obs": torch.tensor(s["obs"], dtype=torch.float32),           # (T, 12, 64, 64)
            "action": torch.tensor(s["action"], dtype=torch.float32),     # (T, A)
            "reward": torch.tensor(s["reward"], dtype=torch.float32),     # (T,)
            "cost": torch.tensor(s["cost"], dtype=torch.float32),         # (T,)
            "next_obs": torch.tensor(s["next_obs"], dtype=torch.float32), # (T, 12, 64, 64)
        }


def load_trajectories(data_path):
    """Load trajectories from a single pkl file or a directory of chunk_*.pkl files."""
    data_path = Path(data_path)

    # If the main file exists and is readable, use it
    if data_path.exists() and data_path.stat().st_size > 0:
        try:
            with open(data_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    # Fall back to loading chunk files from the same directory
    chunk_files = sorted(glob.glob(str(data_path.parent / "chunk_*.pkl")))
    if not chunk_files:
        raise FileNotFoundError(f"No data found at {data_path} or as chunk files in {data_path.parent}")

    print(f"Loading {len(chunk_files)} chunk files...")
    trajectories = []
    for path in chunk_files:
        with open(path, "rb") as f:
            trajectories.extend(pickle.load(f))
        print(f"  Loaded {Path(path).name} — {len(trajectories)} trajectories so far", flush=True)
    return trajectories


def load_dataset(data_path, seq_len, batch_size, num_workers=4):
    trajectories = load_trajectories(data_path)
    print(f"Building dataset from {len(trajectories)} trajectories...")
    dataset = TrajectoryDataset(trajectories, seq_len)
    print(f"Dataset: {len(dataset)} samples")
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    return loader
