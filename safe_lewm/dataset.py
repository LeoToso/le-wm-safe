import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle


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


def load_dataset(data_path, seq_len, batch_size, num_workers=4):
    with open(data_path, "rb") as f:
        trajectories = pickle.load(f)
    dataset = TrajectoryDataset(trajectories, seq_len)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    return loader
