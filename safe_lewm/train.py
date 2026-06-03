import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

from .config import Config
from .model import SafeJEPA
from .losses import pred_loss, sigreg_loss, bisimulation_loss, cost_bce_loss
from .dataset import load_dataset


def train(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    model = SafeJEPA(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    loader = load_dataset(cfg.data_path, cfg.seq_len, cfg.batch_size)
    loader_iter = iter(loader)

    Path(cfg.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    losses_log = {"total": [], "pred": [], "sig": [], "bsim": [], "cost": []}

    for step in tqdm(range(cfg.train_steps), desc="Training"):
        # Get batch
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        obs = batch["obs"].to(device)           # (B, T, 12, 64, 64)
        next_obs = batch["next_obs"].to(device)  # (B, T, 12, 64, 64)
        actions = batch["action"].to(device)     # (B, T, A)
        rewards = batch["reward"].to(device)     # (B, T)
        costs = batch["cost"].to(device)         # (B, T)

        B, T = obs.shape[:2]

        # 1. Encode all observations: (B, T, D)
        z = model.encode(obs)
        z_next = model.encode(next_obs)

        # 2. Prediction loss: predict z_{t+1} given history up to t
        pred_z = model.predict(z[:, :-1], actions[:, :-1])  # (B, T-1, D)
        l_pred = pred_loss(pred_z, z[:, 1:])

        # 3. SIGReg on z
        l_sig = sigreg_loss(z, model.sigreg)

        # 4. Bisimulation loss — use first timestep
        z0 = z[:, 0]        # (B, D)
        a0 = actions[:, 0]  # (B, A)
        r0 = rewards[:, 0]  # (B,)
        c0 = costs[:, 0]    # (B,)
        l_bsim = bisimulation_loss(z0, r0, c0, model.transition_model, a0, cfg.rho, cfg.gamma)

        # 5. Cost head loss
        z_flat = z.reshape(B * T, -1)
        cost_pred = model.cost_head(z_flat)  # (B*T, 1)
        cost_true = costs.reshape(B * T)     # (B*T,)
        l_cost = cost_bce_loss(cost_pred, cost_true)

        # 6. Total loss
        loss = l_pred + cfg.lambda_sig * l_sig + cfg.lambda_bsim * l_bsim + cfg.lambda_cost * l_cost

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        if (step + 1) % cfg.log_every == 0:
            print(f"Step {step+1}: loss={loss.item():.4f}, pred={l_pred.item():.4f}, "
                  f"sig={l_sig.item():.4f}, bsim={l_bsim.item():.4f}, cost={l_cost.item():.4f}")

        if (step + 1) % cfg.save_every == 0:
            torch.save(model.state_dict(), cfg.checkpoint_path)
            print(f"Saved checkpoint at step {step+1}")

    torch.save(model.state_dict(), cfg.checkpoint_path)
    print("Training complete!")
    return model


if __name__ == "__main__":
    cfg = Config()
    train(cfg)
