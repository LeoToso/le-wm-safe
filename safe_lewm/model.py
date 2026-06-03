import torch
from torch import nn

from .config import Config
from .modules import (
    CNNEncoder,
    ARPredictor,
    Embedder,
    TransitionModel,
    CostHead,
    SIGReg,
)


class SafeJEPA(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.encoder = CNNEncoder(cfg)

        self.action_encoder = Embedder(
            input_dim=cfg.action_dim,
            smoothed_dim=cfg.action_smoothed_dim,
            emb_dim=cfg.action_emb_dim,
        )

        self.predictor = ARPredictor(
            num_frames=cfg.num_frames,
            depth=cfg.pred_depth,
            heads=cfg.pred_heads,
            dim_head=cfg.pred_dim_head,
            mlp_dim=cfg.pred_mlp_dim,
            input_dim=cfg.z_dim,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.z_dim,
        )

        self.transition_model = TransitionModel(cfg.z_dim, cfg.action_dim, cfg.trans_hidden)
        self.cost_head = CostHead(cfg.z_dim)
        self.sigreg = SIGReg(cfg.sig_knots, cfg.sig_num_proj)

    def encode(self, pixels):
        """pixels: (B, T, C, H, W) -> z: (B, T, z_dim)"""
        B, T = pixels.shape[:2]
        flat = pixels.view(B * T, *pixels.shape[2:])
        z = self.encoder(flat)
        return z.view(B, T, -1)

    def predict(self, z_hist, act_hist):
        """z_hist: (B, T, D), act_hist: (B, T, A) -> pred: (B, T, D)"""
        act_emb = self.action_encoder(act_hist)  # (B, T, emb_dim)
        return self.predictor(z_hist, act_emb)

    def rollout(self, z0, actions, history_size=3):
        """
        z0: (B, T_hist, D) initial latent history
        actions: (B, T, A) action sequence to rollout
        Returns: z_seq (B, T_hist + T, D) full sequence of embeddings
        """
        B, T_hist, D = z0.shape
        T = actions.shape[1]

        z_roll = z0.clone()  # (B, T_hist, D)

        for t in range(T):
            HS = min(history_size, z_roll.shape[1])
            z_trunc = z_roll[:, -HS:]  # (B, HS, D)
            act_t = actions[:, t:t+1, :]  # (B, 1, A)

            # Build action embeddings: pad with zeros for history steps
            act_emb_t = self.action_encoder(act_t)  # (B, 1, emb_dim)
            if HS > 1:
                pad_emb = torch.zeros(B, HS - 1, act_emb_t.shape[-1], device=z0.device)
                act_emb_full = torch.cat([pad_emb, act_emb_t], dim=1)  # (B, HS, emb_dim)
            else:
                act_emb_full = act_emb_t

            pred = self.predictor(z_trunc, act_emb_full)  # (B, HS, D)
            z_next = pred[:, -1:, :]  # (B, 1, D)
            z_roll = torch.cat([z_roll, z_next], dim=1)

        return z_roll  # (B, T_hist + T, D)

    def get_cost(self, pixels_hist, actions, goal_pixels):
        """
        Full pipeline for CEM planning.
        pixels_hist: (B, T_hist, C, H, W)
        actions: (B, T, A)
        goal_pixels: (B, C, H, W)
        Returns: reward_cost (B,), safety_cost (B,)
        """
        B = pixels_hist.shape[0]

        # Encode history
        z_hist = self.encode(pixels_hist)  # (B, T_hist, D)

        # Encode goal
        z_goal = self.encoder(goal_pixels)  # (B, D)

        # Rollout
        z_seq = self.rollout(z_hist, actions, self.cfg.history_size)  # (B, T_hist+T, D)
        z_final = z_seq[:, -1, :]  # (B, D)

        # Reward cost: distance to goal
        reward_cost = (z_final - z_goal).pow(2).sum(-1)  # (B,)

        # Safety cost: cumulative predicted cost along rollout
        T = actions.shape[1]
        z_rollout = z_seq[:, -T:, :]  # (B, T, D)
        z_flat = z_rollout.reshape(B * T, -1)
        cost_pred = self.cost_head(z_flat).reshape(B, T)  # (B, T)
        safety_cost = cost_pred.sum(-1)  # (B,)

        return reward_cost, safety_cost
