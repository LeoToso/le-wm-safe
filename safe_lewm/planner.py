import torch
import torch.nn.functional as F


class CEMPlanner:
    """
    Cross-Entropy Method planner with safety filter.

    At each step:
    1. Sample action sequences from current distribution
    2. Rollout in latent space
    3. Score with reward cost (||z_H - z_goal||^2) + safety_cost (cumulative predicted cost)
    4. Update elite distribution
    5. Safety filter: pick best plan that satisfies C_safety <= threshold
    """

    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.device = next(model.parameters()).device

    def plan(self, z_hist, z_goal, action_dim):
        """
        z_hist: (1, T_hist, D) current latent history
        z_goal: (1, D) goal embedding
        Returns: best_action (action_dim,)
        """
        cfg = self.cfg
        H = cfg.cem_horizon
        N = cfg.cem_samples
        K = cfg.cem_elites

        # Initialize action distribution
        mu = torch.zeros(H, action_dim, device=self.device)
        std = torch.ones(H, action_dim, device=self.device)

        act_low = -1.0
        act_high = 1.0

        for it in range(cfg.cem_iterations):
            # Sample N action sequences: (N, H, A)
            eps = torch.randn(N, H, action_dim, device=self.device)
            actions = (mu.unsqueeze(0) + std.unsqueeze(0) * eps).clamp(act_low, act_high)

            # Expand z_hist: (N, T_hist, D)
            T_hist = z_hist.shape[1]
            z_curr = z_hist.expand(N, -1, -1)
            z_roll = z_curr.clone()

            safety_costs = torch.zeros(N, device=self.device)

            for t in range(H):
                act_t = actions[:, t, :]  # (N, A)
                HS = min(cfg.history_size, z_roll.shape[1])
                z_trunc = z_roll[:, -HS:]         # (N, HS, D)
                act_trunc = act_t.unsqueeze(1)    # (N, 1, A)
                act_emb = self.model.action_encoder(act_trunc)  # (N, 1, emb_dim)

                if HS > 1:
                    pad_emb = torch.zeros(N, HS - 1, act_emb.shape[-1], device=self.device)
                    act_emb_full = torch.cat([pad_emb, act_emb], dim=1)  # (N, HS, emb_dim)
                else:
                    act_emb_full = act_emb

                pred = self.model.predictor(z_trunc, act_emb_full)  # (N, HS, D)
                z_next = pred[:, -1:, :]  # (N, 1, D)

                with torch.no_grad():
                    c_hat = self.model.cost_head(z_next.squeeze(1))  # (N, 1)
                    safety_costs += c_hat.squeeze(-1)

                z_roll = torch.cat([z_roll, z_next], dim=1)

            # Final predicted embedding
            z_final = z_roll[:, -1, :]  # (N, D)

            # Reward cost: distance to goal
            reward_cost = F.mse_loss(z_final, z_goal.expand(N, -1), reduction='none').sum(-1)  # (N,)

            # Total cost for ranking
            total_cost = reward_cost + cfg.alpha_safety * safety_costs  # (N,)

            # Select elites
            elite_idx = total_cost.argsort()[:K]
            elite_actions = actions[elite_idx]  # (K, H, A)

            # Update distribution
            mu = elite_actions.mean(0)
            std = elite_actions.std(0) + 1e-6

        # Safety filter: among elites, pick plan satisfying safety constraint
        elite_safety = safety_costs[elite_idx]  # (K,)
        elite_reward = reward_cost[elite_idx]   # (K,)

        safe_mask = elite_safety <= cfg.safety_threshold
        if safe_mask.any():
            safe_elite_reward = elite_reward.clone()
            safe_elite_reward[~safe_mask] = float('inf')
            best_idx = safe_elite_reward.argmin()
        else:
            best_idx = 0  # already sorted by total cost

        best_action = elite_actions[best_idx, 0, :]  # first action of best plan
        return best_action.cpu().numpy()
