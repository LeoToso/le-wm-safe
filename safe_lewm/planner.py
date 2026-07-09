"""
MPPI planner with conformal safety constraint.

Follows SLS-squared's planning stage:
  - Sample N action sequences
  - Roll out latent dynamics autoregressively
  - Cost = goal_distance + safety_penalty
  - Safety penalty: softplus(safe_threshold + margin - NN(z_t)) for each step
  - Importance-weighted action mean (MPPI update)

Reference: SLS-squared (trustworthyrobotics/SLS-squared), Algorithm 1 (MPPI warm-start).
"""
import torch
import torch.nn.functional as F


class MPPIPlanner:
    """
    MPPI planner that uses:
      - world model (SafeJEPA) for latent rollouts
      - ObstacleMLP classifier + conformal threshold for safety
    """

    def __init__(
        self,
        model,
        classifier,
        z_mean: torch.Tensor,
        z_std: torch.Tensor,
        safe_threshold: float,
        action_dim: int = 2,
        action_low: float = -1.0,
        action_high: float = 1.0,
        horizon: int = 10,
        n_samples: int = 512,
        temperature: float = 0.1,
        safety_margin: float = 0.2,
        safety_weight: float = 10.0,
        history_size: int = 3,
        device: str = "cpu",
    ):
        self.model = model
        self.classifier = classifier
        self.z_mean = z_mean.to(device)
        self.z_std  = z_std.to(device)
        self.safe_threshold = safe_threshold
        self.action_dim  = action_dim
        self.action_low  = action_low
        self.action_high = action_high
        self.horizon     = horizon
        self.n_samples   = n_samples
        self.temperature = temperature
        self.safety_margin  = safety_margin
        self.safety_weight  = safety_weight
        self.history_size   = history_size
        self.device = device
        self.U = torch.zeros(horizon, action_dim, device=device)

    def _normalize_z(self, z):
        return (z - self.z_mean) / self.z_std

    def _safety_penalty(self, z_flat):
        """softplus(threshold + margin - NN(z)) — 0 when safe, large when unsafe."""
        z_norm = self._normalize_z(z_flat)
        score = self.classifier(z_norm)
        return F.softplus(self.safe_threshold + self.safety_margin - score)

    @torch.no_grad()
    def plan(self, z_hist, z_goal, n_iterations=1):
        """
        Args:
            z_hist: (T_hist, z_dim) current latent history
            z_goal: (z_dim,) goal latent
            n_iterations: MPPI refinement iterations

        Returns:
            action: (action_dim,) first action of best plan
        """
        self.model.eval()
        self.classifier.eval()

        N, H = self.n_samples, self.horizon
        z_hist_b = z_hist.unsqueeze(0).expand(N, -1, -1).to(self.device)
        z_goal_b = z_goal.unsqueeze(0).to(self.device)

        for _ in range(n_iterations):
            noise = torch.randn(N, H, self.action_dim, device=self.device)
            U_b = (self.U.unsqueeze(0) + noise).clamp(self.action_low, self.action_high)

            z_seq    = self.model.rollout(z_hist_b, U_b, self.history_size)
            z_rolled = z_seq[:, -H:, :]          # (N, H, D)

            # Goal cost
            goal_cost = (z_rolled[:, -1, :] - z_goal_b).pow(2).sum(-1)  # (N,)

            # Safety cost — sum softplus penalty over horizon
            pen         = self._safety_penalty(z_rolled.reshape(N * H, -1))
            safety_cost = pen.reshape(N, H).sum(-1)

            total_cost = goal_cost + self.safety_weight * safety_cost

            beta    = total_cost.min()
            weights = torch.exp(-(total_cost - beta) / self.temperature)
            weights = weights / (weights.sum() + 1e-8)

            self.U = (weights[:, None, None] * U_b).sum(0).clamp(self.action_low, self.action_high)

        action   = self.U[0].clone()
        self.U   = torch.roll(self.U, -1, dims=0)
        self.U[-1] = 0.0
        return action

    def reset(self):
        self.U = torch.zeros(self.horizon, self.action_dim, device=self.device)


def load_planner(model, clf_path, device="cpu", horizon=10, n_samples=512,
                 temperature=0.1, safety_margin=0.2, safety_weight=10.0):
    """Load saved classifier checkpoint and build MPPIPlanner."""
    from safe_lewm.classifier import ObstacleMLP

    ckpt = torch.load(clf_path, map_location=device, weights_only=False)
    clf  = ObstacleMLP(
        z_dim=ckpt["z_dim"], hidden_dim=ckpt["hidden_dim"],
        depth=ckpt["depth"], dropout=ckpt.get("dropout", 0.1),
    ).to(device)
    clf.load_state_dict(ckpt["classifier_state"])
    clf.eval()

    planner = MPPIPlanner(
        model=model, classifier=clf,
        z_mean=ckpt["z_mean"], z_std=ckpt["z_std"],
        safe_threshold=ckpt["safe_threshold"],
        action_dim=model.cfg.action_dim,
        horizon=horizon, n_samples=n_samples, temperature=temperature,
        safety_margin=safety_margin, safety_weight=safety_weight,
        history_size=model.cfg.history_size, device=device,
    )
    print(f"Loaded classifier from {clf_path}")
    print(f"  Conformal threshold (delta={ckpt['delta']}): {ckpt['safe_threshold']:.4f}")
    return planner
