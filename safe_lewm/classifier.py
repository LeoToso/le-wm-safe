"""
Safety classifier trained on frozen encoder latents.
Architecture follows SLS-squared (ObstacleMLP): small MLP with LayerNorm + GELU + Dropout.
Loss: signed hinge loss  mean(clamp(m - y * score, min=0))  with y=+1 safe, y=-1 unsafe.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ObstacleMLP(nn.Module):
    """
    Maps a latent vector z -> scalar score.
    Positive score = safe, negative score = unsafe/obstacle.
    """
    def __init__(self, z_dim: int, hidden_dim: int = 64, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = z_dim
        for _ in range(depth):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (..., z_dim) -> score: (...,)"""
        return self.net(z).squeeze(-1)


def hinge_loss(scores: torch.Tensor, labels: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """
    scores: (B,) raw classifier output
    labels: (B,) float — 1.0 for safe, -1.0 for unsafe
    loss = mean(clamp(margin - labels * scores, min=0))
    """
    return F.relu(margin - labels * scores).mean()
