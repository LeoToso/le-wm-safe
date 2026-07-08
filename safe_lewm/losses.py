import torch
import torch.nn.functional as F


def w2_diagonal(mu1, sigma1, mu2, sigma2):
    """W2 squared between diagonal Gaussians. All shapes (B, D)."""
    return (mu1 - mu2).pow(2).sum(-1) + (sigma1 - sigma2).pow(2).sum(-1)


def bisimulation_loss(z, rewards, costs, transition_model, actions, rho=0.1, gamma=0.99):
    """
    Safety-only bisimulation: target = rho * |c_i - c_j|

    Only the safety cost term is kept — reward and next-state distribution
    (W2) terms are disabled. This forces the encoder to separate safe from
    unsafe states while remaining agnostic to reward structure.

    z: (B, D) embeddings
    costs: (B,) cost scalars
    rho: scales the safety separation
    rewards, transition_model, actions, gamma: unused, kept for API compatibility
    """
    B = z.shape[0]
    idx = torch.randperm(B, device=z.device)
    half = B // 2
    i_idx = idx[:half]
    j_idx = idx[half: half * 2]

    z_i, z_j = z[i_idx], z[j_idx]
    c_i, c_j = costs[i_idx], costs[j_idx]

    # Latent L1 distance — gradients flow through encoder
    latent_dist = (z_i - z_j).abs().sum(-1)  # (B//2,)

    # Safety-only target (stop-gradient)
    target = (rho * (c_i - c_j).abs()).detach()

    loss = 0.5 * (latent_dist - target).pow(2).mean()
    return loss


def pred_loss(pred_z, target_z):
    """MSE between predicted and actual next embeddings."""
    return F.mse_loss(pred_z, target_z.detach())


def cost_bce_loss(cost_pred, cost_true):
    """BCE for cost prediction. cost_pred: (B,1), cost_true: (B,)"""
    return F.binary_cross_entropy(cost_pred.squeeze(-1), cost_true.float())


def sigreg_loss(z, sigreg_module):
    """z: (B, T, D) -> scalar. Transposes to (T, B, D) for SIGReg."""
    z_tbf = z.permute(1, 0, 2)  # (T, B, D)
    return sigreg_module(z_tbf)
