import torch
import torch.nn.functional as F


def w2_diagonal(mu1, sigma1, mu2, sigma2):
    """W2 squared between diagonal Gaussians. All shapes (B, D)."""
    return (mu1 - mu2).pow(2).sum(-1) + (sigma1 - sigma2).pow(2).sum(-1)


def bisimulation_loss(z, rewards, costs, transition_model, actions, rho=0.1, gamma=0.99):
    """
    z: (B, D) embeddings
    rewards: (B,) reward scalars
    costs: (B,) cost scalars
    transition_model: outputs (mean, log_std) given (z, a)
    actions: (B, A)
    rho: cost weight
    gamma: discount
    """
    B = z.shape[0]
    idx = torch.randperm(B, device=z.device)
    half = B // 2
    i_idx = idx[:half]
    j_idx = idx[half: half * 2]

    z_i, z_j = z[i_idx], z[j_idx]
    r_i, r_j = rewards[i_idx], rewards[j_idx]
    c_i, c_j = costs[i_idx], costs[j_idx]
    a_i, a_j = actions[i_idx], actions[j_idx]

    # Latent distance (L1): gradients flow through encoder
    latent_dist = (z_i - z_j).abs().sum(-1)  # (B//2,)

    # Transition model — DETACH for target computation
    with torch.no_grad():
        mu_i, log_std_i = transition_model(z_i.detach(), a_i)
        mu_j, log_std_j = transition_model(z_j.detach(), a_j)
        sigma_i = log_std_i.exp()
        sigma_j = log_std_j.exp()
        w2_sq = w2_diagonal(mu_i, sigma_i, mu_j, sigma_j)
        w2_dist = w2_sq.sqrt().clamp(min=0)

    # Target (stop-gradient)
    target = (r_i - r_j).abs() + rho * (c_i - c_j).abs() + gamma * w2_dist
    target = target.detach()

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
