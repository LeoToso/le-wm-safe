"""
Conformal calibration of the safety classifier.

Nonconformity score (on unsafe/obstacle samples only):
    s_i = max(0, NN(z_i))
    — positive when the classifier incorrectly says "safe" for an unsafe state.

Threshold at miscoverage level delta:
    rank = ceil((n + 1) * (1 - delta)) - 1
    safe_threshold = sorted(scores)[rank]

Guarantee: with probability >= 1 - delta over future latent states,
any truly-unsafe state z satisfies NN(z) <= safe_threshold.
At planning time, enforce NN(z) >= safe_threshold to exclude unsafe states.
"""
import math
import numpy as np
import torch


def compute_nonconformity_scores(
    classifier: torch.nn.Module,
    z_unsafe: torch.Tensor,
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute nonconformity scores for held-out unsafe (obstacle) latent vectors.

    Args:
        classifier: trained ObstacleMLP
        z_unsafe: (N, z_dim) latent vectors from unsafe/cost=1 observations
        device: torch device string

    Returns:
        scores: (N,) numpy array of max(0, NN(z)) values
    """
    classifier.eval()
    z_unsafe = z_unsafe.to(device)
    with torch.no_grad():
        raw_scores = classifier(z_unsafe).cpu().numpy()  # (N,)
    return np.maximum(0.0, raw_scores)


def calibrate(scores: np.ndarray, delta: float = 0.1) -> float:
    """
    Standard split conformal calibration.

    Args:
        scores: nonconformity scores from calibration unsafe samples
        delta: desired miscoverage level (e.g. 0.1 for 90% coverage)

    Returns:
        safe_threshold: scalar — at planning time enforce NN(z) >= safe_threshold
    """
    n = len(scores)
    rank = math.ceil((n + 1) * (1 - delta)) - 1
    rank = max(0, min(rank, n - 1))
    sorted_scores = np.sort(scores)
    threshold = float(sorted_scores[rank])
    return threshold


def calibrate_classifier(
    classifier: torch.nn.Module,
    z_cal_unsafe: torch.Tensor,
    delta: float = 0.1,
    device: str = "cpu",
) -> float:
    """
    End-to-end: compute scores + return conformal threshold.

    Args:
        classifier: trained ObstacleMLP
        z_cal_unsafe: (N, z_dim) calibration latent vectors from unsafe observations
        delta: miscoverage level
        device: torch device

    Returns:
        safe_threshold: float
    """
    scores = compute_nonconformity_scores(classifier, z_cal_unsafe, device)
    threshold = calibrate(scores, delta)
    print(f"Conformal calibration: n={len(scores)}, delta={delta:.2f} -> threshold={threshold:.4f}")
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}], "
          f"mean={scores.mean():.4f}, std={scores.std():.4f}")
    return threshold
