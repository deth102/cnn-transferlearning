import torch
from typing import Tuple

def _covariance_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    Compute unbiased covariance matrix of x (n x d).
    Returns (d x d) covariance matrix. If n <= 1 returns zero matrix.
    """
    n, d = x.shape
    if n <= 1:
        return torch.zeros((d, d), device=x.device, dtype=x.dtype)
    x_centered = x - x.mean(dim=0, keepdim=True)
    # unbiased estimator (divide by n-1)
    cov = x_centered.t().matmul(x_centered) / (n - 1)
    return cov

def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    CORAL loss between source (Ns x D) and target (Nt x D).
    Formula: (1 / (4 * d^2)) * ||Cs - Ct||_F^2
    """
    if source.numel() == 0 or target.numel() == 0:
        return torch.tensor(0.0, device=source.device if source.numel() else target.device)

    # ensure 2D float tensors
    src = source.float().reshape(source.size(0), -1)
    tgt = target.float().reshape(target.size(0), -1)
    d = src.size(1)

    Cs = _covariance_matrix(src)
    Ct = _covariance_matrix(tgt)

    diff = Cs - Ct
    # squared Frobenius norm
    loss = torch.norm(diff, p='fro') ** 2
    loss = loss / (4.0 * (d ** 2))
    return loss

def coral_loss_from_batch(features: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
    """
    Convenience wrapper: split features by domain_ids (0=source, 1=target)
    and compute coral_loss. Returns 0 if either domain has <2 samples.
    features: (B, D)
    domain_ids: (B,) values in {0,1}
    """
    src = features[domain_ids == 0]
    tgt = features[domain_ids == 1]

    if src.size(0) < 2 or tgt.size(0) < 2:
        return torch.tensor(0.0, device=features.device, dtype=features.dtype)

    return coral_loss(src, tgt)
