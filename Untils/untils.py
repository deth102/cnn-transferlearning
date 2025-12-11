import torch
# ========================
# Utilities: domain losses
# ========================

DOMAIN_LOSS = "mmd"  # "coral" hoặc "mmd"

def _covariance_matrix(x: torch.Tensor) -> torch.Tensor:
    n, d = x.shape
    if n <= 1:
        return torch.zeros((d, d), device=x.device, dtype=x.dtype)
    xc = x - x.mean(dim=0, keepdim=True)
    return xc.t().matmul(xc) / (n - 1)

def coral_loss(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """CORAL loss between src (n x d) and tgt (m x d)."""
    if src.numel() == 0 or tgt.numel() == 0:
        return torch.tensor(0.0, device=src.device)
    Cs = _covariance_matrix(src)
    Ct = _covariance_matrix(tgt)
    diff = Cs - Ct
    d = src.size(1)
    loss = torch.norm(diff, p="fro") ** 2
    return loss / (4.0 * (d ** 2))

def _rbf_kernel_matrix(x: torch.Tensor, y: torch.Tensor, kernel_mul: float = 2.0, kernel_num: int = 5) -> torch.Tensor:
    total = torch.cat([x, y], dim=0)
    L2 = ((total.unsqueeze(0) - total.unsqueeze(1)) ** 2).sum(dim=2)
    sigma = L2.mean().detach() + 1e-8
    sigmas = [sigma * (kernel_mul ** i) for i in range(kernel_num)]
    kernels = [torch.exp(-L2 / s) for s in sigmas]
    return sum(kernels)

def mmd_rbf_loss(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """RBF-MMD loss between src and tgt."""
    if src.size(0) < 1 or tgt.size(0) < 1:
        return torch.tensor(0.0, device=src.device)
    K = _rbf_kernel_matrix(src, tgt)
    m = src.size(0)
    n = tgt.size(0)
    K_ss = K[:m, :m]
    K_tt = K[m:, m:]
    K_st = K[:m, m:]
    loss = K_ss.mean() + K_tt.mean() - 2.0 * K_st.mean()
    return loss

def domain_loss_from_batch(features: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
    """
    Convenience wrapper: split features by domain_id (0=source,1=target)
    and compute chosen domain loss.
    """
    src = features[domain_ids == 0]
    tgt = features[domain_ids == 1]
    if DOMAIN_LOSS == "coral":
        return coral_loss(src, tgt)
    else:
        return mmd_rbf_loss(src, tgt)