import torch

# =======================================
# Gaussian RBF Kernel
# =======================================
def gaussian_kernel(x, y, kernel_mul=2.0, kernel_num=5):
    total = torch.cat([x, y], dim=0)
    L2 = ((total.unsqueeze(0) - total.unsqueeze(1)) ** 2).sum(dim=2)

    sigma = L2.mean().detach()
    kernels = [torch.exp(-L2 / (sigma * (kernel_mul ** i)))
               for i in range(kernel_num)]
    return sum(kernels)


# =======================================
# Domain MMD Loss (align Source ↔ Target)
# =======================================
def domain_mmd_loss(features, domain_ids):
    """
    features:   (B, D)
    domain_ids: (B,) với giá trị 0 (source) hoặc 1 (target)
    """

    src = features[domain_ids == 0]
    tgt = features[domain_ids == 1]

    if len(src) < 2 or len(tgt) < 2:
        return torch.tensor(0.0, device=features.device)

    kernels = gaussian_kernel(src, tgt)
    m, n = src.size(0), tgt.size(0)

    K_ss = kernels[:m, :m]
    K_tt = kernels[m:, m:]
    K_st = kernels[:m, m:]

    # MMD^2
    loss = K_ss.mean() + K_tt.mean() - 2 * K_st.mean()
    return loss
