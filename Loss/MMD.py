import torch

def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = source.size(0) + target.size(0)
    total = torch.cat([source, target], dim=0)

    L2 = ((total.unsqueeze(0) - total.unsqueeze(1)) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2.data) / (n_samples**2 - n_samples)

    bandwidth /= (kernel_mul ** (kernel_num // 2))

    kernel_vals = [
        torch.exp(-L2 / (bandwidth * (kernel_mul ** i)))
        for i in range(kernel_num)
    ]

    return sum(kernel_vals)


def domain_mmd_loss(features, labels, domain_ids):
    """
    features: (B, feat_dim)
    labels:   (B,)
    domain_ids: (B,)
    """

    # Lấy Normal class (label == 0)
    normal_mask = (labels == 0)
    if normal_mask.sum() < 2:
        return torch.tensor(0.0, device=features.device)

    normal_feat = features[normal_mask]
    normal_domains = domain_ids[normal_mask]

    # Domain 0nm = 0, Domain 2nm = 1
    src = normal_feat[normal_domains == 0]
    tgt = normal_feat[normal_domains == 1]

    if len(src) == 0 or len(tgt) == 0:
        return torch.tensor(0.0, device=features.device)

    # Compute kernel matrix
    kernels = gaussian_kernel(src, tgt)

    m = src.size(0)
    n = tgt.size(0)

    XX = kernels[:m, :m]
    YY = kernels[m:, m:]
    XY = kernels[:m, m:]
    YX = kernels[m:, :m]

    # ---- CÔNG THỨC ĐÚNG KHÔNG PHỤ THUỘC SHAPE ----
    loss = XX.mean() + YY.mean() - XY.mean() - YX.mean()
    return loss
