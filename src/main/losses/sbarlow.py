import torch
import torch.nn.functional as F

from solo.losses.util import tuple2tensor, autocorr_asym
from solo.losses.util import make_idx


def sbarlow_loss_func(
    z1: torch.Tensor,
    z2: torch.Tensor,
    lamb: float = 5e-3,
    scale_loss: float = 0.025,
    exponent: int = 2,
    group_size: int = 2048,
    rand_idx: torch.Tensor = None,
) -> torch.Tensor:
    """Computes Barlow Twins style loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        lamb (float, optional): off-diagonal scaling factor for the cross-covariance matrix.
            Defaults to 5e-3.
        scale_loss (float, optional): final scaling factor of the loss. Defaults to 0.025.

    Returns:
        torch.Tensor: loss.
    """

    N, D = z1.size()

    # to match the original code
    bn = torch.nn.BatchNorm1d(D, affine=False).to(z1.device)
    z1 = bn(z1)  # N x D
    z2 = bn(z2)  # N x D

    # diagonal loss
    var = torch.sum(z1 * z2, dim=0) / N  # diagonal elements in C: D
    on_diag = F.mse_loss(var, torch.ones_like(var), reduction="sum")

    # feature permutation
    if rand_idx != None:
        z1 = z1[:, rand_idx]  # N x D
        z2 = z2[:, rand_idx]  # N x D

    # grouping
    group_z1 = z1.split(group_size, dim=1)  # tuple: num_group x N x group_size
    group_z2 = z2.split(group_size, dim=1)  # tuple: num_group x N x group_size
    group_size_last = group_z1[-1].shape[1]
    Z1 = tuple2tensor(
        group_z1, group_size, group_size_last
    )  # num_groups x N x group_size
    Z2 = tuple2tensor(
        group_z2, group_size, group_size_last
    )  # num_groups x N x group_size

    # off-diagonal loss in diagonal sub matrices
    # circular correlation
    corr_vec = autocorr_asym(Z1, Z2) / N  # num_groups x group_size
    # exclude 0-th value
    if exponent == 1:
        off_diag = torch.sum(corr_vec[:, 1:].abs())  # 1
    elif exponent == 2:
        off_diag = torch.sum(corr_vec[:, 1:].pow(2))  # 1

    # off-diagonal loss in off-diagonal sub-matrices
    if Z1.shape[0] > 1:
        # make indexes to compute the combination of sub-matrices
        idx_z1, idx_z2, idx_last = make_idx(Z1.shape[0])
        Z1_off = Z1[idx_z1, :, :]  # B_off x N x group_size
        Z2_off = Z2[idx_z2, :, :]  # B_off x N x group_size
        # circualr correlation
        corr_vec = autocorr_asym(Z1_off, Z2_off) / N  # B_off x group_size
        # include 0-th value
        if exponent == 1:
            off_diag += torch.sum(corr_vec.abs())  # 1
        elif exponent == 2:
            off_diag += torch.sum(corr_vec.pow(2))  # 1

    loss = scale_loss * (on_diag + lamb * off_diag)

    return loss
