import torch
import torch.nn.functional as F

from solo.losses.util import tuple2tensor, autocorr, autocorr_asym
from solo.losses.util import make_idx_tri


def invariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Computes mse loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.

    Returns:
        torch.Tensor: invariance loss (mean squared error).
    """

    return F.mse_loss(z1, z2)


def covariance_loss(
    z: torch.Tensor,
    exponent: int = 2,
    group_size: int = 2048,
    rand_idx: torch.Tensor = None,
    eps: float = 1e-4,
) -> [torch.Tensor, torch.Tensor]:
    N, D = z.size()
    z = z - z.mean(dim=0)  # centering: N x D

    # diagonal loss
    var = torch.sum(z * z, dim=0) / (N - 1)  # variance: D
    std = torch.sqrt(var + eps)  # std: D
    on_diag = torch.mean(F.relu(1 - std))  # 1

    # permutation
    if rand_idx != None:
        z = z[:, rand_idx]  # N x D

    # grouping
    group_z = z.split(
        group_size, dim=1
    )  # len(tuple): num_groups, tuple: N x group_size
    group_size_last = group_z[-1].shape[1]
    Z = tuple2tensor(
        group_z, group_size, group_size_last
    )  # num_groups x N x group_size

    # cal loss on diagonal sub-matrices
    # circular correlation
    corr_vec = autocorr(Z) / (N - 1)  # num_groups x group_size

    # normalize
    corr_vec[:-1, :] /= group_size
    corr_vec[-1, :] /= group_size_last

    # off-diagonal loss in diagonal sub-matrices
    if exponent == 1:
        group_corr_vec = torch.sum(corr_vec[:, 1:].abs(), dim=1)  # num_groups
    elif exponent == 2:
        group_corr_vec = torch.sum(corr_vec[:, 1:].pow(2), dim=1)  # num_groups
    off_diag = torch.mean(group_corr_vec)  # 1

    # off-diagonal loss in off-diagonal sub-matrices
    if Z.shape[0] > 1:
        idx_z1, idx_z2, idx_last = make_idx_tri(Z.shape[0])
        Z1 = Z[idx_z1, :, :]  # B_off x N x group_size
        Z2 = Z[idx_z2, :, :]  # B_off x N x group_size

        # circualr correlation
        corr_vec = autocorr_asym(Z1, Z2) / (N - 1)  # B_off x group_size

        # normalize
        corr_vec[idx_last == 0, :] /= group_size
        corr_vec[idx_last == 1, :] /= group_size_last

        # cal loss on all elements
        if exponent == 1:
            group_corr_vec = torch.sum(corr_vec.abs(), dim=1)  # B_off
        elif exponent == 2:
            group_corr_vec = torch.sum(corr_vec.pow(2), dim=1)  # B_off
        off_diag += 2 * torch.mean(group_corr_vec)  # 1

    return on_diag, off_diag


#################


def svicreg_loss_func(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sim_loss_weight: float = 25.0,
    var_loss_weight: float = 25.0,
    cov_loss_weight: float = 1.0,
    scale_loss: float = 1.0,
    exponent: int = 2,
    group_size: int = 2048,
    rand_idx: torch.Tensor = None,
) -> torch.Tensor:
    """Computes VICReg style loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        sim_loss_weight (float): invariance loss weight.
        var_loss_weight (float): variance loss weight.
        cov_loss_weight (float): covariance loss weight.

    Returns:
        torch.Tensor: VICReg style loss.
    """

    sim_loss = invariance_loss(z1, z2)

    z1_on_diag, z1_off_diag = covariance_loss(z1, exponent, group_size, rand_idx)
    z2_on_diag, z2_off_diag = covariance_loss(z2, exponent, group_size, rand_idx)
    var_loss = z1_on_diag + z2_on_diag
    cov_loss = z1_off_diag + z2_off_diag

    loss = (
        sim_loss_weight * sim_loss
        + var_loss_weight * var_loss
        + cov_loss_weight * cov_loss
    )
    loss *= scale_loss
    return loss
