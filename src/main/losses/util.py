import torch
import torch.nn.functional as F
import torch.fft as fft


def tuple2tensor(group_z: list, group_size: int, group_size_last: int) -> torch.Tensor:
    # pad zero if the last group has smaller dimension size
    if group_size > group_size_last:
        pad_dim = group_size - group_size_last
        pad_z = F.pad(group_z[-1], (0, pad_dim), "constant", 0)  # N x group_size
        # tuple2list to replace the last group
        group_z = list(group_z)
        group_z[-1] = pad_z
    Z = torch.stack(group_z, dim=0)  # num_groups x N x group_size
    return Z


def make_idx(num_groups: int) -> (list, list, torch.Tensor):
    z1 = []
    z2 = []
    last = []
    for i in range(num_groups):
        for j in range(num_groups):
            if i != j:
                z1.append(i)
                z2.append(j)
                if i == num_groups - 1 or j == num_groups - 1:
                    last.append(1)
                else:
                    last.append(0)
    last = torch.Tensor(last)
    return z1, z2, last


def make_idx_tri(num_groups: int) -> (list, list, torch.Tensor):
    # make indexes to compute the upper triangle sub-matrices
    z1 = []
    z2 = []
    last = []
    for i in range(num_groups):
        for j in range(i, num_groups):
            if i != j:
                z1.append(i)
                z2.append(j)
                if i == num_groups - 1 or j == num_groups - 1:
                    last.append(1)
                else:
                    last.append(0)
    last = torch.Tensor(last)
    return z1, z2, last


def autocorr(z: torch.Tensor) -> torch.Tensor:
    assert z.dim() == 3, "not batch"
    fz = fft.rfft(z)  # B x N x Df (= 1 + D / 2)
    fz_conj = fz.conj()  # B x N x Df
    fz_prod = fz_conj * fz  # B x N x Df
    fc = torch.sum(fz_prod, dim=1)  # B x Df
    corr_vec = fft.irfft(fc)  # B x D (= 2 * (Df - 1))
    assert z.shape[2] == corr_vec.shape[1]
    return corr_vec  # B x D


def autocorr_asym(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    assert z1.dim() == 3, "not batch"
    assert z2.dim() == 3, "not batch"
    fz1 = fft.rfft(z1)  # B x N x Df
    fz2 = fft.rfft(z2)  # B x N x Df
    fz1_conj = fz1.conj()  # B x N x Df
    fz_prod = fz1_conj * fz2  # B x N x Df
    fc = torch.sum(fz_prod, dim=1)  # B x Df
    corr_vec = fft.irfft(fc)  # B x D
    assert z1.shape[2] == corr_vec.shape[1]
    return corr_vec  # B x D
