import torch
import torch.nn as nn
from .config import MarketParams


def compute_derivatives(model, t, S):
    t = t.requires_grad_(True)
    S = S.requires_grad_(True)
    v = model(t, S)
    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_S = torch.autograd.grad(v, S, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_SS = torch.autograd.grad(v_S, S, grad_outputs=torch.ones_like(v_S), create_graph=True)[0]
    return v, v_t, v_S, v_SS


def pde_residual(v_t, v_S, v_SS, v, S, r, sigma):
    residual = -v_t - r * S * v_S - 0.5 * sigma**2 * S**2 * v_SS + r * v
    return residual


def compute_derivatives_nd(model, t, S):
    t = t.requires_grad_(True)
    S = S.requires_grad_(True)
    v = model(t, S)

    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]  # shape (N, 1)
    v_S = torch.autograd.grad(v, S, grad_outputs=torch.ones_like(v), create_graph=True)[0]  # shape (N, n_assets)

    n_samples, n_assets = S.shape

    # Allocate Hessian tensor: shape (n_samples, n_assets, n_assets)
    H = torch.zeros((n_samples, n_assets, n_assets), dtype=v.dtype, device=v.device)

    for i in range(n_assets):
        vi = v_S[:, i].unsqueeze(-1)
        vi_S = torch.autograd.grad(vi, S, grad_outputs=torch.ones_like(vi), create_graph=True)[0]  # shape (N, n_assets)
        H[:, i, :] = vi_S

    # Now the H tensor is of form H[b, i, j] = d^2 v(b) / dS_i dS_j

    return v, v_t, v_S, H


def pde_residual_nd(v, v_t, v_S, H, S, r, Sigma):
    """
    Compute PDE residual for N-dimensional Black-Scholes operator.

    Parameters
    ----------
    v: (B, 1)           model value
    v_t: (B, 1)         time derivative
    v_S: (B, N)         first derivatives wrt S
    H: (B, N, N)        Hessian per batch
    S: (B, N)           asset prices (S1,...,SN)
    r: float            risk-free rate
    Sigma: (N, N) or (N,)
        If (N, N): full covariance matrix
        If (N,): diagonal variances (cross terms zero)
    """
    if not torch.is_tensor(Sigma):
        Sigma = torch.as_tensor(Sigma, dtype=S.dtype, device=S.device)
    else:
        Sigma = Sigma.to(dtype=S.dtype, device=S.device)

    B, N = S.shape

    drift = r * torch.sum(S * v_S, dim=1, keepdim=True)  # shape (B, 1)

    S_outer = S.unsqueeze(2) * S.unsqueeze(1)  # shape (B, N, N)

    # Broadcast Sigma to batch if needed:
    if Sigma.dim() == 1:
        Sigma_mat = torch.diag(Sigma).unsqueeze(0).expand(B, N, N)  # (B, N, N)
    elif Sigma.dim() == 2:
        Sigma_mat = Sigma.unsqueeze(0).expand(B, N, N)  # (B, N, N)
    else:
        raise ValueError("Sigma must be either (N,) or (N,N)")

    elements = Sigma_mat * S_outer * H  # shape (B, N, N)
    diffusion = 0.5 * torch.sum(elements, dim=(1, 2), keepdim=True)  # shape (B, 1)

    residual = -v_t - drift - diffusion + r * v  # shape (B, 1)
    return residual


def compute_derivatives_2d(model, t, S):
    t = t.requires_grad_(True)
    S = S.requires_grad_(True)   # requires_grad on the whole S tensor
    v = model(t, S)              # returns (N,1)

    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]

    v_S = torch.autograd.grad(v, S, grad_outputs=torch.ones_like(v), create_graph=True)[0]  # shape (N,2)
    v_S1 = v_S[:, 0:1]
    v_S2 = v_S[:, 1:2]

    v_S1_wrt_S = torch.autograd.grad(v_S1, S, grad_outputs=torch.ones_like(v_S1), create_graph=True)[0]
    v_S1S1 = v_S1_wrt_S[:, 0:1]
    v_S1S2 = v_S1_wrt_S[:, 1:2]   # derivative of v_S1 wrt S2
    v_S2S2 = torch.autograd.grad(v_S2, S, grad_outputs=torch.ones_like(v_S2), create_graph=True)[0][:, 1:2]

    return v, v_t, v_S1, v_S2, v_S1S1, v_S2S2, v_S1S2


def pde_residual_2d(v_t, v_S1, v_S2, v_S1S1, v_S2S2, v_S1S2, v, S1, S2, r, sigma1, sigma2, rho):
    residual = -v_t - r * (S1 * v_S1 + S2 * v_S2) - 0.5 * sigma1**2 * S1**2 * v_S1S1 - 0.5 * sigma2**2 * S2**2 * v_S2S2 - rho * sigma1 * sigma2 * S1 * S2 * v_S1S2 + r * v
    return residual
