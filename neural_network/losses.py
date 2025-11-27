import torch
import torch.nn as nn


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
