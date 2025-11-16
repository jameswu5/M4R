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


