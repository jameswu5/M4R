import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class Payoff(ABC):
    @abstractmethod
    def __call__(self, S, K):
        pass

    @abstractmethod
    def boundary_loss(self, model, t_boundary, S_boundary, **kwargs):
        pass

    @abstractmethod
    def sobolev_loss(self, model, **kwargs):
        pass


class Put(Payoff):
    def __call__(self, S, K):
        return torch.relu(K - S)

    def boundary_loss(self, model, t_boundary, S_boundary, **kwargs):
        K = kwargs.get('K', None)
        S_max = kwargs.get('S_max', None)[0]
        S_min = kwargs.get('S_min', None)[0]

        if K is None or S_max is None or S_min is None:
            raise ValueError("K, S_max, and S_min must be provided for boundary loss calculation.")

        length = t_boundary.shape[0]
        ones = torch.ones((length, 1))

        v_1 = model(ones, S_boundary)
        payoff = self(S_boundary, K)
        boundary_loss = nn.MSELoss()(v_1, payoff)

        # f(t, S_max) = 0
        v_Smax = model(t_boundary, ones * S_max)
        boundary_Smax_loss = nn.MSELoss()(v_Smax, torch.zeros((length, 1)))

        v_Smin = model(t_boundary, ones * S_min)
        boundary_Smin_loss = nn.MSELoss()(v_Smin, ones * (K - S_min))

        total_boundary_loss = 3 * boundary_loss + boundary_Smax_loss + boundary_Smin_loss

        return total_boundary_loss

    def sobolev_loss(self, model, S_interior, t1_interior, t2_interior, **kwargs):
        K = kwargs.get('K', None)
        a = kwargs.get('a', None)

        S_interior.requires_grad_(True)
        length = S_interior.shape[0]
        zeros = torch.zeros((length, 1))
        ones = torch.ones((length, 1))

        # --- w2: H^1 on Omega at t=0 ---
        # Compute u0(x) = f(0, x) - g(x)
        v0 = model(zeros, S_interior)
        g0 = self(S_interior, K)
        u0 = v0 - g0

        # L2 part (value)
        L2_w2 = torch.mean(u0 ** 2)

        # H^1 seminorm: ||du0/dx||_L2
        # Compute gradient of u0 wrt S_interior
        grad_u0 = torch.autograd.grad(u0, S_interior, grad_outputs=torch.ones_like(u0), create_graph=True)[0]
        H1_w2 = torch.mean(grad_u0 ** 2)

        w2 = L2_w2 + H1_w2

        # --- w3: mixed fractional norm on Sigma ---
        # Boundary points x1 = 1/a, x2 = a
        x1 = ones / a
        x2 = ones * a

        # Evaluate u at boundary times and points
        M = t1_interior.shape[0]
        x1M = x1.expand(M, -1)
        x2M = x2.expand(M, -1)
        v_t1_x1 = model(t1_interior, x1M)
        v_t1_x2 = model(t1_interior, x2M)
        v_t2_x1 = model(t2_interior, x1M)
        v_t2_x2 = model(t2_interior, x2M)

        g_x1 = self(x1M, K)
        g_x2 = self(x2M, K)

        u_t1_x1 = v_t1_x1 - g_x1
        u_t1_x2 = v_t1_x2 - g_x2
        u_t2_x1 = v_t2_x1 - g_x1
        u_t2_x2 = v_t2_x2 - g_x2

        dt = torch.abs(t1_interior - t2_interior)
        denom_t_w3 = (dt ** 2.5).clamp_min(1e-8)

        time_frac_x1 = torch.mean(((u_t1_x1 - u_t2_x1) ** 2) / denom_t_w3)
        time_frac_x2 = torch.mean(((u_t1_x2 - u_t2_x2) ** 2) / denom_t_w3)
        time_frac = 0.5 * (time_frac_x1 + time_frac_x2)

        dx = torch.abs(x2 - x1).clamp_min(1e-8)
        u_t1_xdiff = (model(t1_interior, x2M) - g_x2) - (model(t1_interior, x1M) - g_x1)
        spatial_frac = torch.mean((u_t1_xdiff ** 2) / (dx ** 4))

        val_term = 0.5 * (torch.mean(u_t1_x1 ** 2) + torch.mean(u_t1_x2 ** 2))

        w3 = val_term + time_frac + spatial_frac

        # --- w4: derivative norm on Sigma in H^{1/4, 1/2} ---
        dx_val = (x2 - x1).clamp_min(1e-8)
        u_t1_x1_v = model(t1_interior, x1M) - g_x1
        u_t1_x2_v = model(t1_interior, x2M) - g_x2
        du_est = (u_t1_x2_v - u_t1_x1_v) / dx_val

        deriv_L2 = torch.mean(du_est ** 2)

        denom_t_w4 = (dt ** 1.5).clamp_min(1e-8)

        du_t1 = (model(t1_interior, x2M) - self(x2M, K) - (model(t1_interior, x1M) - self(x1M, K))) / dx_val
        du_t2 = (model(t2_interior, x2M) - self(x2M, K) - (model(t2_interior, x1M) - self(x1M, K))) / dx_val
        time_frac_du = torch.mean(((du_t1 - du_t2) ** 2) / denom_t_w4)
        w4 = deriv_L2 + time_frac_du

        return w2 + w3 + w4


class PutProductMultipleAssets(Payoff):
    def __call__(self, S, K):
        product = torch.prod(S, dim=1, keepdim=True)
        return torch.relu(K - product)

    def boundary_loss(self, model, t_boundary, S_boundary, **kwargs):
        K = kwargs.get('K', None)
        S_max = kwargs.get('S_max', None)
        S_min = kwargs.get('S_min', None)

        # S_max shape: (n_assets,)
        # S_min shape: (n_assets,)

        if K is None or S_max is None or S_min is None:
            raise ValueError("K, S_max, and S_min must be provided for boundary loss calculation.")

        length = t_boundary.shape[0]
        ones = torch.ones((length, 1))

        v_1 = model(ones, S_boundary)
        payoff = self(S_boundary, K)
        exercise_loss = nn.MSELoss()(v_1, payoff)

        # f(t, S1, ..., SN) where any Si = Si_max = 0
        boundary_max_losses = []
        n_assets = S_boundary.shape[1]
        for i in range(n_assets):
            S_boundary_max = S_boundary.clone()
            S_boundary_max[:, i] = S_max[i]
            v_Si_max = model(t_boundary, S_boundary_max)
            boundary_loss_Si_max = nn.MSELoss()(v_Si_max, torch.zeros((length, 1)))
            boundary_max_losses.append(boundary_loss_Si_max)

        # f(t, S1, ..., SN) where any Si = Si_min = max(K - prod(S_min), 0)
        boundary_min_losses = []
        for i in range(n_assets):
            S_boundary_min = S_boundary.clone()
            S_boundary_min[:, i] = S_min[i]
            v_Si_min = model(t_boundary, S_boundary_min)
            boundary_loss_Si_min = nn.MSELoss()(v_Si_min, ones * torch.relu(K - torch.prod(S_boundary_min[0])))
            boundary_min_losses.append(boundary_loss_Si_min)

        total_boundary_loss = 3 * exercise_loss + sum(boundary_max_losses) + sum(boundary_min_losses)

        return total_boundary_loss

    def sobolev_loss(self, model):
        raise NotImplementedError("Sobolev loss is not implemented for PutProductMultipleAssets.")
