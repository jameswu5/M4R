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


class Call(Payoff):
    def __call__(self, S, K):
        return torch.relu(S - K)

    def boundary_loss(self, model, t_boundary, S_boundary, **kwargs):
        raise NotImplementedError("Boundary loss not implemented for Call payoff.")

    def sobolev_loss(self, model, **kwargs):
        raise NotImplementedError("Sobolev loss not implemented for Call payoff.")

    def heston_loss(self, model, t, S, V, **kwargs):
        market_params = kwargs.get('market_params', None)
        if market_params is None:
            raise ValueError("market_params must be provided for Heston loss calculation.")

        # European boundary conditions
        K = market_params.K
        S_max = market_params.S_max
        V_max = market_params.V_max
        T = market_params.T

        ones = torch.ones_like(t)
        zeros = torch.zeros_like(t)

        # J2
        payoff_loss = torch.mean((
            model(ones * T, S, V) - torch.maximum(S - K * ones, zeros)
        )**2)

        # J3
        S_min_loss = torch.mean((
            model(t, zeros, V)
        )**2)

        # J4
        S_max_tensor = (ones * S_max).requires_grad_(True)
        f_Smax = model(t, S_max_tensor, V)
        df_Smax_dS = torch.autograd.grad(
            f_Smax, S_max_tensor, grad_outputs=torch.ones_like(f_Smax), create_graph=True, retain_graph=True
        )[0]

        S_max_loss = torch.mean((
            df_Smax_dS - ones
        )**2)

        # J5
        V_min = zeros.requires_grad_(True)
        f_Vmin = model(t, S, V_min)
        df_dt = torch.autograd.grad(
            f_Vmin, t, grad_outputs=torch.ones_like(f_Vmin), create_graph=True, retain_graph=True
        )[0]
        df_dS = torch.autograd.grad(
            f_Vmin, S, grad_outputs=torch.ones_like(f_Vmin), create_graph=True, retain_graph=True
        )[0]
        df_dV = torch.autograd.grad(
            f_Vmin, V_min, grad_outputs=torch.ones_like(f_Vmin), create_graph=True, retain_graph=True
        )[0]
        r = market_params.r
        kappa = market_params.kappa
        theta = market_params.theta

        V_min_loss = torch.mean((
            r * S * df_dS + kappa * theta * df_dV - r * f_Vmin + df_dt
        )**2)

        V_max_loss = torch.mean((
            model(t, S, ones * V_max) - S
        )**2)

        return payoff_loss, S_min_loss, S_max_loss, V_min_loss, V_max_loss


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

        # exercise loss: f(T, S) = payoff(S)
        v_1 = model(ones, S_boundary)
        payoff = self(S_boundary, K)
        exercise_loss = nn.MSELoss()(v_1, payoff)

        # S_max loss: f(t, S_max) = 0
        v_Smax = model(t_boundary, ones * S_max)
        boundary_Smax_loss = nn.MSELoss()(v_Smax, torch.zeros((length, 1)))

        # S_min loss: f(t, S_min) = K - S_min
        v_Smin = model(t_boundary, ones * S_min)
        boundary_Smin_loss = nn.MSELoss()(v_Smin, ones * (K - S_min))

        return exercise_loss, boundary_Smax_loss, boundary_Smin_loss

    def sobolev_loss(self, model, **kwargs):
        K = kwargs.get('K', None)
        # a = kwargs.get('a', None)
        S_max = kwargs.get('S_max', None)[0]
        S_min = kwargs.get('S_min', None)[0]

        S_interior = kwargs.get('S_interior', None)
        t1_interior = kwargs.get('t1_interior', None)
        t2_interior = kwargs.get('t2_interior', None)

        if K is None or S_max is None or S_min is None or S_interior is None or t1_interior is None or t2_interior is None:
            raise ValueError("K, S_max, S_min, S_interior, t1_interior, and t2_interior must be provided for Sobolev loss calculation.")

        S_interior.requires_grad_(True)
        length = S_interior.shape[0]
        zeros = torch.zeros((length, 1))
        ones = torch.ones((length, 1))

        # --- J2: H^1 on Omega at t=0 ---
        # Compute u0(x) = f(0, x) - g(x)
        v0 = model(zeros, S_interior)
        g0 = self(S_interior, K)
        u0 = v0 - g0

        # L2 part (value)
        L2_J2 = torch.mean(u0 ** 2)

        # H^1 seminorm: ||du0/dx||_L2
        # Compute gradient of u0 wrt S_interior
        grad_u0 = torch.autograd.grad(u0, S_interior, grad_outputs=torch.ones_like(u0), create_graph=True)[0]
        H1_J2 = torch.mean(grad_u0 ** 2)

        J2 = L2_J2 + H1_J2

        # --- J3: mixed fractional norm on Sigma ---
        # Boundary points x1 = S_min, x2 = S_max
        x1 = ones * S_min
        x2 = ones * S_max

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
        denom_t_J3 = (dt ** 2.5).clamp_min(1e-8)

        time_frac_x1 = torch.mean(((u_t1_x1 - u_t2_x1) ** 2) / denom_t_J3)
        time_frac_x2 = torch.mean(((u_t1_x2 - u_t2_x2) ** 2) / denom_t_J3)
        time_frac = time_frac_x1 + time_frac_x2

        # Value term (L2 over Sigma) averaged across two boundary batches
        val_term = 0.5 * (torch.mean(u_t1_x1 ** 2) + torch.mean(u_t1_x2 ** 2))

        J3 = val_term + time_frac

        # --- J4: derivative norm on Sigma in H^{1/4, 1/2} ---
        # Val term remains the same
        # Time frac is same as J3 but with different exponent
        denom_t_J4 = (dt ** 1.5).clamp_min(1e-8)
        time_frac_x1_J4 = torch.mean(((u_t1_x1 - u_t2_x1) ** 2) / denom_t_J4)
        time_frac_x2_J4 = torch.mean(((u_t1_x2 - u_t2_x2) ** 2) / denom_t_J4)
        time_frac_J4 = time_frac_x1_J4 + time_frac_x2_J4

        # Spatial frac:
        x_denom_J4 = 2 * (S_min**2 + S_max**2)
        du_val = (u_t1_x2 - u_t1_x1) ** 2
        spatial_frac_J4 = torch.mean(du_val / x_denom_J4)

        J4 = val_term + time_frac_J4 + spatial_frac_J4

        return J2, J3, J4

    def heston_loss(self, model, t, S, V, **kwargs):
        market_params = kwargs.get('market_params', None)
        if market_params is None:
            raise ValueError("market_params must be provided for Heston loss calculation.")

        # American boundary conditions
        K = market_params.K
        S_max = market_params.S_max
        V_max = market_params.V_max
        T = market_params.T

        ones = torch.ones_like(t)
        zeros = torch.zeros_like(t)

        # J2
        payoff_loss = torch.mean((
            model(ones * T, S, V) - self(S, K)
        )**2)

        # J3
        S_min_loss = torch.mean((
            model(t, zeros, V) - K * ones
        )**2)

        # J4
        S_max_loss = torch.mean((
            model(t, ones * S_max, V)
        )**2)

        # J5
        V_min = zeros.requires_grad_(True)
        f_Vmin = model(t, S, V_min)
        df_dt = torch.autograd.grad(
            f_Vmin, t, grad_outputs=torch.ones_like(f_Vmin), create_graph=True, retain_graph=True
        )[0]
        df_dS = torch.autograd.grad(
            f_Vmin, S, grad_outputs=torch.ones_like(f_Vmin), create_graph=True, retain_graph=True
        )[0]
        df_dV = torch.autograd.grad(
            f_Vmin, V_min, grad_outputs=torch.ones_like(f_Vmin), create_graph=True, retain_graph=True
        )[0]
        r = market_params.r
        kappa = market_params.kappa
        theta = market_params.theta

        V_min_loss = torch.mean((
            r * S * df_dS + kappa * theta * df_dV - r * f_Vmin + df_dt
        )**2)

        # J6
        V_max_loss = torch.mean((
            model(t, S, ones * V_max) - self(S, K)
        )**2)

        return payoff_loss, S_min_loss, S_max_loss, V_min_loss, V_max_loss


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
        T = kwargs.get('T', 1.0)

        v_1 = model(ones * T, S_boundary)
        payoff = self(S_boundary, K)
        exercise_loss = nn.MSELoss()(v_1, payoff)

        # Max boundary loss: if prod of assets greater than threshold, set to 0
        max_threshold = np.prod(S_max) * 0.8
        max_mask = torch.prod(S_boundary, dim=1) >= max_threshold
        if max_mask.any():
            v_max_boundary = model(t_boundary[max_mask], S_boundary[max_mask])
            boundary_max_loss = nn.MSELoss()(v_max_boundary, torch.zeros_like(v_max_boundary))
        else:
            boundary_max_loss = 0.0

        # f(t, S1, ..., SN) where any Si = Si_min = max(K - prod(S_min), 0)
        n_assets = S_boundary.shape[1]
        boundary_min_losses = []
        for i in range(n_assets):
            S_boundary_min = S_boundary.clone()
            S_boundary_min[:, i] = S_min[i]
            v_Si_min = model(t_boundary, S_boundary_min)
            boundary_loss_Si_min = nn.MSELoss()(v_Si_min, ones * torch.relu(K - torch.prod(S_boundary_min[0])))
            boundary_min_losses.append(boundary_loss_Si_min)

        # return exercise_loss, sum(boundary_max_losses), sum(boundary_min_losses)
        return exercise_loss, boundary_max_loss, sum(boundary_min_losses)

    def sobolev_loss(self, model, **kwargs):
        K = kwargs.get('K', None)
        a = kwargs.get('a', None)
        b = kwargs.get('b', None)

        S_interior = kwargs.get('S_interior', None)
        t1_interior = kwargs.get('t1_interior', None)
        t2_interior = kwargs.get('t2_interior', None)
        S1_boundary = kwargs.get('S1_boundary', None)
        S2_boundary = kwargs.get('S2_boundary', None)

        # Fractional exponents
        s_time_J3 = kwargs.get('s_time_J3', 0.75)
        s_space_J3 = kwargs.get('s_space_J3', 1.5)
        s_time_J4 = kwargs.get('s_time_J4', 0.25)
        s_space_J4 = kwargs.get('s_space_J4', 0.5)

        device = S_interior.device
        dtype = S_interior.dtype

        # Which index is set to a or b on the boundary
        S1_face = kwargs.get('S1_face', None)
        S2_face = kwargs.get('S2_face', None)

        if K is None or a is None or b is None or S_interior is None or t1_interior is None or t2_interior is None or S1_boundary is None or S2_boundary is None or S1_face is None or S2_face is None:
            raise ValueError("Missing required parameters for Sobolev loss calculation.")

        d = S_interior.shape[1]

        # --- J2: H^1 on Omega at t=0 ---
        S_interior.requires_grad_(True)
        t0 = torch.zeros((S_interior.shape[0], 1), device=device, dtype=dtype)
        v0 = model(t0, S_interior)
        g0 = self(S_interior, K)
        u0 = v0 - g0

        J2_L2 = torch.mean(u0 ** 2)
        grad_u0 = torch.autograd.grad(u0, S_interior, grad_outputs=torch.ones_like(u0), create_graph=True)[0]
        J2_grad = torch.mean(torch.sum(grad_u0 ** 2, dim=1, keepdim=True))

        J2 = J2_L2 + J2_grad

        # Boundary evaluations for J3 and J4
        t1 = t1_interior.to(device=device, dtype=dtype)
        t2 = t2_interior.to(device=device, dtype=dtype)

        # Evaluate model at required points
        v_t1_S1 = model(t1, S1_boundary)
        v_t1_S2 = model(t1, S2_boundary)
        v_t2_S1 = model(t2, S1_boundary)
        v_t2_S2 = model(t2, S2_boundary)

        g_S1 = self(S1_boundary, K)
        g_S2 = self(S2_boundary, K)

        d_t1_S1 = v_t1_S1 - g_S1
        d_t1_S2 = v_t1_S2 - g_S2
        d_t2_S1 = v_t2_S1 - g_S1
        d_t2_S2 = v_t2_S2 - g_S2

        # --- J3: mixed fractional norm on Sigma ---
        # Value term (L2 over Sigma) averaged across two boundary batches
        J3_val = 0.5 * (torch.mean(d_t1_S1 ** 2) + torch.mean(d_t1_S2 ** 2))

        # Time fractional: denom is |t1 - t2|^(1 + 2*s_time_J3)
        denom_time_J3 = (torch.abs(t1 - t2) ** (1 + 2 * s_time_J3)).clamp_min(1e-8)
        time_frac_S1 = torch.mean(((d_t1_S1 - d_t2_S1) ** 2) / denom_time_J3)
        time_frac_S2 = torch.mean(((d_t1_S2 - d_t2_S2) ** 2) / denom_time_J3)
        J3_time_frac = 0.5 * (time_frac_S1 + time_frac_S2)  # Average over two batches

        # Space fractional: denom exponent is 2 * (fractional s_space_J3) + d - 1
        frac_s_space_J3 = s_space_J3 - torch.floor(torch.tensor(s_space_J3))
        s_exponent_J3 = 2 * frac_s_space_J3 + d - 1
        diff_xy = S1_boundary - S2_boundary
        dist_xy = torch.norm(diff_xy, dim=1, keepdim=True).clamp_min(1e-8)
        spatial_frac = torch.mean(((d_t1_S1 - d_t1_S2) ** 2) / (dist_xy ** s_exponent_J3))

        J3 = J3_val + J3_time_frac + spatial_frac

        # --- J4: normal derivative norm on Sigma in H^{s_time_J4, s_space_J4} ---
        S1_boundary.requires_grad_(True)
        S2_boundary.requires_grad_(True)

        v_t1_S1 = model(t1, S1_boundary) - self(S1_boundary, K)
        v_t1_S2 = model(t1, S2_boundary) - self(S2_boundary, K)

        grad_d_S1 = torch.autograd.grad(
            v_t1_S1, S1_boundary, grad_outputs=torch.ones_like(v_t1_S1), create_graph=True
        )[0]
        grad_d_S2 = torch.autograd.grad(
            v_t1_S2, S2_boundary, grad_outputs=torch.ones_like(v_t1_S2), create_graph=True
        )[0]

        # Identify tangential derivative
        idx_rows = torch.arange(grad_d_S1.shape[0], device=device)
        dnu_S1 = grad_d_S1[idx_rows, S1_face].unsqueeze(1)
        dnu_S2 = grad_d_S2[idx_rows, S2_face].unsqueeze(1)

        # L2 part
        J4_L2 = torch.mean(dnu_S1 ** 2) + torch.mean(dnu_S2 ** 2)

        # Time fractional part
        # Compute gradients at t2 for time fractional difference
        v_t2_S1 = model(t2, S1_boundary) - self(S1_boundary, K)
        v_t2_S2 = model(t2, S2_boundary) - self(S2_boundary, K)

        grad_t2_S1 = torch.autograd.grad(
            v_t2_S1, S1_boundary, grad_outputs=torch.ones_like(v_t2_S1), create_graph=True
        )[0]
        grad_t2_S2 = torch.autograd.grad(
            v_t2_S2, S2_boundary, grad_outputs=torch.ones_like(v_t2_S2), create_graph=True
        )[0]

        dnu_t2_S1 = grad_t2_S1[idx_rows, S1_face].unsqueeze(1)
        dnu_t2_S2 = grad_t2_S2[idx_rows, S2_face].unsqueeze(1)

        denom_time_J4 = (torch.abs(t1 - t2) ** (1 + 2 * s_time_J4)).clamp_min(1e-8)
        J4_time = 0.5 * (
            torch.mean(((dnu_S1 - dnu_t2_S1) ** 2) / denom_time_J4) +
            torch.mean(((dnu_S2 - dnu_t2_S2) ** 2) / denom_time_J4)
        )

        # Spatial fractional part
        frac_s_space_J4 = s_space_J4 - torch.floor(torch.tensor(s_space_J4))
        s_exponent_J4 = 2 * frac_s_space_J4 + d - 1
        diff_xy = S1_boundary - S2_boundary
        dist_xy = torch.norm(diff_xy, dim=1, keepdim=True).clamp_min(1e-8)
        J4_space = torch.mean(((dnu_S1 - dnu_S2) ** 2) / (dist_xy ** s_exponent_J4))

        J4 = J4_L2 + J4_time + J4_space

        return J2, J3, J4

    def heston_loss(self, model, t, S, V, **kwargs):
        n_assets = S.shape[1]

        market_params = kwargs.get('market_params', None)
        if market_params is None:
            raise ValueError("market_params must be provided for Heston loss calculation.")

        K = market_params.K
        T = market_params.T

        S_list = [S[:, i].unsqueeze(1) for i in range(n_assets)]
        V_list = [V[:, i].unsqueeze(1) for i in range(n_assets)]

        ones = torch.ones_like(t)

        # J2
        payoff_loss = torch.mean((
            model(ones * T, *S_list, *V_list) - self(S, K)
        )**2)

        # J3
        S_min_loss = 0

        # J4
        S_max_loss = 0

        # J5
        V_min_loss = 0

        # J6
        V_max_loss = 0

        return payoff_loss, S_min_loss, S_max_loss, V_min_loss, V_max_loss
