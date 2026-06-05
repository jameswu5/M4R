"""Sobolev-regularised PINN for pricing an American put (product payoff) under
n-dimensional Black-Scholes."""

import torch
import torch.nn.functional as F

from utility.model import PINN


class BlackScholesSobolevMultiAsset(PINN):
    """Sobolev-regularised PINN for pricing a multi-asset American put (product payoff) under Black-Scholes."""

    def __init__(self, model_config, seed):
        super().__init__(model_config, seed)
        self.history = {
            'loss': [],
            'pde_loss': [],
            'J2_loss': [],
            'J3_loss': [],
            'J4_loss': [],
        }

    def set_params(self, K, r, sigmas, corr, T, S_mins, S_maxs):
        """
        Set the multi-asset Black-Scholes model and domain parameters.

        Parameters
        ----------
        K : float
            Strike price.
        r : float
            Risk-free rate (annualised).
        sigmas : array-like of float, length n_assets
            Per-asset volatilities (annualised).
        corr : array-like of float, shape (n_assets, n_assets)
            Correlation matrix between asset log-returns.
        T : float
            Time to expiry (in years).
        S_mins : array-like of float, length n_assets
            Lower spatial boundary for each asset price.
        S_maxs : array-like of float, length n_assets
            Upper spatial boundary for each asset price.
        """
        self.K = K
        self.r = r
        self.sigmas = torch.tensor(sigmas, dtype=torch.float32)  # (n_assets,)
        self.corr = torch.tensor(corr, dtype=torch.float32)      # (n_assets, n_assets)
        self.T = T
        self.S_mins = S_mins  # array of length n_assets
        self.S_maxs = S_maxs  # array of length n_assets
        self.n_assets = len(sigmas)

    def train(self, batch_size, epochs, early_stopping, anneal_freq=500, alpha=0.9):
        """
        Train using Sobolev regularity losses J2, J3, J4 with automatic loss reweighting.

        Parameters
        ----------
        batch_size : int
            Collocation points per training batch.
        epochs : int
            Maximum training epochs.
        early_stopping : EarlyStopping
            Halts training and restores best model when validation loss stagnates.
        anneal_freq : int, optional
            Epochs between loss weight updates (default 500).
        alpha : float, optional
            EMA smoothing factor for loss weight updates (default 0.9).
        """
        # Respect weights set via set_loss_weights(); otherwise default to equal.
        if not getattr(self, 'loss_weights', None):
            self.set_loss_weights({'pde': 1.0, 'J2': 1.0, 'J3': 1.0, 'J4': 1.0})

        # Fixed validation batches
        val_t1, val_t2 = self.__sample_time_pairs(batch_size)
        val_S = self.__sample_S_interior(batch_size)
        val_S1_bnd, val_S2_bnd, val_face1, val_face2 = self.__sample_S_boundary_pairs(batch_size)

        for i in range(epochs):
            t1, t2 = self.__sample_time_pairs(batch_size)
            S = self.__sample_S_interior(batch_size)
            S1_bnd, S2_bnd, face1, face2 = self.__sample_S_boundary_pairs(batch_size)

            pde_loss = self.__pde_loss(t1, S)
            J2, J3, J4 = self.__sobolev_loss(t1, t2, S, S1_bnd, S2_bnd, face1, face2)

            if i > 2000 and i % anneal_freq == 0:
                self.__anneal_weights({'pde': pde_loss, 'J2': J2, 'J3': J3, 'J4': J4}, alpha)

            loss = self.__process_loss(pde_loss, J2, J3, J4)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Validation
            val_pde = self.__pde_loss(val_t1, val_S)
            val_J2, _, _ = self.__sobolev_loss(
                val_t1, val_t2, val_S, val_S1_bnd, val_S2_bnd, val_face1, val_face2
            )
            val_loss = val_pde + val_J2

            if i % 500 == 0:
                weight_str = "  ".join(f"{k}={v:.3f}" for k, v in self.loss_weights.items())
                print(f"Iter {i:>6} | Train: {loss.item():.4e} "
                      f"| Val: {val_loss.item():.4e} | Weights: {weight_str}")

            if early_stopping and early_stopping.step(val_loss.item(), self.model):
                print(f"Early stopping at epoch {i}")
                early_stopping.restore(self.model)
                break

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def __process_loss(self, pde_loss, J2, J3, J4, update_dict=True):
        """Apply loss weights, sum, and optionally append to history."""
        w_pde = self.loss_weights['pde'] * pde_loss
        w_J2  = self.loss_weights['J2']  * J2
        w_J3  = self.loss_weights['J3']  * J3
        w_J4  = self.loss_weights['J4']  * J4
        loss = w_pde + w_J2 + w_J3 + w_J4
        if update_dict:
            self.history['loss'].append(loss.item())
            self.history['pde_loss'].append(w_pde.item())
            self.history['J2_loss'].append(w_J2.item())
            self.history['J3_loss'].append(w_J3.item())
            self.history['J4_loss'].append(w_J4.item())
        return loss

    def __anneal_weights(self, unweighted_losses: dict, alpha: float, frozen=('J4',)):
        """Rescale loss weights by gradient-norm ratios, apply EMA, and renormalise.

        Weights named in `frozen` are held fixed and excluded from the
        gradient-norm balancing. Their (noisy) gradients therefore cannot drive
        the weight update, and the remaining terms share the leftover budget
        (1 - sum of frozen weights). This stops the balancer from pumping weight
        into a noise-dominated term such as J4.
        """
        params = list(self.model.parameters())
        total_loss = sum(self.loss_weights[k] * v for k, v in unweighted_losses.items())
        total_grads = torch.autograd.grad(
            total_loss, params, retain_graph=True, create_graph=True, allow_unused=True
        )
        peak_grad = max(g.abs().max().item() for g in total_grads if g is not None)

        balanced = [k for k in unweighted_losses if k not in frozen]

        new_weights = {}
        for name in balanced:
            loss_term = unweighted_losses[name]
            grads = torch.autograd.grad(
                self.loss_weights[name] * loss_term, params,
                retain_graph=True, create_graph=True, allow_unused=True
            )
            grad_tensors = [g for g in grads if g is not None]
            mean_grad = (
                sum(g.abs().mean().item() for g in grad_tensors) / len(grad_tensors)
            ) if grad_tensors else 1.0
            lambda_hat = peak_grad / (mean_grad + 1e-8)
            new_weights[name] = alpha * self.loss_weights[name] + (1.0 - alpha) * lambda_hat

        # Hold frozen weights fixed; let the balanced terms share the rest.
        frozen_mass = sum(self.loss_weights[k] for k in unweighted_losses if k in frozen)
        balanced_total = sum(new_weights.values())
        if balanced_total > 0:
            scale = (1.0 - frozen_mass) / balanced_total
            for k in balanced:
                new_weights[k] *= scale
        for k in unweighted_losses:
            if k in frozen:
                new_weights[k] = self.loss_weights[k]

        self.loss_weights = new_weights

    def __sample_time_pairs(self, batch_size):
        """Sample pairs of time coordinates separated by at least `eps` from [0, T].

        `eps` bounds the fractional-seminorm denominators |t1 - t2|^(1+2s) away
        from zero; raise it if the time Gagliardo terms still spike.
        """
        eps = 0.05
        t1, t2, _, _ = self.sampler.uniform_pair(
            0, self.T, batch_size, 1, epsilon=eps, boundary=False
        )
        return t1, t2

    def __sample_S_interior(self, batch_size):
        """Sample asset prices uniformly from the interior domain."""
        return self.sampler.uniform(self.S_mins, self.S_maxs, (batch_size, self.n_assets))

    def __sample_S_boundary_pairs(self, batch_size):
        """Sample pairs of boundary asset-price points (both on the SAME face) with
        their shared face index, for a well-defined normal-derivative comparison.

        `eps` bounds the spatial Gagliardo denominator ||S1 - S2||^(2*theta+n-1)
        away from zero (the distance is now measured within the shared face);
        raise it if the spatial terms still spike.
        """
        eps = 0.05
        S1, S2, face1, face2 = self.sampler.uniform_pair(
            self.S_mins, self.S_maxs, batch_size, self.n_assets,
            epsilon=eps, boundary=True
        )
        return S1, S2, face1, face2

    def __bs_residual(self, t, S):
        """Evaluate the multi-asset Black-Scholes PDE residual at (t, S) via automatic differentiation."""
        batch_size = S.shape[0]
        v = self.model(t, S)

        v_t, v_S = torch.autograd.grad(
            v, (t, S), grad_outputs=torch.ones_like(v), create_graph=True
        )

        rows = []
        for i in range(self.n_assets):
            v_Si = v_S[:, i].unsqueeze(-1)
            v_Si_S = torch.autograd.grad(
                v_Si, S, grad_outputs=torch.ones_like(v_Si), create_graph=True
            )[0]
            rows.append(v_Si_S.unsqueeze(1))   # (batch, 1, n_assets)
        v_SS = torch.cat(rows, dim=1)           # (batch, n_assets, n_assets)

        cov = torch.outer(self.sigmas, self.sigmas) * self.corr  # (n_assets, n_assets)

        drift = self.r * torch.sum(S * v_S, dim=1, keepdim=True)

        S_outer = S.unsqueeze(2) * S.unsqueeze(1)   # (batch, n_assets, n_assets)
        cov_bc = cov.unsqueeze(0).expand(batch_size, -1, -1)
        diffusion = 0.5 * torch.sum(cov_bc * S_outer * v_SS, dim=(1, 2)).unsqueeze(-1)

        # Backward (calendar-time) Black-Scholes operator: -(v_t + A[v] - r v),
        # consistent with the payoff condition being imposed at t = T (see J2 below).
        residual = -v_t - drift - diffusion + self.r * v
        return residual, v

    def __pde_loss(self, t, S):
        """American put complementarity loss min(L[v], v-g)^2 with product payoff at interior points."""
        t = t.detach().requires_grad_(True)
        S = S.detach().requires_grad_(True)

        residual, v = self.__bs_residual(t, S)
        g = F.relu(self.K - torch.prod(S, dim=1, keepdim=True))

        complementarity = torch.min(residual, v - g)
        return torch.mean(complementarity ** 2)

    def __sobolev_loss(self, t1, t2, S_interior, S1_boundary, S2_boundary,
                       face1, face2,
                       s_time_J3=0.75, s_space_J3=1.5,
                       s_time_J4=0.25, s_space_J4=0.5):
        """Sobolev regularity losses J2 (H^1 payoff), J3 (mixed lateral), J4 (normal derivative lateral)."""
        d = self.n_assets
        device = S_interior.device

        # ------------------------------------------------------------------
        # J2: H^1 norm of u(T, S) = v(T, S) - g(S) on the interior
        #     (payoff condition imposed at t = T)
        # ------------------------------------------------------------------
        S_interior = S_interior.detach().requires_grad_(True)
        batch = S_interior.shape[0]
        t_T = torch.ones(batch, 1) * self.T

        v_T = self.model(t_T, S_interior)
        g_T = F.relu(self.K - torch.prod(S_interior, dim=1, keepdim=True))
        u_T = v_T - g_T

        J2_L2 = torch.mean(u_T ** 2)
        grad_u_T = torch.autograd.grad(
            u_T, S_interior, grad_outputs=torch.ones_like(u_T), create_graph=True
        )[0]
        J2_H1 = torch.mean(torch.sum(grad_u_T ** 2, dim=1, keepdim=True))
        J2 = J2_L2 + J2_H1

        # ------------------------------------------------------------------
        # Boundary evaluations (shared between J3 and J4)
        # requires_grad=True needed for the normal-derivative computation in J4
        # ------------------------------------------------------------------
        S1_boundary = S1_boundary.detach().requires_grad_(True)
        S2_boundary = S2_boundary.detach().requires_grad_(True)

        v_t1_S1 = self.model(t1, S1_boundary)
        v_t1_S2 = self.model(t1, S2_boundary)
        v_t2_S1 = self.model(t2, S1_boundary)
        v_t2_S2 = self.model(t2, S2_boundary)

        g_S1 = F.relu(self.K - torch.prod(S1_boundary, dim=1, keepdim=True))
        g_S2 = F.relu(self.K - torch.prod(S2_boundary, dim=1, keepdim=True))

        d_t1_S1 = v_t1_S1 - g_S1   # u(t1, S1)
        d_t1_S2 = v_t1_S2 - g_S2   # u(t1, S2)
        d_t2_S1 = v_t2_S1 - g_S1   # u(t2, S1)
        d_t2_S2 = v_t2_S2 - g_S2   # u(t2, S2)

        dt = torch.abs(t1 - t2)

        # ------------------------------------------------------------------
        # J3: H^{0,1} term (L^2 value + spatial-gradient L^2) + time Gagliardo
        #     seminorm + space Gagliardo seminorm on the lateral boundary
        # ------------------------------------------------------------------
        J3_val = 0.5 * (torch.mean(d_t1_S1 ** 2) + torch.mean(d_t1_S2 ** 2))

        denom_time_J3 = (dt ** (1 + 2 * s_time_J3)).clamp_min(1e-8)
        time_frac_S1 = torch.mean(((d_t1_S1 - d_t2_S1) ** 2) / denom_time_J3)
        time_frac_S2 = torch.mean(((d_t1_S2 - d_t2_S2) ** 2) / denom_time_J3)
        J3_time = 0.5 * (time_frac_S1 + time_frac_S2)

        # Gradient of u at boundary points (needed for the H^{0,1} spatial term
        # and reused for the J4 normal derivative)
        grad_u_S1 = torch.autograd.grad(
            d_t1_S1, S1_boundary, grad_outputs=torch.ones_like(d_t1_S1), create_graph=True
        )[0]
        grad_u_S2 = torch.autograd.grad(
            d_t1_S2, S2_boundary, grad_outputs=torch.ones_like(d_t1_S2), create_graph=True
        )[0]

        # H^{0,1} contribution: L^2 norm of the spatial gradients on the boundary
        J3_grad_L2 = 0.5 * (
            torch.mean(torch.sum(grad_u_S1 ** 2, dim=1, keepdim=True)) +
            torch.mean(torch.sum(grad_u_S2 ** 2, dim=1, keepdim=True))
        )

        # Spatial Gagliardo seminorm on the function value (writeup form):
        # |u(t1, S1) - u(t1, S2)|^2 / |S1 - S2|^(2*theta + n - 1)
        frac_space_J3 = s_space_J3 % 1
        s_exp_J3 = 2 * frac_space_J3 + d - 1
        diff_xy = S1_boundary - S2_boundary
        dist_xy = torch.norm(diff_xy, dim=1, keepdim=True).clamp_min(1e-8)
        J3_space = torch.mean(
            ((d_t1_S1 - d_t1_S2) ** 2) / (dist_xy ** s_exp_J3)
        )

        J3 = J3_val + J3_grad_L2 + J3_time + J3_space

        # ------------------------------------------------------------------
        # J4: outward normal derivative dnu = du/dS_{face_i} on each face
        #     L^2 term + time Gagliardo + spatial Gagliardo
        # ------------------------------------------------------------------
        # Reuse grad_u_S1 / grad_u_S2 already computed for J3
        idx = torch.arange(S1_boundary.shape[0], device=device)
        face1_t = torch.tensor(face1, dtype=torch.long, device=device)
        face2_t = torch.tensor(face2, dtype=torch.long, device=device)

        dnu_t1_S1 = grad_u_S1[idx, face1_t].unsqueeze(1)
        dnu_t1_S2 = grad_u_S2[idx, face2_t].unsqueeze(1)

        # L^2 term: single ||d||^2_{L^2(Sigma)} estimated by averaging the
        # two boundary-sample batches (matches the J3 value-term convention)
        J4_L2 = 0.5 * (torch.mean(dnu_t1_S1 ** 2) + torch.mean(dnu_t1_S2 ** 2))

        # Time fractional part: compare normal derivatives at t1 vs t2
        grad_d_t2_S1 = torch.autograd.grad(
            d_t2_S1, S1_boundary, grad_outputs=torch.ones_like(d_t2_S1), create_graph=True
        )[0]
        grad_d_t2_S2 = torch.autograd.grad(
            d_t2_S2, S2_boundary, grad_outputs=torch.ones_like(d_t2_S2), create_graph=True
        )[0]

        dnu_t2_S1 = grad_d_t2_S1[idx, face1_t].unsqueeze(1)
        dnu_t2_S2 = grad_d_t2_S2[idx, face2_t].unsqueeze(1)

        denom_time_J4 = (dt ** (1 + 2 * s_time_J4)).clamp_min(1e-8)
        J4_time = 0.5 * (
            torch.mean(((dnu_t1_S1 - dnu_t2_S1) ** 2) / denom_time_J4) +
            torch.mean(((dnu_t1_S2 - dnu_t2_S2) ** 2) / denom_time_J4)
        )

        # Spatial fractional part: compare normal derivatives at S1 vs S2
        frac_space_J4 = s_space_J4 % 1
        s_exp_J4 = 2 * frac_space_J4 + d - 1
        J4_space = torch.mean(((dnu_t1_S1 - dnu_t1_S2) ** 2) / (dist_xy ** s_exp_J4))

        J4 = J4_L2 + J4_time + J4_space

        return J2, J3, J4
