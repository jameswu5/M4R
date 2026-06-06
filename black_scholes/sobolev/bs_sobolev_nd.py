"""Sobolev-regularised PINN for pricing an American put (product payoff) under
n-dimensional Black-Scholes."""

import numpy as np
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

    def train(self, batch_size, epochs, early_stopping, anneal_freq=500, alpha=0.9,
              grad_clip=1.0, lbfgs_steps=500, causal_eps=2.0, fb_frac=0.65):
        """
        Train using Sobolev regularity losses J2, J3, J4 with automatic loss reweighting.

        Adam optimises for `epochs`, then (optionally) an LBFGS phase refines the
        same 4-term loss. Adam plateaus on PINN residuals around 1e-2; LBFGS drives
        the continuation-region complementarity residual down by 1-2 orders of
        magnitude, which is what removes the accumulated t=0 overpricing.

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
        grad_clip : float or None, optional
            Max global gradient norm. The Gagliardo seminorms J3/J4 are heavy-tailed
            MC estimators that occasionally produce huge gradients near the free
            boundary; clipping bounds their effect on a single step (default 1.0).
            Pass None to disable.
        lbfgs_steps : int, optional
            Total LBFGS iteration budget for the post-Adam refinement phase
            (default 500). Set 0 to skip. Run in short rounds on freshly resampled
            collocation batches to avoid overfitting a single batch.
        causal_eps : float, optional
            Causality tolerance for time weighting of the PDE residual (default 2.0).
            The terminal condition anchors t=T; this weights each point's residual by
            exp(-causal_eps * accumulated residual over later times, i.e. closer to T),
            so the residual is enforced near t=T first and propagated backward instead
            of being averaged uniformly (which lets t=0 error accumulate). Set 0 to
            disable (plain mean).
        fb_frac : float, optional
            Fraction of PDE collocation points drawn near the free boundary
            prod(S) = K (default 0.65), where the residual/curvature is largest. The
            remaining points stay uniform for domain coverage. J2/J3/J4 sampling is
            unchanged. Set 0 to disable.
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
            S_pde = self.__sample_S_interior(batch_size, fb_frac=fb_frac)
            S1_bnd, S2_bnd, face1, face2 = self.__sample_S_boundary_pairs(batch_size)

            pde_loss = self.__pde_loss(t1, S_pde, causal_eps=causal_eps)
            J2, J3, J4 = self.__sobolev_loss(t1, t2, S, S1_bnd, S2_bnd, face1, face2)

            if i > 2000 and i % anneal_freq == 0:
                self.__anneal_weights({'pde': pde_loss, 'J2': J2, 'J3': J3, 'J4': J4}, alpha)

            loss = self.__process_loss(pde_loss, J2, J3, J4)

            self.optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
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

        if lbfgs_steps:
            self.__lbfgs_refine(batch_size, lbfgs_steps, fb_frac=fb_frac)

    def __lbfgs_refine(self, batch_size, total_steps, inner_iter=25, fb_frac=0.0):
        """Second-order refinement of the 4-term loss after Adam.

        Runs LBFGS in short rounds, resampling the collocation batch each round so
        the optimiser cannot overfit a single fixed batch. Loss weights are held at
        their final annealed values; no new loss terms are introduced. Causal time
        weighting is left off here so the closure loss is stationary within a round's
        line search; free-boundary importance sampling (fb_frac) is still applied.
        """
        rounds = max(1, total_steps // inner_iter)
        print(f"LBFGS refinement: {rounds} rounds x {inner_iter} iters...")

        optimizer = torch.optim.LBFGS(
            self.model.parameters(), max_iter=inner_iter, history_size=50,
            line_search_fn='strong_wolfe', tolerance_grad=1e-9, tolerance_change=1e-12,
        )

        for rnd in range(rounds):
            t1, t2 = self.__sample_time_pairs(batch_size)
            S = self.__sample_S_interior(batch_size)
            S_pde = self.__sample_S_interior(batch_size, fb_frac=fb_frac)
            S1_bnd, S2_bnd, face1, face2 = self.__sample_S_boundary_pairs(batch_size)

            def closure():
                optimizer.zero_grad()
                pde_loss = self.__pde_loss(t1, S_pde)
                J2, J3, J4 = self.__sobolev_loss(t1, t2, S, S1_bnd, S2_bnd, face1, face2)
                loss = self.__process_loss(pde_loss, J2, J3, J4, update_dict=False)
                loss.backward()
                return loss

            optimizer.step(closure)

            # Record one point per round so the loss history stays continuous.
            pde_loss = self.__pde_loss(t1, S_pde)
            J2, J3, J4 = self.__sobolev_loss(t1, t2, S, S1_bnd, S2_bnd, face1, face2)
            loss = self.__process_loss(pde_loss, J2, J3, J4)
            if rnd % max(1, rounds // 10) == 0:
                print(f"  LBFGS round {rnd:>4} | Loss: {loss.item():.4e}")

    def __process_loss(self, pde_loss, J2, J3, J4, update_dict=True):
        """Apply loss weights, sum, and optionally append to history."""
        w_pde = self.loss_weights['pde'] * pde_loss
        w_J2  = self.loss_weights['J2']  * J2
        w_J3  = self.loss_weights['J3']  * J3
        w_J4  = self.loss_weights['J4']  * J4
        loss = w_pde + w_J2 + w_J3 + w_J4
        if update_dict:
            self.history['loss'].append(loss.item())
            self.history['pde_loss'].append(pde_loss.item())
            self.history['J2_loss'].append(J2.item())
            self.history['J3_loss'].append(J3.item())
            self.history['J4_loss'].append(J4.item())
        return loss

    @staticmethod
    def __robust_mean(ratio, q):
        """Winsorised mean of a heavy-tailed per-sample fractional ratio.

        Caps each contribution at the q-quantile (computed on detached values)
        before averaging, so the handful of near-singular Gagliardo pairs at the
        free boundary cannot dominate the seminorm estimate or its gradient.
        q=None (or q>=1) disables clipping and recovers the plain mean.
        """
        if q is not None and q < 1.0:
            cap = torch.quantile(ratio.detach(), q)
            ratio = ratio.clamp(max=cap)
        return ratio.mean()

    def __anneal_weights(self, unweighted_losses: dict, alpha: float, frozen=('pde', 'J4')):
        """Rescale loss weights by gradient-norm ratios, apply EMA, and renormalise.

        Weights named in `frozen` are held fixed and excluded from the
        gradient-norm balancing. Their gradients therefore cannot drive the
        weight update, and the remaining terms share the leftover budget
        (1 - sum of frozen weights).

        Both `pde` and `J4` are frozen by default. J4 because it is noise-
        dominated; `pde` because the continuation-region price *level* is governed
        almost entirely by the complementarity residual, and gradient-norm
        balancing otherwise dilutes it against the terminal/regularity terms,
        leaving the residual under-converged (the cause of the t=0 overpricing).
        Freezing `pde` keeps its authority at the user-set weight throughout.
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
        eps = 0.1
        t1, t2, _, _ = self.sampler.uniform_pair(
            0, self.T, batch_size, 1, epsilon=eps, boundary=False
        )
        return t1, t2

    def __sample_S_interior(self, batch_size, fb_frac=0.0):
        """Sample asset prices from the interior domain.

        With fb_frac>0, the first int(batch_size*fb_frac) rows are drawn near the
        free-boundary manifold prod(S) = K (where the obstacle constraint switches
        and the complementarity residual/curvature is largest), and the rest stay
        uniform for domain coverage. fb_frac=0 recovers a fully uniform batch.
        """
        S = self.sampler.uniform(self.S_mins, self.S_maxs, (batch_size, self.n_assets))
        n_fb = int(batch_size * fb_frac)
        if n_fb > 0:
            S[:n_fb] = self.__sample_near_free_boundary(n_fb)
        return S

    def __sample_near_free_boundary(self, n, width=0.2):
        """Sample n interior points clustered near the manifold prod(S) = K.

        Draw a base point uniformly, pick a target product spread lognormally
        around K (multiplicative width), then rescale every coordinate by
        (target / prod)^(1/n_assets) so the product hits the target while the
        point stays in the interior. Coordinates are clamped back into the domain,
        which slightly perturbs the product but keeps the cluster tight.
        """
        S_mins = torch.as_tensor(self.S_mins, dtype=torch.float32)
        S_maxs = torch.as_tensor(self.S_maxs, dtype=torch.float32)
        lo = torch.clamp(S_mins, min=1e-3)

        base = lo + (S_maxs - lo) * torch.rand(n, self.n_assets)
        prod = torch.prod(base, dim=1, keepdim=True).clamp_min(1e-8)

        log_target = np.log(self.K) + width * torch.randn(n, 1)
        target = torch.exp(log_target)

        scale = (target / prod) ** (1.0 / self.n_assets)
        S = base * scale
        return torch.clamp(S, min=lo, max=S_maxs)

    def __sample_S_boundary_pairs(self, batch_size):
        """Sample pairs of boundary asset-price points (both on the SAME face) with
        their shared face index, for a well-defined normal-derivative comparison.

        `eps` bounds the spatial Gagliardo denominator ||S1 - S2||^(2*theta+n-1)
        away from zero (the distance is now measured within the shared face);
        raise it if the spatial terms still spike.
        """
        eps = 0.1
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

    def __pde_loss(self, t, S, causal_eps=0.0, n_bins=16):
        """American put complementarity loss min(L[v], v-g)^2 with product payoff at interior points.

        With causal_eps>0 the per-point squared residual is weighted so that points
        near the terminal anchor t=T are enforced before earlier ones (see
        __causal_weighted); causal_eps=0 averages uniformly.
        """
        t = t.detach().requires_grad_(True)
        S = S.detach().requires_grad_(True)

        residual, v = self.__bs_residual(t, S)
        g = F.relu(self.K - torch.prod(S, dim=1, keepdim=True))

        complementarity = torch.min(residual, v - g)
        sq = complementarity ** 2
        if causal_eps > 0:
            return self.__causal_weighted(sq, t, causal_eps, n_bins)
        return torch.mean(sq)

    def __causal_weighted(self, sq, t, eps, n_bins):
        """Causal (terminal-anchored) time weighting of the per-point residual.

        The terminal payoff is imposed at t=T, so the solution is trustworthy near
        T and propagates backward. Bin points by time, and weight each bin by
        exp(-eps * cumulative residual over all *later* bins, i.e. those nearer T).
        A bin is down-weighted until the residual at every later time has fallen,
        which enforces the residual front-to-back instead of letting t=0 error
        accumulate under a uniform average. Weights are detached (they gate the
        loss, they are not differentiated through).
        """
        t_flat = t.detach().reshape(-1)
        sq_flat = sq.reshape(-1)

        edges = torch.linspace(0.0, self.T, n_bins + 1, device=t_flat.device)[1:-1]
        bin_idx = torch.bucketize(t_flat, edges)

        bin_sum = torch.zeros(n_bins, device=t_flat.device)
        bin_cnt = torch.zeros(n_bins, device=t_flat.device)
        bin_sum.scatter_add_(0, bin_idx, sq_flat.detach())
        bin_cnt.scatter_add_(0, bin_idx, torch.ones_like(sq_flat))
        bin_mean = bin_sum / bin_cnt.clamp_min(1.0)

        # Exclusive cumulative residual over later (nearer-T) bins, going backward.
        rev = torch.flip(bin_mean, dims=[0])
        cum_rev = torch.cumsum(rev, dim=0) - rev          # exclusive
        cum = torch.flip(cum_rev, dims=[0])
        bin_w = torch.exp(-eps * cum)

        w = bin_w[bin_idx]
        return torch.sum(w * sq_flat) / w.sum().clamp_min(1e-8)

    def __sobolev_loss(self, t1, t2, S_interior, S1_boundary, S2_boundary,
                       face1, face2,
                       s_time_J3=0.75, s_space_J3=1.5,
                       s_time_J4=0.25, s_space_J4=0.5,
                       robust_q=0.99):
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
        time_frac_S1 = self.__robust_mean(((d_t1_S1 - d_t2_S1) ** 2) / denom_time_J3, robust_q)
        time_frac_S2 = self.__robust_mean(((d_t1_S2 - d_t2_S2) ** 2) / denom_time_J3, robust_q)
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
        J3_space = self.__robust_mean(
            ((d_t1_S1 - d_t1_S2) ** 2) / (dist_xy ** s_exp_J3), robust_q
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
            self.__robust_mean(((dnu_t1_S1 - dnu_t2_S1) ** 2) / denom_time_J4, robust_q) +
            self.__robust_mean(((dnu_t1_S2 - dnu_t2_S2) ** 2) / denom_time_J4, robust_q)
        )

        # Spatial fractional part: compare normal derivatives at S1 vs S2
        frac_space_J4 = s_space_J4 % 1
        s_exp_J4 = 2 * frac_space_J4 + d - 1
        J4_space = self.__robust_mean(((dnu_t1_S1 - dnu_t1_S2) ** 2) / (dist_xy ** s_exp_J4), robust_q)

        J4 = J4_L2 + J4_time + J4_space

        return J2, J3, J4
