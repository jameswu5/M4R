"""Sobolev-regularised network for pricing an American put under Black-Scholes."""

import torch
import torch.nn.functional as F

from utility.model import PINN


class BlackScholesSobolev(PINN):
    def __init__(self, model_config, seed):
        super().__init__(model_config, seed)
        self.history = {
            'loss': [],
            'pde_loss': [],
            'J2_loss': [],
            'J3_loss': [],
            'J4_loss': [],
        }

    def set_params(self, K, r, sigma, T, S_min, S_max):
        """
        Set the Black-Scholes model parameters.

        Parameters
        ----------
        K : float
            Strike price of the option.
        r : float
            Risk-free interest rate (annualised).
        sigma : float
            Volatility of the underlying asset (annualised).
        T : float
            Time to expiry (in years).
        S_min : float
            Minimum asset price used as the lower spatial boundary.
        S_max : float
            Maximum asset price used as the upper spatial boundary.
        """
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.S_min = S_min
        self.S_max = S_max

    def train(self, batch_size, epochs, early_stopping, anneal_freq=500, alpha=0.9):
        """
        Train with Sobolev regularity losses J2, J3, J4 and automatic loss
        reweighting via learning rate annealing.

        Parameters
        ----------
        batch_size : int
            Number of samples per training batch.
        epochs : int
            Maximum number of training epochs.
        early_stopping : EarlyStopping
            Early stopping callback; training halts and the best model is
            restored when the validation loss stops improving.
        anneal_freq : int, optional
            Number of epochs between loss weight updates (default 500).
        alpha : float, optional
            EMA smoothing factor for loss reweighting (default 0.9).
        """
        self.loss_weights = {
            'pde': 0.25,
            'J2':  0.25,
            'J3':  0.25,
            'J4':  0.25,
        }

        # Fixed validation batches for early stopping
        val_t1, val_t2 = self.__sample_time_pairs(batch_size)
        val_S = self.__sample_S(batch_size)

        for i in range(epochs):
            t1, t2 = self.__sample_time_pairs(batch_size)
            S = self.__sample_S(batch_size)

            pde_loss = self.__pde_loss(t1, S)
            J2, J3, J4 = self.__sobolev_loss(t1, t2, S)

            if i > 2000 and i % anneal_freq == 0:
                unweighted = {'pde': pde_loss, 'J2': J2, 'J3': J3, 'J4': J4}
                self.__anneal_weights(unweighted, alpha)

            loss = self.__process_loss(pde_loss, J2, J3, J4)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Validation loss (unweighted sum for early stopping)
            val_pde = self.__pde_loss(val_t1, val_S)
            val_J2, val_J3, val_J4 = self.__sobolev_loss(val_t1, val_t2, val_S)
            val_loss = val_pde + val_J2 + val_J3 + val_J4

            if i % 500 == 0:
                weight_str = "  ".join(
                    f"{k}={v:.3f}" for k, v in self.loss_weights.items())
                print(f"Iter {i:>6} | Train: {loss.item():.4e} "
                      f"| Val: {val_loss.item():.4e} | Weights: {weight_str}")

            if early_stopping and early_stopping.step(val_loss.item(), self.model):
                print(f"Early stopping at epoch {i}")
                early_stopping.restore(self.model)
                break

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def __process_loss(self, pde_loss, J2, J3, J4, update_dict=True):
        """
        Apply loss weights, sum the components, and optionally record to history.

        Parameters
        ----------
        pde_loss : torch.Tensor
            Interior PDE complementarity loss.
        J2 : torch.Tensor
            H^1 terminal boundary loss.
        J3 : torch.Tensor
            H^{3/4} fractional-in-time lateral boundary loss.
        J4 : torch.Tensor
            H^{1/4} fractional-in-time normal-derivative lateral boundary loss.
        update_dict : bool, optional
            If True (default), append each weighted loss to ``self.history``.

        Returns
        -------
        loss : torch.Tensor
            Scalar weighted total loss.
        """
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

    def __anneal_weights(self, unweighted_losses: dict, alpha: float):
        """
        Update loss weights using gradient-based learning rate annealing.

        Uses the PDE residual gradient as the reference peak (Wang et al. 2021)
        rather than the total loss gradient, since rescaling all terms relative
        to the total would produce equal ``lambda_hat`` values and leave the
        weights unchanged after normalisation.

        Parameters
        ----------
        unweighted_losses : dict
            Mapping from loss name to its unweighted scalar ``torch.Tensor``.
            Keys must match those in ``self.loss_weights``.
        alpha : float
            EMA smoothing factor in ``[0, 1)``.  Higher values retain more of
            the current weight; lower values adapt more aggressively.
        """
        params = list(self.model.parameters())
        # Use the PDE residual loss gradient as the reference (Wang et al. 2021).
        # Using the total loss here makes all lambda_hat values equal after
        # normalisation, so the annealing would have no effect.
        pde_grads = torch.autograd.grad(
            unweighted_losses['pde'], params, retain_graph=True, create_graph=False, allow_unused=True
        )
        peak_grad = max(g.abs().max().item() for g in pde_grads if g is not None)

        new_weights = {}
        for name, loss in unweighted_losses.items():
            weighted_loss = self.loss_weights[name] * loss
            grads = torch.autograd.grad(
                weighted_loss, params, retain_graph=True, create_graph=False, allow_unused=True
            )
            grad_tensors = [g for g in grads if g is not None]
            mean_grad = (
                sum(g.abs().mean().item() for g in grad_tensors) / len(grad_tensors)
            ) if grad_tensors else 1.0
            lambda_hat = peak_grad / (mean_grad + 1e-8)
            new_weights[name] = alpha * self.loss_weights[name] + (1.0 - alpha) * lambda_hat

        total = sum(new_weights.values())
        self.loss_weights = {k: v / total for k, v in new_weights.items()}

    def __sample_time_pairs(self, batch_size):
        """
        Sample pairs of distinct time coordinates from the interior of ``[0, T]``.

        Parameters
        ----------
        batch_size : int
            Number of pairs to sample.

        Returns
        -------
        t1 : torch.Tensor, shape (batch_size, 1)
            First time coordinate of each pair.
        t2 : torch.Tensor, shape (batch_size, 1)
            Second time coordinate of each pair, guaranteed to differ from
            ``t1`` by at least ``0.01``.
        """
        t1, t2, _, _ = self.sampler.uniform_pair(
            0, self.T, batch_size, 1, epsilon=0.01, boundary=False
        )
        return t1, t2

    def __sample_S(self, batch_size):
        """
        Sample asset-price coordinates uniformly from ``[S_min, S_max]``.

        Parameters
        ----------
        batch_size : int
            Number of points to sample.

        Returns
        -------
        S : torch.Tensor, shape (batch_size, 1)
            Asset prices sampled from the spatial domain.
        """
        return self.sampler.uniform(self.S_min, self.S_max, (batch_size, 1))

    def __bs_residual(self, t, S):
        """
        Evaluate the Black-Scholes PDE residual at given collocation points.

        Computes ``L[f] = -f_t - r S f_S - 0.5 sigma^2 S^2 f_SS + r f`` via
        automatic differentiation.  For the European BS PDE,
        ``f_t + r S f_S + 0.5 sigma^2 S^2 f_SS - r f = 0``, so ``L[f] = 0``
        under the forward-time convention (``t = 0`` today, ``t = T`` at expiry).

        Parameters
        ----------
        t : torch.Tensor, shape (N, 1)
            Time coordinates.  Must have ``requires_grad=True``.
        S : torch.Tensor, shape (N, 1)
            Asset-price coordinates.  Must have ``requires_grad=True``.

        Returns
        -------
        residual : torch.Tensor, shape (N, 1)
            PDE residual at each collocation point.
        f : torch.Tensor, shape (N, 1)
            Network output (option price) at each collocation point.
        """
        f = self.model(t, S)
        f_t, f_S = torch.autograd.grad(
            f, (t, S), grad_outputs=torch.ones_like(f), create_graph=True
        )
        f_SS = torch.autograd.grad(
            f_S, S, grad_outputs=torch.ones_like(f_S), create_graph=True
        )[0]
        residual = (
            -f_t
            - self.r * S * f_S
            - 0.5 * self.sigma ** 2 * S ** 2 * f_SS
            + self.r * f
        )
        return residual, f

    def __pde_loss(self, t, S):
        """
        Compute the American put complementarity loss.

        Enforces ``min(L[f], f - g) = 0`` in the least-squares sense, where
        ``L[f]`` is the Black-Scholes residual and ``g = max(K - S, 0)`` is
        the put's intrinsic value.

        Parameters
        ----------
        t : torch.Tensor, shape (N, 1)
            Time coordinates (detached; ``requires_grad`` is set internally).
        S : torch.Tensor, shape (N, 1)
            Asset-price coordinates (detached; ``requires_grad`` is set internally).

        Returns
        -------
        pde_loss : torch.Tensor
            Scalar mean-squared complementarity loss.
        """
        t = t.detach().requires_grad_(True)
        S = S.detach().requires_grad_(True)

        residual, v = self.__bs_residual(t, S)
        g = F.relu(self.K - S)

        complementarity = torch.min(residual, v - g)
        pde_loss = torch.mean(complementarity ** 2)

        return pde_loss

    def __sobolev_loss(self, t1, t2, S_interior):
        """
        Compute Sobolev regularity losses for ``u = v - g``.

        All three terms operate on the residual ``u = v - g`` where
        ``g(S) = max(K - S, 0)`` is the put's intrinsic value.

        * **J2** — H^1 norm of ``u(T, S)`` on the interior: L^2 value term
          plus L^2 norm of ``du/dS``.
        * **J3** — H^{3/4} fractional-in-time norm on the lateral boundary
          ``{S_min, S_max}``: L^2 value plus Gagliardo seminorm with exponent
          ``1 + 2 * 0.75 = 2.5``.
        * **J4** — H^{1/4} fractional-in-time norm of the outward normal
          derivative ``du/dS`` on ``{S_min, S_max}``: L^2 term, time Gagliardo
          seminorm with exponent ``1 + 2 * 0.25 = 1.5``, and a spatial
          difference term across the two boundary faces.

        Parameters
        ----------
        t1 : torch.Tensor, shape (N, 1)
            First time coordinate of each sample pair.
        t2 : torch.Tensor, shape (N, 1)
            Second time coordinate of each sample pair.
        S_interior : torch.Tensor, shape (N, 1)
            Interior asset-price coordinates for computing J2.

        Returns
        -------
        J2 : torch.Tensor
            Scalar H^1 terminal loss.
        J3 : torch.Tensor
            Scalar H^{3/4}-in-time lateral boundary loss.
        J4 : torch.Tensor
            Scalar H^{1/4}-in-time normal-derivative lateral boundary loss.
        """
        S_interior = S_interior.detach().requires_grad_(True)
        batch = S_interior.shape[0]
        ones = torch.ones(batch, 1, device=S_interior.device, dtype=S_interior.dtype)

        # ------------------------------------------------------------------
        # J2: H^1 norm of u(T, S) = v(T, S) - g(S) at t = T
        # ------------------------------------------------------------------
        t_terminal = ones * self.T
        v_T = self.model(t_terminal, S_interior)
        g_T = F.relu(self.K - S_interior)
        u_T = v_T - g_T

        J2_L2 = torch.mean(u_T ** 2)
        grad_u_T = torch.autograd.grad(
            u_T, S_interior, grad_outputs=torch.ones_like(u_T), create_graph=True
        )[0]
        J2_H1 = torch.mean(grad_u_T ** 2)
        J2 = J2_L2 + J2_H1

        # ------------------------------------------------------------------
        # Boundary evaluations for J3 and J4
        # ------------------------------------------------------------------
        # Need requires_grad for computing spatial derivatives at boundaries
        x1 = (ones * self.S_min).requires_grad_(True)   # S_min boundary
        x2 = (ones * self.S_max).requires_grad_(True)   # S_max boundary

        v_t1_x1 = self.model(t1, x1)
        v_t1_x2 = self.model(t1, x2)
        v_t2_x1 = self.model(t2, x1)
        v_t2_x2 = self.model(t2, x2)

        g_x1 = F.relu(self.K - x1)
        g_x2 = F.relu(self.K - x2)

        u_t1_x1 = v_t1_x1 - g_x1
        u_t1_x2 = v_t1_x2 - g_x2
        u_t2_x1 = v_t2_x1 - g_x1
        u_t2_x2 = v_t2_x2 - g_x2

        dt = torch.abs(t1 - t2)

        # ------------------------------------------------------------------
        # J3: H^{3/4} fractional norm in time on {S_min, S_max}
        #     Gagliardo seminorm: |u(t1,x) - u(t2,x)|^2 / |t1-t2|^{1+2s}
        #     with s = 3/4  =>  exponent = 2.5
        # ------------------------------------------------------------------
        denom_t_J3 = (dt ** 2.5).clamp_min(1e-8)

        time_frac_x1 = torch.mean(((u_t1_x1 - u_t2_x1) ** 2) / denom_t_J3)
        time_frac_x2 = torch.mean(((u_t1_x2 - u_t2_x2) ** 2) / denom_t_J3)

        J3_L2 = 0.5 * (torch.mean(u_t1_x1 ** 2) + torch.mean(u_t1_x2 ** 2))
        J3 = J3_L2 + time_frac_x1 + time_frac_x2

        # ------------------------------------------------------------------
        # J4: H^{1/4} fractional norm in time for the normal derivative
        #     ∂u/∂S on {S_min, S_max}
        #     s = 1/4  =>  exponent = 1.5
        # ------------------------------------------------------------------
        dv_dS_t1_x1 = torch.autograd.grad(
            v_t1_x1, x1, grad_outputs=torch.ones_like(v_t1_x1), create_graph=True
        )[0]
        dv_dS_t1_x2 = torch.autograd.grad(
            v_t1_x2, x2, grad_outputs=torch.ones_like(v_t1_x2), create_graph=True
        )[0]
        dv_dS_t2_x1 = torch.autograd.grad(
            v_t2_x1, x1, grad_outputs=torch.ones_like(v_t2_x1), create_graph=True
        )[0]
        dv_dS_t2_x2 = torch.autograd.grad(
            v_t2_x2, x2, grad_outputs=torch.ones_like(v_t2_x2), create_graph=True
        )[0]

        # g(S) = relu(K - S).  dg/dS = -1 if S < K, else 0.
        # At fixed boundary points this is a constant, so the fractional
        # differences of du/dS = dv/dS - dg/dS reduce to differences of dv/dS.
        dg_dS_x1 = -1.0 if self.S_min < self.K else 0.0
        dg_dS_x2 = -1.0 if self.S_max < self.K else 0.0

        du_dS_t1_x1 = dv_dS_t1_x1 - dg_dS_x1
        du_dS_t1_x2 = dv_dS_t1_x2 - dg_dS_x2
        du_dS_t2_x1 = dv_dS_t2_x1 - dg_dS_x1
        du_dS_t2_x2 = dv_dS_t2_x2 - dg_dS_x2

        denom_t_J4 = (dt ** 1.5).clamp_min(1e-8)

        time_frac_x1_J4 = torch.mean(
            ((du_dS_t1_x1 - du_dS_t2_x1) ** 2) / denom_t_J4
        )
        time_frac_x2_J4 = torch.mean(
            ((du_dS_t1_x2 - du_dS_t2_x2) ** 2) / denom_t_J4
        )

        spatial_denom_J4 = (self.S_max - self.S_min) ** 2
        spatial_frac_J4 = torch.mean(
            (du_dS_t1_x2 - du_dS_t1_x1) ** 2 / spatial_denom_J4
        )

        J4_L2 = 0.5 * (
            torch.mean(du_dS_t1_x1 ** 2) + torch.mean(du_dS_t1_x2 ** 2)
        )
        J4 = J4_L2 + time_frac_x1_J4 + time_frac_x2_J4 + spatial_frac_J4

        return J2, J3, J4
