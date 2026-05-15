"""PINN for solving the n-dimensional Black-Scholes PDE for American put options."""

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

from utility.model import PINN
from abc import ABC, abstractmethod
from black_scholes.tree.tree import binomial_tree_batch


class BSMultiPINN(PINN, ABC):
    def __init__(self, model_config, seed):
        super().__init__(model_config, seed)
        self.history = {
            'loss': [],
            'variational_loss': [],
            'terminal_loss': [],
            'Smin_loss': [],
            'Smax_loss': []
        }

    def set_params(self, K, r, sigmas, corr, T, S_mins, S_maxs):
        """
        Set the multi-asset Black-Scholes model parameters.

        Parameters
        ----------
        K : float
            Strike price of the option.
        r : float
            Risk-free interest rate (annualised).
        sigmas : array-like of float, length n_assets
            Volatility of each underlying asset (annualised).
        corr : array-like of float, shape (n_assets, n_assets)
            Correlation matrix between asset log-returns.
        T : float
            Time to expiry (in years).
        S_mins : array-like of float, length n_assets
            Lower spatial boundary for each asset's price.
        S_maxs : array-like of float, length n_assets
            Upper spatial boundary for each asset's price.
        """
        self.K = K
        self.r = r
        self.sigmas = torch.tensor(sigmas, dtype=torch.float32)  # array of length n_assets
        self.corr = torch.tensor(corr, dtype=torch.float32)  # n_assets x n_assets correlation matrix
        self.T = T
        self.S_mins = S_mins
        self.S_maxs = S_maxs

        self.n_assets = len(sigmas)

    def train(self, batch_size, epochs, early_stopping, anneal_freq=500, alpha=0.9):
        """
        Train the PINN model with automatic loss reweighting via learning rate annealing.

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
        val_t_interior, val_S_interior = self._sample_interior(batch_size)
        val_t_boundary, val_S_boundary = self._sample_boundary(batch_size)

        for i in range(epochs):
            variational_loss = self._interior_loss(batch_size)
            terminal_loss, Smin_loss, Smax_loss = self._boundary_loss(batch_size)

            if i > 2000 and i % anneal_freq == 0:
                unweighted_losses = {
                    'variational': variational_loss,
                    'terminal': terminal_loss,
                    'Smin': Smin_loss,
                    'Smax': Smax_loss,
                }
                self._anneal_weights(unweighted_losses, alpha)

            loss = self._process_loss(variational_loss, terminal_loss, Smin_loss, Smax_loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            variational_loss_val = self._interior_loss(batch_size, t=val_t_interior, S=val_S_interior, create_graph=False)
            terminal_loss_val, Smin_loss_val, Smax_loss_val = self._boundary_loss(batch_size, t=val_t_boundary, S=val_S_boundary)
            val_loss = variational_loss_val + terminal_loss_val + Smin_loss_val + Smax_loss_val

            if i % 500 == 0:
                weight_str = "  ".join(f"{k}={v:.3f}" for k, v in self.loss_weights.items())
                print(f"Iter {i:>6} | Train: {loss.item():.4e} | Val: {val_loss.item():.4e} | Weights: {weight_str}")

            if early_stopping and early_stopping.step(val_loss.item(), self.model):
                print(f"Early stopping at epoch {i}")
                early_stopping.restore(self.model)
                break

    def _anneal_weights(self, unweighted_losses: dict, alpha: float):
        """
        Update loss weights using gradient-based learning rate annealing.

        Each weight is rescaled so that the gradient magnitudes of all loss
        terms are approximately equal, then smoothed with an exponential moving
        average and renormalised to sum to one.

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

        total_loss = sum(self.loss_weights[k] * v for k, v in unweighted_losses.items())
        total_grads = torch.autograd.grad(
            total_loss, params, retain_graph=True, create_graph=True, allow_unused=True
        )
        peak_grad = max(g.abs().max().item() for g in total_grads if g is not None)

        new_weights = {}
        for name, loss in unweighted_losses.items():
            weighted_loss = self.loss_weights[name] * loss
            grads = torch.autograd.grad(
                weighted_loss, params, retain_graph=True, create_graph=True, allow_unused=True
            )
            grad_tensors = [g for g in grads if g is not None]
            mean_grad = (
                sum(g.abs().mean().item() for g in grad_tensors) / len(grad_tensors)
            ) if grad_tensors else 1.0
            lambda_hat = peak_grad / (mean_grad + 1e-8)
            new_weights[name] = alpha * self.loss_weights[name] + (1.0 - alpha) * lambda_hat

        total = sum(new_weights.values())
        self.loss_weights = {k: v / total for k, v in new_weights.items()}

    def _process_loss(self, variational_loss, terminal_loss, Smin_loss, Smax_loss, update_dict=True):
        """
        Apply loss weights, sum the components, and optionally record to history.

        Parameters
        ----------
        variational_loss : torch.Tensor
            Interior variational (PDE) loss term.
        terminal_loss : torch.Tensor
            Terminal boundary condition loss at ``t = T``.
        Smin_loss : torch.Tensor
            Boundary condition loss at the lower spatial boundaries.
        Smax_loss : torch.Tensor
            Boundary condition loss at the upper spatial boundaries.
        update_dict : bool, optional
            If True (default), append each weighted loss to ``self.history``.

        Returns
        -------
        loss : torch.Tensor
            Scalar weighted total loss.
        """
        variational_loss *= self.loss_weights['variational']
        terminal_loss *= self.loss_weights['terminal']
        Smin_loss *= self.loss_weights['Smin']
        Smax_loss *= self.loss_weights['Smax']

        loss = variational_loss + terminal_loss + Smin_loss + Smax_loss

        if update_dict:
            self.history['loss'].append(loss.item())
            self.history['variational_loss'].append(variational_loss.item())
            self.history['terminal_loss'].append(terminal_loss.item())
            self.history['Smin_loss'].append(Smin_loss.item())
            self.history['Smax_loss'].append(Smax_loss.item())

        return loss

    def _sample_interior(self, batch_size):
        """
        Sample collocation points uniformly from the interior of the domain.

        Parameters
        ----------
        batch_size : int
            Number of points to sample.

        Returns
        -------
        t : torch.Tensor, shape (batch_size, 1)
            Time coordinates sampled from ``[0, T]``.
        S : torch.Tensor, shape (batch_size, n_assets)
            Asset-price coordinates, each column sampled from
            ``[S_mins[i], S_maxs[i]]``.
        """
        t = self.sampler.uniform(0, self.T, (batch_size, 1))
        S = self.sampler.uniform(self.S_mins, self.S_maxs, (batch_size, self.n_assets))
        return t, S

    def _sample_boundary(self, batch_size):
        """
        Sample points for evaluating the boundary conditions.

        Delegates to ``_sample_interior``; specific boundaries are enforced by
        fixing the appropriate coordinates inside ``_boundary_loss``.

        Parameters
        ----------
        batch_size : int
            Number of points to sample.

        Returns
        -------
        t : torch.Tensor, shape (batch_size, 1)
            Time coordinates sampled from ``[0, T]``.
        S : torch.Tensor, shape (batch_size, n_assets)
            Asset-price coordinates sampled from the interior domain.
        """
        return self._sample_interior(batch_size)

    def _bs_residual(self, t, S, create_graph=True):
        """
        Evaluate the multi-asset Black-Scholes PDE residual at collocation points.

        Computes ``-f_t - r sum_i(S_i f_{S_i}) - 0.5 sum_{i,j}(sigma_i sigma_j
        rho_{ij} S_i S_j f_{S_i S_j}) + r f`` via automatic differentiation,
        where the full Hessian ``f_SS`` is assembled row-by-row.

        Parameters
        ----------
        t : torch.Tensor, shape (N, 1)
            Time coordinates.  Must have ``requires_grad=True``.
        S : torch.Tensor, shape (N, n_assets)
            Asset-price coordinates.  Must have ``requires_grad=True``.
        create_graph : bool, optional
            If True (default), retain the computation graph for higher-order
            gradients during training.  Set to False for validation passes.

        Returns
        -------
        residual : torch.Tensor, shape (N, 1)
            PDE residual at each collocation point.
        f : torch.Tensor, shape (N, 1)
            Network output (option price) at each collocation point.
        """
        batch_size = S.shape[0]

        f = self.model(t, S)

        f_t, f_S = torch.autograd.grad(
            f, (t, S), grad_outputs=torch.ones_like(f), create_graph=True
        )

        rows = []
        for i in range(self.n_assets):
            f_i = f_S[:, i].unsqueeze(-1)
            # retain_graph must be True between loop iterations so the shared f_S graph
            # isn't freed before all Hessian rows are computed; also True on the final
            # iteration when create_graph=True so the training backward can traverse it.
            retain = (i < self.n_assets - 1) or create_graph
            f_i_S = torch.autograd.grad(
                f_i, S, grad_outputs=torch.ones_like(f_i), create_graph=create_graph, retain_graph=retain
            )[0]
            rows.append(f_i_S.unsqueeze(1))  # (batch, 1, n_assets)
        f_SS = torch.cat(rows, dim=1)        # (batch, n_assets, n_assets)

        cov_matrix = torch.outer(self.sigmas, self.sigmas) * self.corr

        drift = self.r * torch.sum(S * f_S, dim=1, keepdim=True)  # shape (batch_size, 1)

        S_outer = S.unsqueeze(2) * S.unsqueeze(1)  # shape (batch_size, n_assets, n_assets)
        cov_broadcast = cov_matrix.unsqueeze(0).expand(batch_size, self.n_assets, self.n_assets)  # shape (batch_size, n_assets, n_assets)

        elements = cov_broadcast * S_outer * f_SS  # shape (batch_size, n_assets, n_assets)
        diffusion = 0.5 * torch.sum(elements, dim=(1, 2), keepdim=False).unsqueeze(-1)  # → (batch_size, 1)

        residual = -f_t - drift - diffusion + self.r * f  # shape (batch_size, 1)

        return residual, f

    @abstractmethod
    def _payoff(self, S):
        """
        Compute the option's intrinsic (exercise) payoff for a batch of prices.

        Parameters
        ----------
        S : torch.Tensor, shape (N, n_assets)
            Asset prices at which to evaluate the payoff.

        Returns
        -------
        payoff : torch.Tensor, shape (N, 1)
            Intrinsic value ``max(K - g(S), 0)`` for each sample, where
            ``g(S)`` is the subclass-specific function of the asset prices.
        """
        raise NotImplementedError("Must be implemented by subclass")

    def _interior_loss(self, batch_size, t=None, S=None, create_graph=True):
        """
        Compute the variational interior loss for the American put constraint.

        The loss enforces ``min(L[f], f - g) = 0`` in the least-squares sense,
        where ``L[f]`` is the Black-Scholes operator residual and ``g`` is the
        intrinsic payoff returned by ``_payoff``.

        Parameters
        ----------
        batch_size : int
            Number of collocation points to sample if ``t`` and ``S`` are not
            provided.
        t : torch.Tensor, shape (batch_size, 1), optional
            Pre-sampled time coordinates.  If ``None``, new points are drawn.
        S : torch.Tensor, shape (batch_size, n_assets), optional
            Pre-sampled asset-price coordinates.  If ``None``, new points are
            drawn.
        create_graph : bool, optional
            Passed to ``_bs_residual``; set to False for validation passes.

        Returns
        -------
        variational_loss : torch.Tensor
            Scalar mean-squared variational loss.
        """
        if t is None or S is None:
            t, S = self._sample_interior(batch_size)

        assert t.shape[0] == batch_size and S.shape[0] == batch_size

        t.requires_grad_(True)
        S.requires_grad_(True)

        residual, f = self._bs_residual(t, S, create_graph=create_graph)
        g = self._payoff(S)

        variational_loss = torch.mean(
            torch.minimum(residual, f - g) ** 2
        )

        return variational_loss

    @abstractmethod
    def _boundary_loss(self, batch_size, t=None, S=None):
        """
        Compute boundary condition losses for terminal and spatial boundaries.

        Parameters
        ----------
        batch_size : int
            Number of collocation points to sample if ``t`` and ``S`` are not
            provided.
        t : torch.Tensor, shape (batch_size, 1), optional
            Pre-sampled time coordinates.  If ``None``, new points are drawn.
        S : torch.Tensor, shape (batch_size, n_assets), optional
            Pre-sampled asset-price coordinates.  If ``None``, new points are
            drawn.

        Returns
        -------
        terminal_loss : torch.Tensor
            Scalar MSE loss for the terminal condition at ``t = T``.
        Smin_loss : torch.Tensor
            Scalar MSE loss averaged over the lower spatial boundaries.
        Smax_loss : torch.Tensor
            Scalar MSE loss averaged over the upper spatial boundaries.
        """
        raise NotImplementedError("Must be implemented by subclass")


class BSProductPINN(BSMultiPINN):
    def _payoff(self, S):
        """
        Compute the product-put payoff ``max(K - prod(S), 0)``.

        Parameters
        ----------
        S : torch.Tensor, shape (N, n_assets)
            Asset prices at which to evaluate the payoff.

        Returns
        -------
        payoff : torch.Tensor, shape (N, 1)
            Intrinsic value based on the product of all asset prices.
        """
        S_prod = torch.prod(S, dim=1, keepdim=True)
        payoff = torch.maximum(self.K - S_prod, torch.zeros_like(S_prod))
        return payoff

    def _boundary_loss(self, batch_size, t=None, S=None):
        """
        Compute boundary condition losses for the product-put option.

        Three conditions are enforced:

        * **Terminal**: ``f(T, S) = max(K - prod(S), 0)``
        * **S_min**: ``f(t, S) = K`` when any ``S_i = 0`` (product collapses to zero)
        * **S_max**: ``f(t, S) = max(K - prod(S), 0)`` when any ``S_i = S_maxs[i]``

        Parameters
        ----------
        batch_size : int
            Number of collocation points to sample if ``t`` and ``S`` are not
            provided.
        t : torch.Tensor, shape (batch_size, 1), optional
            Pre-sampled time coordinates.  If ``None``, new points are drawn.
        S : torch.Tensor, shape (batch_size, n_assets), optional
            Pre-sampled asset-price coordinates.  If ``None``, new points are
            drawn.

        Returns
        -------
        terminal_loss : torch.Tensor
            Scalar MSE loss for the terminal condition at ``t = T``.
        Smin_loss : torch.Tensor
            Scalar MSE loss averaged over all lower spatial boundaries.
        Smax_loss : torch.Tensor
            Scalar MSE loss averaged over all upper spatial boundaries.
        """
        if t is None or S is None:
            t, S = self._sample_interior(batch_size)

        assert t.shape[0] == batch_size and S.shape[0] == batch_size

        zeros = torch.zeros((batch_size, 1))
        ones = torch.ones((batch_size, 1))

        # Terminal condition: f(T, S) = max(K - S, 0)
        f_T = self.model(self.T * ones, S)
        g_T = torch.maximum(self.K - torch.prod(S, dim=1, keepdim=True), zeros)
        terminal_loss = torch.mean((f_T - g_T) ** 2)

        # S_min loss: if any S_i = 0, then f(t, S) = K
        Smin_loss = 0
        for i in range(self.n_assets):
            S_ = S.clone()
            S_[:, i] = 0
            Smin_loss += torch.mean((
                self.model(t, S_) - self.K
            )**2)
        Smin_loss /= self.n_assets

        # S_max loss: at the upper boundary, the option value equals the intrinsic value.
        # With S_mins = 0, the product can still be near 0 when other assets are small, so
        # enforcing f = 0 uniformly conflicts with the S_min condition. Use max(K - product, 0).
        Smax_loss = 0
        for i in range(self.n_assets):
            S_ = S.clone()
            S_[:, i] = self.S_maxs[i]
            g_ = self._payoff(S_)
            Smax_loss += torch.mean((
                self.model(t, S_) - g_
            )**2)
        Smax_loss /= self.n_assets

        return terminal_loss, Smin_loss, Smax_loss


class BSMaxPINN(BSMultiPINN):
    def set_params(self, K, r, sigmas, corr, T, S_mins, S_maxs, n_grid_t=100, n_grid_S=300, compute_interpolators=True):
        """
        Set model parameters and precompute boundary interpolation grids.

        When ``S_i = 0``, ``max(S_0, S_1) = S_{1-i}``, so the option reduces
        to an American put on asset ``1-i``.  A bilinear interpolator over a
        precomputed ``(t, S)`` price grid is built for each asset so that
        these boundary values can be evaluated in ``O(B log n_grid)`` per
        batch rather than running the binomial tree every iteration.

        Parameters
        ----------
        K : float
            Strike price of the option.
        r : float
            Risk-free interest rate (annualised).
        sigmas : array-like of float, length 2
            Volatility of each underlying asset (annualised).
        corr : array-like of float, shape (2, 2)
            Correlation matrix between the two asset log-returns.
        T : float
            Time to expiry (in years).
        S_mins : array-like of float, length 2
            Lower spatial boundary for each asset's price.
        S_maxs : array-like of float, length 2
            Upper spatial boundary for each asset's price.
        n_grid_t : int, optional
            Number of time points in the precomputed grid (default 100).
        n_grid_S : int, optional
            Number of asset-price points per asset in the precomputed grid
            (default 300).
        compute_interpolators : bool, optional
            If True (default), build the boundary interpolators.  Set to
            False to skip the precomputation (e.g. when loading a saved model).
        """
        self.K = K
        self.r = r
        self.sigmas = torch.tensor(sigmas, dtype=torch.float32)  # array of length n_assets
        self.corr = torch.tensor(corr, dtype=torch.float32)  # n_assets x n_assets correlation matrix
        self.T = T
        self.S_mins = S_mins
        self.S_maxs = S_maxs

        self.n_assets = len(sigmas)
        assert self.n_assets == 2, "BSMaxPINN is currently implemented only for 2 assets"

        # Precompute American put price on a (t, S) grid for each asset and store
        # a bilinear interpolator. Avoids running O(B * n_steps^2) tree evaluations
        # per training batch; interpolation is O(B log n_grid) instead.
        if compute_interpolators:
            t_grid = np.linspace(0, T * (1 - 1e-6), n_grid_t)  # avoid tau = 0 at T
            self.interpolators = []
            print("Precomputing boundary interpolation grids...")
            for i in range(self.n_assets):
                S_grid = np.linspace(S_mins[i], S_maxs[i], n_grid_S)
                price_grid = np.zeros((n_grid_t, n_grid_S))
                for j, t_val in enumerate(t_grid):
                    tau = T - t_val
                    price_grid[j] = binomial_tree_batch(
                        S_grid, K, r, sigmas[i], tau, n=100,
                        option_type="put", exercise_type="american"
                    )
                interp = RegularGridInterpolator(
                    (t_grid, S_grid), price_grid,
                    method='linear', bounds_error=False, fill_value=None
                )
                self.interpolators.append(interp)
            print("Done.")

    def _payoff(self, S):
        """
        Compute the max-put payoff ``max(K - max(S), 0)``.

        Parameters
        ----------
        S : torch.Tensor, shape (N, n_assets)
            Asset prices at which to evaluate the payoff.

        Returns
        -------
        payoff : torch.Tensor, shape (N, 1)
            Intrinsic value based on the maximum of all asset prices.
        """
        S_max, _ = torch.max(S, dim=1, keepdim=True)
        payoff = torch.maximum(self.K - S_max, torch.zeros_like(S_max))
        return payoff

    def _boundary_loss(self, batch_size, t=None, S=None):
        """
        Compute boundary condition losses for the max-put option.

        Three conditions are enforced:

        * **Terminal**: ``f(T, S) = max(K - max(S), 0)``
        * **S_min**: when ``S_i = 0``, the option reduces to an American put
          on asset ``1-i``; values are looked up from a precomputed interpolator.
        * **S_max**: ``f(t, S) = max(K - max(S), 0)`` when any ``S_i = S_maxs[i]``

        Parameters
        ----------
        batch_size : int
            Number of collocation points to sample if ``t`` and ``S`` are not
            provided.
        t : torch.Tensor, shape (batch_size, 1), optional
            Pre-sampled time coordinates.  If ``None``, new points are drawn.
        S : torch.Tensor, shape (batch_size, n_assets), optional
            Pre-sampled asset-price coordinates.  If ``None``, new points are
            drawn.

        Returns
        -------
        terminal_loss : torch.Tensor
            Scalar MSE loss for the terminal condition at ``t = T``.
        Smin_loss : torch.Tensor
            Scalar MSE loss averaged over all lower spatial boundaries.
        Smax_loss : torch.Tensor
            Scalar MSE loss averaged over all upper spatial boundaries.
        """
        if t is None or S is None:
            t, S = self._sample_interior(batch_size)

        assert t.shape[0] == batch_size and S.shape[0] == batch_size

        ones = torch.ones((batch_size, 1))

        # Terminal condition: f(T, S) = payoff(S)
        f_T = self.model(self.T * ones, S)
        g = self._payoff(S)
        terminal_loss = torch.mean((f_T - g) ** 2)

        # S_min loss: when S_i = 0, max(S_0, S_1) = S_{1-i}, so the option reduces
        # to an American put on asset (1-i) with its own volatility sigma[1-i].
        Smin_loss = 0
        t_numpy = t.squeeze().detach().numpy()  # (batch_size,)

        for i in range(self.n_assets):
            S_ = S.clone()
            S_[:, i] = 0

            S_other = S_[:, 1 - i].detach().numpy().ravel()  # (batch_size,)
            points = np.stack([t_numpy, S_other], axis=1)    # (batch_size, 2)
            v_put = torch.tensor(
                self.interpolators[1 - i](points), dtype=torch.float32
            ).unsqueeze(1)
            Smin_loss += torch.mean((self.model(t, S_) - v_put) ** 2)
        Smin_loss /= self.n_assets

        # S_max loss: if any S_i is very large, then f(t, S) = intrinsic value
        Smax_loss = 0
        for i in range(self.n_assets):
            S_ = S.clone()
            S_[:, i] = self.S_maxs[i]
            g_ = self._payoff(S_)
            Smax_loss += torch.mean((self.model(t, S_) - g_) ** 2)
        Smax_loss /= self.n_assets

        return terminal_loss, Smin_loss, Smax_loss


class BSMinPINN(BSMultiPINN):
    def set_params(self, K, r, sigmas, corr, T, S_mins, S_maxs, n_grid_t=100, n_grid_S=300, compute_interpolators=True):
        """
        Set model parameters and precompute boundary interpolation grids.

        When ``S_i -> S_maxs[i]``, ``min(S_0, S_1) = S_{1-i}``, so the option
        reduces to an American put on asset ``1-i``.  A bilinear interpolator
        over a precomputed ``(t, S)`` price grid is built for each asset so
        that these boundary values can be evaluated efficiently during training.

        Parameters
        ----------
        K : float
            Strike price of the option.
        r : float
            Risk-free interest rate (annualised).
        sigmas : array-like of float, length 2
            Volatility of each underlying asset (annualised).
        corr : array-like of float, shape (2, 2)
            Correlation matrix between the two asset log-returns.
        T : float
            Time to expiry (in years).
        S_mins : array-like of float, length 2
            Lower spatial boundary for each asset's price.
        S_maxs : array-like of float, length 2
            Upper spatial boundary for each asset's price.
        n_grid_t : int, optional
            Number of time points in the precomputed grid (default 100).
        n_grid_S : int, optional
            Number of asset-price points per asset in the precomputed grid
            (default 300).
        compute_interpolators : bool, optional
            If True (default), build the boundary interpolators.  Set to
            False to skip the precomputation (e.g. when loading a saved model).
        """
        self.K = K
        self.r = r
        self.sigmas = torch.tensor(sigmas, dtype=torch.float32)
        self.corr = torch.tensor(corr, dtype=torch.float32)
        self.T = T
        self.S_mins = S_mins
        self.S_maxs = S_maxs

        self.n_assets = len(sigmas)
        assert self.n_assets == 2, "BSMinPINN is currently implemented only for 2 assets"

        # Precompute American put price on a (t, S) grid for each asset.
        # When S_i -> S_maxs[i], the min collapses to S_{1-i}, so the option
        # reduces to an American put on asset (1-i) with volatility sigma[1-i].
        if compute_interpolators:
            t_grid = np.linspace(0, T * (1 - 1e-6), n_grid_t)
            self.interpolators = []
            print("Precomputing boundary interpolation grids...")
            for i in range(self.n_assets):
                S_grid = np.linspace(S_mins[i], S_maxs[i], n_grid_S)
                price_grid = np.zeros((n_grid_t, n_grid_S))
                for j, t_val in enumerate(t_grid):
                    tau = T - t_val
                    price_grid[j] = binomial_tree_batch(
                        S_grid, K, r, sigmas[i], tau, n=100,
                        option_type="put", exercise_type="american"
                    )
                interp = RegularGridInterpolator(
                    (t_grid, S_grid), price_grid,
                    method='linear', bounds_error=False, fill_value=None
                )
                self.interpolators.append(interp)
            print("Done.")

    def _payoff(self, S):
        """
        Compute the min-put payoff ``max(K - min(S), 0)``.

        Parameters
        ----------
        S : torch.Tensor, shape (N, n_assets)
            Asset prices at which to evaluate the payoff.

        Returns
        -------
        payoff : torch.Tensor, shape (N, 1)
            Intrinsic value based on the minimum of all asset prices.
        """
        S_min, _ = torch.min(S, dim=1, keepdim=True)
        payoff = torch.maximum(self.K - S_min, torch.zeros_like(S_min))
        return payoff

    def _boundary_loss(self, batch_size, t=None, S=None):
        """
        Compute boundary condition losses for the min-put option.

        Three conditions are enforced:

        * **Terminal**: ``f(T, S) = max(K - min(S), 0)``
        * **S_min**: when ``S_i = 0``, ``min(S) = 0`` so the put is worth
          ``K`` immediately; enforces ``f(t, S) = K``.
        * **S_max**: when ``S_i -> S_maxs[i]``, the option reduces to an
          American put on asset ``1-i``; values from a precomputed interpolator.

        Parameters
        ----------
        batch_size : int
            Number of collocation points to sample if ``t`` and ``S`` are not
            provided.
        t : torch.Tensor, shape (batch_size, 1), optional
            Pre-sampled time coordinates.  If ``None``, new points are drawn.
        S : torch.Tensor, shape (batch_size, n_assets), optional
            Pre-sampled asset-price coordinates.  If ``None``, new points are
            drawn.

        Returns
        -------
        terminal_loss : torch.Tensor
            Scalar MSE loss for the terminal condition at ``t = T``.
        Smin_loss : torch.Tensor
            Scalar MSE loss averaged over all lower spatial boundaries.
        Smax_loss : torch.Tensor
            Scalar MSE loss averaged over all upper spatial boundaries.
        """
        if t is None or S is None:
            t, S = self._sample_interior(batch_size)

        assert t.shape[0] == batch_size and S.shape[0] == batch_size

        ones = torch.ones((batch_size, 1))

        # Terminal condition: f(T, S) = payoff(S)
        f_T = self.model(self.T * ones, S)
        g = self._payoff(S)
        terminal_loss = torch.mean((f_T - g) ** 2)

        # S_min loss: when S_i = 0, min(S) = 0 so the put is worth K immediately.
        Smin_loss = 0
        for i in range(self.n_assets):
            S_ = S.clone()
            S_[:, i] = 0
            Smin_loss += torch.mean((self.model(t, S_) - self.K) ** 2)
        Smin_loss /= self.n_assets

        # S_max loss: when S_i -> S_maxs[i], min(S_0, S_1) = S_{1-i}, so the
        # option reduces to an American put on asset (1-i) with sigma[1-i].
        Smax_loss = 0
        t_numpy = t.squeeze().detach().numpy()  # (batch_size,)

        for i in range(self.n_assets):
            S_ = S.clone()
            S_[:, i] = self.S_maxs[i]

            S_other = S_[:, 1 - i].detach().numpy().ravel()   # (batch_size,)
            points = np.stack([t_numpy, S_other], axis=1)     # (batch_size, 2)
            v_put = torch.tensor(
                self.interpolators[1 - i](points), dtype=torch.float32
            ).unsqueeze(1)
            Smax_loss += torch.mean((self.model(t, S_) - v_put) ** 2)
        Smax_loss /= self.n_assets

        return terminal_loss, Smin_loss, Smax_loss
