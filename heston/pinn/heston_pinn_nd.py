"""PINN for solving the n-dimensional Heston PDE for American put options."""

import torch
from utility.model import PINN


class HestonMultiAssetPINN(PINN):
    def __init__(self, model_config, seed):
        super().__init__(model_config, seed)

        self.history = {
            'loss': [],
            'variational_loss': [],
            'terminal_loss': [],
            'Smin_loss': [],
            'Smax_loss': [],
            'Vmin_loss': [],
            'Vmax_loss': []
        }

    def set_params(self, K, r, T, kappa, theta, sigma_bar, sigmas, corr, rho_cross, S_min, S_max, V_min, V_max):
        """
        Set the multi-asset Heston model parameters.

        The model uses a single shared variance process ``V`` that drives all
        assets, with asset-asset correlations given by ``corr`` and asset-variance
        correlations given by ``rho_cross``.

        Parameters
        ----------
        K : float
            Strike price of the option.
        r : float
            Risk-free interest rate (annualised).
        T : float
            Time to expiry (in years).
        kappa : float
            Speed of mean reversion of the shared variance process.
        theta : float
            Long-run mean of the shared variance process.
        sigma_bar : float
            Volatility of the shared variance process (vol-of-vol).
        sigmas : array-like of float, length n_assets
            Instantaneous volatility scaling for each asset.
        corr : array-like of float, shape (n_assets, n_assets)
            Correlation matrix between the asset Brownian motions.
        rho_cross : array-like of float, length n_assets
            Correlation between each asset's Brownian motion and the variance
            Brownian motion.
        S_min : float or array-like
            Lower spatial boundary for each asset price.
        S_max : float or array-like
            Upper spatial boundary for each asset price.
        V_min : float
            Lower spatial boundary for the variance.
        V_max : float
            Upper spatial boundary for the variance.
        """
        self.n_assets = len(sigmas)
        self.K = K
        self.r = r
        self.T = T
        self.kappa = kappa
        self.theta = theta
        self.sigma_bar = sigma_bar
        self.sigmas = sigmas
        self.corr = corr
        self.rho_cross = rho_cross
        self.S_min = S_min
        self.S_max = S_max
        self.V_min = V_min
        self.V_max = V_max

    def train(self, batch_size, epochs, early_stopping):
        """
        Train the PINN model using equal loss weights (no annealing).

        Parameters
        ----------
        batch_size : int
            Number of samples per training batch.
        epochs : int
            Maximum number of training epochs.
        early_stopping : EarlyStopping
            Early stopping callback; training halts and the best model is
            restored when the validation loss stops improving.
        """
        # Create held-out validation set for early stopping
        val_t_interior, val_S_interior, val_V_interior = self.__sample_interior(batch_size)
        val_t_boundary, val_S_boundary, val_V_boundary = self.__sample_boundary(batch_size)

        for i in range(epochs):
            variational_loss = self.__interior_loss(batch_size)
            terminal_loss, Smin_loss, Smax_loss, Vmin_loss, Vmax_loss = self.__boundary_loss(batch_size)

            loss = self.__process_loss(variational_loss, terminal_loss, Smin_loss, Smax_loss, Vmin_loss, Vmax_loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Compute validation loss for early stopping
            variational_loss_val = self.__interior_loss(batch_size, t=val_t_interior, S=val_S_interior, V=val_V_interior)
            terminal_loss_val, Smin_loss_val, Smax_loss_val, Vmin_loss_val, Vmax_loss_val = self.__boundary_loss(batch_size, t=val_t_boundary, S=val_S_boundary, V=val_V_boundary)
            val_loss = self.__process_loss(variational_loss_val, terminal_loss_val, Smin_loss_val, Smax_loss_val, Vmin_loss_val, Vmax_loss_val, update_dict=False)

            if i % 500 == 0:
                print(f"Iteration {i} | Training Loss: {loss.item()} | Validation Loss: {val_loss.item()}")

            if early_stopping and early_stopping.step(val_loss.item(), self.model):
                print(f"Early stopping at epoch {i}")
                break

    def __process_loss(self, variational_loss, terminal_loss, Smin_loss, Smax_loss, Vmin_loss, Vmax_loss, update_dict=True):
        """
        Apply loss weights, sum the components, and optionally record to history.

        Parameters
        ----------
        variational_loss : torch.Tensor
            Interior variational (PDE) loss term.
        terminal_loss : torch.Tensor
            Terminal boundary condition loss at ``t = T``.
        Smin_loss : torch.Tensor
            Boundary condition loss averaged over lower asset-price boundaries.
        Smax_loss : torch.Tensor
            Boundary condition loss averaged over upper asset-price boundaries.
        Vmin_loss : torch.Tensor
            Boundary condition loss at ``V = 0``.
        Vmax_loss : torch.Tensor
            Boundary condition loss at ``V = V_inf``.
        update_dict : bool, optional
            If True (default), append each weighted loss to ``self.history``.

        Returns
        -------
        total_loss : torch.Tensor
            Scalar weighted total loss.
        """
        variational_loss *= self.loss_weights['variational']
        terminal_loss *= self.loss_weights['terminal']
        Smin_loss *= self.loss_weights['Smin']
        Smax_loss *= self.loss_weights['Smax']
        Vmin_loss *= self.loss_weights['Vmin']
        Vmax_loss *= self.loss_weights['Vmax']

        total_loss = variational_loss + terminal_loss + Smin_loss + Smax_loss + Vmin_loss + Vmax_loss

        if update_dict:
            self.history['loss'].append(total_loss.item())
            self.history['variational_loss'].append(variational_loss.item())
            self.history['terminal_loss'].append(terminal_loss.item())
            self.history['Smin_loss'].append(Smin_loss.item())
            self.history['Smax_loss'].append(Smax_loss.item())
            self.history['Vmin_loss'].append(Vmin_loss.item())
            self.history['Vmax_loss'].append(Vmax_loss.item())

        return total_loss

    def __sample_interior(self, batch_size):
        """
        Sample collocation points uniformly from the interior domain.

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
            ``[S_min[i], S_max[i]]``.
        V : torch.Tensor, shape (batch_size, 1)
            Shared variance coordinates sampled from ``[V_min, V_max]``.
        """
        t = self.sampler.uniform(0, self.T, (batch_size, 1))
        S = self.sampler.uniform(self.S_min, self.S_max, (batch_size, self.n_assets))
        V = self.sampler.uniform(self.V_min, self.V_max, (batch_size, 1))
        return t, S, V

    def __sample_boundary(self, batch_size):
        """
        Sample points for evaluating the boundary conditions.

        Delegates to ``__sample_interior``; specific boundaries are enforced by
        fixing the appropriate coordinates inside ``__boundary_loss``.

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
        V : torch.Tensor, shape (batch_size, 1)
            Variance coordinates sampled from the interior domain.
        """
        return self.__sample_interior(batch_size)

    def __heston_residual(self, t, S, V):
        """
        Evaluate the multi-asset Heston PDE residual at given collocation points.

        Computes the residual of:
        ``-f_t - r sum_i(S_i f_{S_i}) - kappa(theta - V) f_V
        - 0.5 sum_{i,k}(V sigma_i sigma_k rho_{ik} S_i S_k f_{S_i S_k})
        - 0.5 sigma_bar^2 V f_VV
        - sum_i(V sigma_i sigma_bar rho_cross_i S_i f_{S_i V}) + r f``
        via automatic differentiation, assembling the Hessian row-by-row.

        Parameters
        ----------
        t : torch.Tensor, shape (N, 1)
            Time coordinates.  Must have ``requires_grad=True``.
        S : torch.Tensor, shape (N, n_assets)
            Asset-price coordinates.  Must have ``requires_grad=True``.
        V : torch.Tensor, shape (N, 1)
            Shared variance coordinates.  Must have ``requires_grad=True``.

        Returns
        -------
        residual : torch.Tensor, shape (N, 1)
            PDE residual at each collocation point.
        f : torch.Tensor, shape (N, 1)
            Network output (option price) at each collocation point.
        """
        f = self.model(t, S, V)

        f_t, f_S, f_V = torch.autograd.grad(
            f, (t, S, V), torch.ones_like(f), create_graph=True
        )

        f_SS = []
        for i in range(self.n_assets):
            grad_i = torch.autograd.grad(
                f_S[:, i:i+1], S,
                torch.ones_like(f_S[:, i:i+1]),
                create_graph=True
            )[0]
            f_SS.append(grad_i.unsqueeze(1))
        f_SS = torch.cat(f_SS, dim=1)   # (N,n,n)

        f_VV = torch.autograd.grad(
            f_V, V, torch.ones_like(f_V), create_graph=True
        )[0]

        f_SV = []
        for i in range(self.n_assets):
            grad_i = torch.autograd.grad(
                f_S[:, i:i+1], V,
                torch.ones_like(f_S[:, i:i+1]),
                create_graph=True
            )[0]
            f_SV.append(grad_i)
        f_SV = torch.cat(f_SV, dim=1)   # (N,n)

        Lf = self.r * torch.sum(S * f_S, dim=1, keepdim=True)
        Lf += self.kappa * (self.theta - V) * f_V

        # Asset-asset diffusion
        diffusion = 0
        for i in range(self.n_assets):
            for k in range(self.n_assets):
                coeff = V * self.sigmas[i] * self.sigmas[k] * self.corr[i, k] * S[:, i:i+1] * S[:, k:k+1]
                diffusion += coeff * f_SS[:, i, k:k+1]
        Lf += 0.5 * diffusion

        # Variance diffusion
        Lf += 0.5 * self.sigma_bar ** 2 * V * f_VV

        # Cross S-V terms
        cross = 0
        for i in range(self.n_assets):
            coeff = V * self.sigmas[i] * self.sigma_bar * self.rho_cross[i] * S[:, i:i+1]
            cross += coeff * f_SV[:, i:i+1]

        Lf += cross

        residual = -f_t - Lf + self.r * f

        return residual, f

    def __interior_loss(self, batch_size, t=None, S=None, V=None):
        """
        Compute the variational interior loss for the American put constraint.

        Enforces the complementarity condition using the Fischer-Burmeister
        function rather than ``min``, with payoff ``g = max(K - prod(S), 0)``.

        Parameters
        ----------
        batch_size : int
            Number of collocation points to sample if ``t``, ``S``, or ``V``
            are not provided.
        t : torch.Tensor, shape (batch_size, 1), optional
            Pre-sampled time coordinates.  If ``None``, new points are drawn.
        S : torch.Tensor, shape (batch_size, n_assets), optional
            Pre-sampled asset-price coordinates.  If ``None``, new points are
            drawn.
        V : torch.Tensor, shape (batch_size, 1), optional
            Pre-sampled variance coordinates.  If ``None``, new points are
            drawn.

        Returns
        -------
        variational_loss : torch.Tensor
            Scalar mean-squared Fischer-Burmeister complementarity loss.
        """
        if t is None or S is None or V is None:
            t, S, V = self.__sample_interior(batch_size)

        t.requires_grad_(True)
        S.requires_grad_(True)
        V.requires_grad_(True)

        zeros = torch.zeros((batch_size, 1))

        residual, f = self.__heston_residual(t, S, V)
        S_prod = torch.prod(S, dim=1, keepdim=True)
        g = torch.maximum(self.K - S_prod, zeros)

        # variational_loss = torch.mean(
        #     torch.minimum(residual, f - g) ** 2
        # )

        variational_loss = torch.mean(
            self.__fischer_burmeister(residual, f - g) ** 2
        )

        return variational_loss

    def __boundary_loss(self, batch_size, t=None, S=None, V=None):
        """
        Compute the boundary condition losses for the multi-asset Heston put.

        Five conditions are enforced:

        * **Terminal**: ``f(T, S, V) = max(K - prod(S), 0)``
        * **S_min**: ``f(t, S', V) = K`` when any ``S'_i = 0``
          (product collapses to zero)
        * **S_max**: ``f(t, S', V) = 0`` when any ``S'_i = 10 * S_max[i]``
          (product is very large)
        * **V_min**: ``f(t, S, 0) = max(K - prod(S), 0)`` (zero vol = intrinsic)
        * **V_max**: ``f(t, S, V_inf) = K`` (infinite vol ≈ strike)

        Parameters
        ----------
        batch_size : int
            Number of collocation points to sample if ``t``, ``S``, or ``V``
            are not provided.
        t : torch.Tensor, shape (batch_size, 1), optional
            Pre-sampled time coordinates.  If ``None``, new points are drawn.
        S : torch.Tensor, shape (batch_size, n_assets), optional
            Pre-sampled asset-price coordinates.  If ``None``, new points are
            drawn.
        V : torch.Tensor, shape (batch_size, 1), optional
            Pre-sampled variance coordinates.  If ``None``, new points are
            drawn.

        Returns
        -------
        terminal_loss : torch.Tensor
            Scalar MSE loss for the terminal condition at ``t = T``.
        Smin_loss : torch.Tensor
            Scalar MSE loss averaged over lower asset-price boundaries.
        Smax_loss : torch.Tensor
            Scalar MSE loss averaged over upper asset-price boundaries.
        Vmin_loss : torch.Tensor
            Scalar MSE loss for the zero-variance boundary.
        Vmax_loss : torch.Tensor
            Scalar MSE loss for the infinite-variance boundary.
        """
        if t is None or S is None or V is None:
            t, S, V = self.__sample_boundary(batch_size)

        S_list = [S[:, i].unsqueeze(1) for i in range(self.n_assets)]

        zeros = torch.zeros((batch_size, 1))
        ones = torch.ones((batch_size, 1))

        # Terminal condition f(T, S, V) = payoff(S)
        f_T = self.model(self.T * ones, S, V)
        S_prod = torch.prod(S, dim=1, keepdim=True)
        g = torch.maximum(self.K - S_prod, zeros)
        terminal_loss = torch.mean((f_T - g)**2)

        # Smin loss: f(t, S', V) = K if S' = S_min for any asset, else f(t, S', V) = 0
        Smin_loss = 0
        for i in range(self.n_assets):
            S_boundary = S.clone()
            S_boundary[:, i] = 0
            S_boundary_list = [S_boundary[:, j].unsqueeze(1) for j in range(self.n_assets)]
            Smin_loss += torch.mean((
                self.model(t, *S_boundary_list, V) - self.K
            )**2)

        # Smax loss: f(t, S', V) = 0 if S' = S_max for any asset
        Smax_loss = 0
        for i in range(self.n_assets):
            S_boundary = S.clone()
            S_boundary[:, i] = self.S_max[i] * 10
            S_boundary_list = [S_boundary[:, j].unsqueeze(1) for j in range(self.n_assets)]
            Smax_loss += torch.mean((
                self.model(t, *S_boundary_list, V)
            )**2)

        # Vmin loss: f(t, S, 0) = payoff(S)
        f_Vmin = self.model(t, *S_list, zeros)
        Vmin_loss = torch.mean((f_Vmin - g) ** 2)

        # Vmax loss: f(t, S, v_inf) = K
        V_inf = ones * self.V_max * 100
        f_Vinf = self.model(t, *S_list, V_inf)
        Vmax_loss = torch.mean((f_Vinf - self.K) ** 2)

        return terminal_loss, Smin_loss, Smax_loss, Vmin_loss, Vmax_loss

    def __fischer_burmeister(self, a, b, lambda_=1e-6):
        """
        Evaluate the smoothed Fischer-Burmeister complementarity function.

        Provides a smooth alternative to ``min(a, b)`` for enforcing the
        complementarity condition ``min(L[f], f - g) = 0``.

        Parameters
        ----------
        a : torch.Tensor
            First argument (typically the PDE residual).
        b : torch.Tensor
            Second argument (typically ``f - g``).
        lambda_ : float, optional
            Smoothing parameter; makes the function differentiable at the
            origin (default ``1e-6``).

        Returns
        -------
        fb : torch.Tensor
            ``a + b - sqrt(a^2 + b^2 + lambda_)``.  Equals zero when
            ``min(a, b) = 0`` and is smooth everywhere.
        """
        return a + b - torch.sqrt(a**2 + b**2 + lambda_)
