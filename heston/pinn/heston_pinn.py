"""PINN for solving the 1D Heston PDE for American put options."""

import torch
from utility.model import PINN


class HestonPINN(PINN):
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

    def set_params(self, K, r, T, kappa, theta, sigma, rho, S_min, S_max, V_min, V_max):
        """
        Set the Heston model parameters.

        Parameters
        ----------
        K : float
            Strike price of the option.
        r : float
            Risk-free interest rate (annualised).
        T : float
            Time to expiry (in years).
        kappa : float
            Speed of mean reversion of the variance process.
        theta : float
            Long-run mean of the variance process.
        sigma : float
            Volatility of variance (vol-of-vol).
        rho : float
            Correlation between the asset and variance Brownian motions.
        S_min : float
            Lower spatial boundary for the asset price.
        S_max : float
            Upper spatial boundary for the asset price.
        V_min : float
            Lower spatial boundary for the variance.
        V_max : float
            Upper spatial boundary for the variance.
        """
        self.K = K
        self.r = r
        self.T = T
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
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
            Boundary condition loss at ``S = S_min``.
        Smax_loss : torch.Tensor
            Boundary condition loss at ``S = S_max``.
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
        S : torch.Tensor, shape (batch_size, 1)
            Asset-price coordinates sampled from ``[S_min, S_max]``.
        V : torch.Tensor, shape (batch_size, 1)
            Variance coordinates sampled from ``[V_min, V_max]``.
        """
        t = self.sampler.uniform(0, self.T, (batch_size, 1))
        S = self.sampler.uniform(self.S_min, self.S_max, (batch_size, 1))
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
        S : torch.Tensor, shape (batch_size, 1)
            Asset-price coordinates sampled from the interior domain.
        V : torch.Tensor, shape (batch_size, 1)
            Variance coordinates sampled from the interior domain.
        """
        return self.__sample_interior(batch_size)

    def __heston_residual(self, t, S, V):
        """
        Evaluate the Heston PDE residual at given collocation points.

        Computes ``L[f] = -f_t - r S f_S - kappa(theta - V) f_V
        - 0.5(S^2 V f_SS + 2 rho sigma S V f_SV + sigma^2 V f_VV) + r f``
        via automatic differentiation.

        Parameters
        ----------
        t : torch.Tensor, shape (N, 1)
            Time coordinates.  Must have ``requires_grad=True``.
        S : torch.Tensor, shape (N, 1)
            Asset-price coordinates.  Must have ``requires_grad=True``.
        V : torch.Tensor, shape (N, 1)
            Variance coordinates.  Must have ``requires_grad=True``.

        Returns
        -------
        residual : torch.Tensor, shape (N, 1)
            PDE residual at each collocation point.
        f : torch.Tensor, shape (N, 1)
            Network output (option price) at each collocation point.
        """
        f = self.model(t, S, V)

        f_t, f_S, f_V = torch.autograd.grad(
            f, (t, S, V), grad_outputs=torch.ones_like(f), create_graph=True, retain_graph=True
        )

        f_SS = torch.autograd.grad(
            f_S, S, grad_outputs=torch.ones_like(f_S), create_graph=True, retain_graph=True
        )[0]

        f_VV = torch.autograd.grad(
            f_V, V, grad_outputs=torch.ones_like(f_V), create_graph=True, retain_graph=True
        )[0]

        f_SV = torch.autograd.grad(
            f_S,
            V,
            grad_outputs=torch.ones_like(f_S),
            create_graph=True,
            retain_graph=True
        )[0]

        Lf = (
            self.r * S * f_S
            + self.kappa * (self.theta - V) * f_V
            + 0.5 * (
                S**2 * V * f_SS
                + 2.0 * self.rho * self.sigma * S * V * f_SV
                + self.sigma**2 * V * f_VV
            )
        )

        residual = -f_t - Lf + self.r * f

        return residual, f

    def __interior_loss(self, batch_size, t=None, S=None, V=None):
        """
        Compute the variational interior loss for the American put constraint.

        Enforces ``min(L[f], f - g) = 0`` in the least-squares sense, where
        ``L[f]`` is the Heston PDE residual and ``g = max(K - S, 0)`` is the
        intrinsic payoff.

        Parameters
        ----------
        batch_size : int
            Number of collocation points to sample if ``t``, ``S``, or ``V``
            are not provided.
        t : torch.Tensor, shape (batch_size, 1), optional
            Pre-sampled time coordinates.  If ``None``, new points are drawn.
        S : torch.Tensor, shape (batch_size, 1), optional
            Pre-sampled asset-price coordinates.  If ``None``, new points are
            drawn.
        V : torch.Tensor, shape (batch_size, 1), optional
            Pre-sampled variance coordinates.  If ``None``, new points are
            drawn.

        Returns
        -------
        variational_loss : torch.Tensor
            Scalar mean-squared variational loss.
        """
        if t is None or S is None or V is None:
            t, S, V = self.__sample_interior(batch_size)

        t.requires_grad_(True)
        S.requires_grad_(True)
        V.requires_grad_(True)

        zeros = torch.zeros((batch_size, 1))

        residual, f = self.__heston_residual(t, S, V)
        g = torch.maximum(self.K - S, zeros)

        variational_loss = torch.mean(
            torch.minimum(residual, f - g) ** 2
        )

        return variational_loss

    def __boundary_loss(self, batch_size, t=None, S=None, V=None):
        """
        Compute the boundary condition losses for the Heston put option.

        Five conditions are enforced:

        * **Terminal**: ``f(T, S, V) = max(K - S, 0)``
        * **S_min**: ``f(t, S_min, V) = K - S_min`` (deep in-the-money)
        * **S_max**: ``f(t, S_max, V) = 0`` (deep out-of-the-money)
        * **V_min**: ``f(t, S, 0) = max(K - S, 0)`` (zero vol = intrinsic value)
        * **V_max**: ``f(t, S, V_inf) = K`` (infinite vol ≈ strike)

        Parameters
        ----------
        batch_size : int
            Number of collocation points to sample if ``t``, ``S``, or ``V``
            are not provided.
        t : torch.Tensor, shape (batch_size, 1), optional
            Pre-sampled time coordinates.  If ``None``, new points are drawn.
        S : torch.Tensor, shape (batch_size, 1), optional
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
            Scalar MSE loss for the lower asset-price boundary.
        Smax_loss : torch.Tensor
            Scalar MSE loss for the upper asset-price boundary.
        Vmin_loss : torch.Tensor
            Scalar MSE loss for the zero-variance boundary.
        Vmax_loss : torch.Tensor
            Scalar MSE loss for the infinite-variance boundary.
        """
        if t is None or S is None or V is None:
            t, S, V = self.__sample_boundary(batch_size)

        zeros = torch.zeros((batch_size, 1))
        ones = torch.ones((batch_size, 1))

        # Terminal condition: f(T, S, V) = max(K - S, 0)
        f_T = self.model(ones * self.T, S, V)
        g_T = torch.maximum(self.K - S, zeros)
        terminal_loss = torch.mean((f_T - g_T) ** 2)

        # S_min loss: f(t, S_min, V) = K - S_min
        f_Smin = self.model(t, ones * self.S_min, V)
        g_Smin = self.K - self.S_min
        Smin_loss = torch.mean((f_Smin - g_Smin) ** 2)

        # S_max loss: f(t, S_max, V) = 0
        f_Smax = self.model(t, ones * self.S_max, V)
        Smax_loss = torch.mean(f_Smax ** 2)

        # V_0 loss: f(t, S, V_min) = max(K - S, 0)
        f_Vmin = self.model(t, S, zeros)
        g_Vmin = torch.maximum(self.K - S, zeros)
        Vmin_loss = torch.mean((f_Vmin - g_Vmin) ** 2)

        # V_inf loss: f(t, S, V_inf) = K
        V_inf = self.V_max * 100
        f_Vinf = self.model(t, S, ones * V_inf)
        g_Vinf = self.K
        Vmax_loss = torch.mean((f_Vinf - g_Vinf) ** 2)

        return terminal_loss, Smin_loss, Smax_loss, Vmin_loss, Vmax_loss
