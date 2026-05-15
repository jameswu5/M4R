"""PINN for solving the 1D Black-Scholes PDE for American put options."""

import torch

from utility.model import PINN


class BlackScholesPINN(PINN):
    def __init__(self, model_config, seed):
        super().__init__(model_config, seed)
        self.history = {
            'loss': [],
            'variational_loss': [],
            'terminal_loss': [],
            'Smin_loss': [],
            'Smax_loss': []
        }

        # Start tracking the (t=0, S=K) price during training to monitor convergence
        self.atm_price = []

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

        self.loss_weights = {
            'variational': 0.25,
            'terminal': 0.25,
            'Smin': 0.25,
            'Smax': 0.25
        }

        # Create held-out validation set for early stopping
        val_t_interior, val_S_interior = self.__sample_interior(batch_size)
        val_t_boundary, val_S_boundary = self.__sample_boundary(batch_size)

        for i in range(epochs):
            variational_loss = self.__interior_loss(batch_size)
            terminal_loss, Smin_loss, Smax_loss = self.__boundary_loss(batch_size)

            if i > 2000 and i % anneal_freq == 0:
                unweighted_losses = {
                    'variational': variational_loss,
                    'terminal': terminal_loss,
                    'Smin': Smin_loss,
                    'Smax': Smax_loss
                }

                self.__anneal_weights(unweighted_losses, alpha)

            loss = self.__process_loss(variational_loss, terminal_loss, Smin_loss, Smax_loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.atm_price.append(
                self.model(torch.tensor([[0.0]]), torch.tensor([[self.K]]))
            ).item()

            # Compute validation loss for early stopping
            variational_loss_val = self.__interior_loss(batch_size, t=val_t_interior, S=val_S_interior)
            terminal_loss_val, Smin_loss_val, Smax_loss_val = self.__boundary_loss(batch_size, t=val_t_boundary, S=val_S_boundary)
            # val_loss = self.__process_loss(variational_loss_val, terminal_loss_val, Smin_loss_val, Smax_loss_val, update_dict=False)
            val_loss = variational_loss_val + terminal_loss_val + Smin_loss_val + Smax_loss_val

            if i % 500 == 0:
                weight_str = "  ".join(
                    f"{k}={v:.3f}" for k, v in self.loss_weights.items())
                print(f"Iter {i:>6} | Train: {loss.item():.4e} "
                      f"| Val: {val_loss.item():.4e} | Weights: {weight_str}")

            if early_stopping and early_stopping.step(val_loss.item(), self.model):
                print(f"Early stopping at epoch {i}")
                early_stopping.restore(self.model)
                break

    def __process_loss(self, variational_loss, terminal_loss, Smin_loss, Smax_loss, update_dict=True):
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

    def __anneal_weights(self, unweighted_losses: dict, alpha: float):
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
        peak_grad = max(
            g.abs().max().item() for g in total_grads if g is not None
        )

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

        # Renormalise
        total = sum(new_weights.values())
        self.loss_weights = {k: v / total for k, v in new_weights.items()}

    def __sample_interior(self, batch_size):
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
        S : torch.Tensor, shape (batch_size, 1)
            Asset-price coordinates sampled from ``[S_min, S_max]``.
        """
        t = self.sampler.uniform(0, self.T, (batch_size, 1))
        S = self.sampler.uniform(self.S_min, self.S_max, (batch_size, 1))
        return t, S

    def __sample_boundary(self, batch_size):
        """
        Sample points for evaluating the boundary conditions.

        Points are drawn uniformly from the same ``(t, S)`` range as the
        interior; the specific boundary (terminal, S_min, S_max) is enforced
        by fixing the appropriate coordinate inside ``__boundary_loss``.

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
        """
        t = self.sampler.uniform(0, self.T, (batch_size, 1))
        S = self.sampler.uniform(self.S_min, self.S_max, (batch_size, 1))
        return t, S

    def __bs_residual(self, t, S):
        """
        Evaluate the Black-Scholes PDE residual at given collocation points.

        Computes ``-f_t - r S f_S - 0.5 sigma^2 S^2 f_SS + r f`` via
        automatic differentiation, where ``f`` is the network output.

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

        residual = -f_t - self.r * S * f_S - 0.5 * self.sigma**2 * S**2 * f_SS + self.r * f

        return residual, f

    def __interior_loss(self, batch_size, t=None, S=None):
        """
        Compute the variational interior loss for the American put constraint.

        The loss enforces ``min(L[f], f - g) = 0`` in the least-squares sense,
        where ``L[f]`` is the Black-Scholes operator residual and
        ``g = max(K - S, 0)`` is the intrinsic value (early-exercise payoff).

        Parameters
        ----------
        batch_size : int
            Number of collocation points to sample if ``t`` and ``S`` are not
            provided.
        t : torch.Tensor, shape (batch_size, 1), optional
            Pre-sampled time coordinates.  If ``None``, new points are drawn.
        S : torch.Tensor, shape (batch_size, 1), optional
            Pre-sampled asset-price coordinates.  If ``None``, new points are
            drawn.

        Returns
        -------
        variational_loss : torch.Tensor
            Scalar mean-squared variational loss.
        """
        # If t and S are provided, use them instead of sampling new points
        if t is None or S is None:
            t, S = self.__sample_interior(batch_size)

        t.requires_grad_(True)
        S.requires_grad_(True)

        zeros = torch.zeros((batch_size, 1))

        pde_residual, f = self.__bs_residual(t, S)
        g = torch.maximum(self.K - S, zeros)

        variational_loss = torch.mean(
            torch.minimum(pde_residual, f - g) ** 2
        )

        return variational_loss

    def __boundary_loss(self, batch_size, t=None, S=None):
        """
        Compute the boundary condition losses for terminal and spatial boundaries.

        Three conditions are enforced:

        * **Terminal**: ``f(T, S) = max(K - S, 0)``
        * **S_min**: ``f(t, 0) = K`` (put approaches strike as S → 0)
        * **S_max**: ``f(t, S_max) = 0`` (put is worthless deep out-of-the-money)

        Parameters
        ----------
        batch_size : int
            Number of collocation points to sample if ``t`` and ``S`` are not
            provided.
        t : torch.Tensor, shape (batch_size, 1), optional
            Pre-sampled time coordinates.  If ``None``, new points are drawn.
        S : torch.Tensor, shape (batch_size, 1), optional
            Pre-sampled asset-price coordinates.  If ``None``, new points are
            drawn.

        Returns
        -------
        terminal_loss : torch.Tensor
            Scalar MSE loss for the terminal condition at ``t = T``.
        Smin_loss : torch.Tensor
            Scalar MSE loss for the lower spatial boundary at ``S = 0``.
        Smax_loss : torch.Tensor
            Scalar MSE loss for the upper spatial boundary at ``S = S_max``.
        """
        # If t and S are provided, use them instead of sampling new points
        if t is None or S is None:
            t, S = self.__sample_boundary(batch_size)

        zeros = torch.zeros((batch_size, 1))
        ones = torch.ones((batch_size, 1))

        # Terminal condition: f(T, S) = max(K - S, 0)
        f_T = self.model(ones * self.T, S)
        g_T = torch.maximum(self.K - S, zeros)
        terminal_loss = torch.mean((f_T - g_T) ** 2)

        # S_min loss: f(t, 0) = K
        f_min = self.model(t, zeros)
        Smin_loss = torch.mean((f_min - self.K) ** 2)

        # S_max loss: f(t, S_max) = 0
        f_inf = self.model(t, ones * self.S_max)
        Smax_loss = torch.mean(f_inf ** 2)

        return terminal_loss, Smin_loss, Smax_loss
