"""PINN for solving the n-dimensional Heston PDE for American put options."""

import torch
from utility.model import PINN


class HestonMultiAssetPINN(PINN):
    """Physics-informed neural network for pricing a multi-asset American put under a shared Heston variance process."""

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
        Set the multi-asset Heston model and domain parameters.

        All assets share a single variance process V; asset-variance correlations
        are given by rho_cross.

        Parameters
        ----------
        K : float
            Strike price.
        r : float
            Risk-free rate (annualised).
        T : float
            Time to expiry (in years).
        kappa : float
            Mean reversion speed of the shared variance process.
        theta : float
            Long-run mean of the shared variance process.
        sigma_bar : float
            Volatility of the shared variance process (vol-of-vol).
        sigmas : array-like of float, length n_assets
            Instantaneous volatility scaling for each asset.
        corr : array-like of float, shape (n_assets, n_assets)
            Correlation matrix between the asset Brownian motions.
        rho_cross : array-like of float, length n_assets
            Correlation between each asset's Brownian motion and the variance process.
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
        Train the PINN using equal loss weights (no annealing).

        Parameters
        ----------
        batch_size : int
            Collocation points per training batch.
        epochs : int
            Maximum training epochs.
        early_stopping : EarlyStopping
            Halts training and restores best model when validation loss stagnates.
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
        """Apply loss weights, sum, and optionally append to history."""
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
        """Sample (t, S, V) collocation points uniformly from the domain interior."""
        t = self.sampler.uniform(0, self.T, (batch_size, 1))
        S = self.sampler.uniform(self.S_min, self.S_max, (batch_size, self.n_assets))
        V = self.sampler.uniform(self.V_min, self.V_max, (batch_size, 1))
        return t, S, V

    def __sample_boundary(self, batch_size):
        """Sample (t, S, V) points for boundary condition evaluation."""
        return self.__sample_interior(batch_size)

    def __heston_residual(self, t, S, V):
        """Evaluate the multi-asset Heston PDE residual at (t, S, V) via automatic differentiation."""
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
        """Fischer-Burmeister complementarity loss on the interior with product-put payoff."""
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
        """MSE losses for the five multi-asset Heston boundary conditions."""
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
        """Smooth approximation a + b - sqrt(a^2 + b^2 + lambda_) to min(a, b)."""
        return a + b - torch.sqrt(a**2 + b**2 + lambda_)
