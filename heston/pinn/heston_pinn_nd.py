"""PINN for solving the n-dimensional Heston PDE for American put options."""

import numpy as np
import torch
import torch.nn.functional as F
from utility.model import PINN


class HestonMultiAssetPINN(PINN):
    """Physics-informed neural network for pricing a multi-asset American put under a shared Heston variance process."""

    def __init__(self, model_config, seed):
        super().__init__(model_config, seed)

        # Fraction of interior collocation points drawn near the free boundary
        # prod(S) = K, and the multiplicative spread of that cloud around it.
        self.boundary_frac = 0.5
        self.boundary_band = 0.25

        # Fraction of interior V samples drawn from a high-density band around the
        # long-run variance theta (the rest spread over the full [V_min, V_max]).
        self.v_band_frac = 0.6

        self.history = {
            'loss': [],
            'variational_loss': [],
            'terminal_loss': [],
            'Smin_loss': [],
            'Smax_loss': [],
            'Vmin_loss': [],
            'Vmax_loss': []
        }
        # Worst-point interior residual per step (tracked separately from the
        # weighted history so it does not distort plot_losses, which it dwarfs).
        self.interior_max_history = []

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
        self.S_min = S_min
        self.S_max = S_max
        self.V_min = V_min
        self.V_max = V_max

        # Per-asset upper bound used to scale the asset inputs into ~[0, 1].
        self.S_max_t = torch.tensor(np.asarray(S_max, dtype=float), dtype=torch.float32)

        self.sigmas = torch.tensor(sigmas, dtype=torch.float32)
        self.corr = torch.tensor(corr, dtype=torch.float32)

        L = np.linalg.cholesky(corr)
        L_rho = L @ rho_cross

        self.sigma_L_rho = torch.tensor(
            sigmas * L_rho, dtype=torch.float32
        )

        # precompute sigma_i * sigma_k * Sigma_ik
        self.sigma_outer = torch.tensor(
            np.outer(sigmas, sigmas) * corr, dtype=torch.float32
        )

    def train(self, batch_size, epochs, early_stopping, clip_norm=1.0, lbfgs_epochs=500):
        """
        Train the PINN with Adam (gradient-clipped) followed by an L-BFGS
        refinement phase.

        Parameters
        ----------
        batch_size : int
            Collocation points per training batch.
        epochs : int
            Maximum Adam training epochs.
        early_stopping : EarlyStopping
            Halts the Adam phase when validation loss stagnates; its best model
            is restored before the L-BFGS phase begins.
        clip_norm : float, optional
            Max global gradient norm for the Adam phase (set None/0 to disable).
        lbfgs_epochs : int, optional
            Max L-BFGS iterations for the refinement phase (set 0 to skip).
        """
        # Create held-out validation set for early stopping
        val_t_interior, val_S_interior, val_V_interior = self.__sample_interior(batch_size)
        val_t_boundary, val_S_boundary, val_V_boundary = self.__sample_boundary(batch_size)

        for i in range(epochs):
            variational_loss = self.__interior_loss(batch_size)
            interior_max = self._last_interior_max
            terminal_loss, Smin_loss, Smax_loss, Vmin_loss, Vmax_loss = self.__boundary_loss(batch_size)

            loss = self.__process_loss(variational_loss, terminal_loss, Smin_loss, Smax_loss, Vmin_loss, Vmax_loss)
            self.interior_max_history.append(interior_max)

            self.optimizer.zero_grad()
            loss.backward()
            if clip_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
            self.optimizer.step()
            self.scheduler.step()

            # Compute validation loss for early stopping
            variational_loss_val = self.__interior_loss(batch_size, t=val_t_interior, S=val_S_interior, V=val_V_interior)
            terminal_loss_val, Smin_loss_val, Smax_loss_val, Vmin_loss_val, Vmax_loss_val = self.__boundary_loss(batch_size, t=val_t_boundary, S=val_S_boundary, V=val_V_boundary)
            val_loss = self.__process_loss(variational_loss_val, terminal_loss_val, Smin_loss_val, Smax_loss_val, Vmin_loss_val, Vmax_loss_val, update_dict=False)

            if i % 500 == 0:
                print(f"Iteration {i} | Training Loss: {loss.item():.3e} | Validation Loss: {val_loss.item():.3e} | Max interior sq-residual: {interior_max:.3e}")

            if early_stopping and early_stopping.step(val_loss.item(), self.model):
                print(f"Early stopping at epoch {i}")
                break

        # Restore the best Adam iterate so refinement starts from the best point.
        if early_stopping is not None:
            early_stopping.restore(self.model)

        if lbfgs_epochs:
            self.__lbfgs_finetune(batch_size, lbfgs_epochs, clip_norm)

    def __lbfgs_finetune(self, batch_size, max_iter, clip_norm):
        """
        Refine the network with L-BFGS on a fixed collocation set.

        L-BFGS assumes a deterministic objective, so the collocation points are
        sampled once and reused for every closure evaluation (unlike the Adam
        phase, which resamples each step).
        """
        t_int, S_int, V_int = self.__sample_interior(batch_size)
        t_bnd, S_bnd, V_bnd = self.__sample_boundary(batch_size)

        optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            max_iter=max_iter,
            history_size=50,
            line_search_fn='strong_wolfe',
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
        )

        step = [0]

        def closure():
            optimizer.zero_grad()
            variational_loss = self.__interior_loss(batch_size, t=t_int, S=S_int, V=V_int)
            terminal_loss, Smin_loss, Smax_loss, Vmin_loss, Vmax_loss = self.__boundary_loss(batch_size, t=t_bnd, S=S_bnd, V=V_bnd)
            loss = self.__process_loss(variational_loss, terminal_loss, Smin_loss, Smax_loss, Vmin_loss, Vmax_loss)
            self.interior_max_history.append(self._last_interior_max)

            loss.backward()
            if clip_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)

            step[0] += 1
            if step[0] % 50 == 0:
                print(f"L-BFGS closure {step[0]} | Loss: {loss.item():.3e} | Max interior sq-residual: {self._last_interior_max:.3e}")
            return loss

        print("Starting L-BFGS refinement...")
        optimizer.step(closure)

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

    def _normalize(self, t, *args):
        """
        Scale raw (t, S..., V) inputs into ~[0, 1] before the network.

        The last positional argument is always the variance V; the rest are the
        asset prices (either a single (N, n_assets) tensor or individual
        columns). Normalisation is differentiable, so gradients taken w.r.t. the
        raw t, S, V (used by the PDE residual) remain correct via the chain rule.
        """
        t = torch.as_tensor(t, dtype=torch.float32)
        args = [torch.as_tensor(a, dtype=torch.float32) for a in args]

        *S_args, V = args
        t_n = t / self.T
        V_n = (V - self.V_min) / (self.V_max - self.V_min)

        if len(S_args) == 1 and S_args[0].dim() == 2 and S_args[0].shape[-1] == self.n_assets:
            S_n = [S_args[0] / self.S_max_t]
        else:
            S_n = [s / self.S_max_t[i] for i, s in enumerate(S_args)]

        return [t_n, *S_n, V_n]

    def _f(self, t, *args):
        """Network output passed through softplus to enforce f >= 0."""
        return F.softplus(self.model(*self._normalize(t, *args)))

    def predict(self, t, *S):
        self.model.eval()
        with torch.no_grad():
            return F.softplus(self.model(*self._normalize(t, *S)))

    def __sample_curve(self, n):
        """
        Sample asset prices concentrated near the free boundary prod(S) = K.

        The first n_assets-1 coordinates are drawn log-uniformly; the last is set
        to K / prod(others) and given a multiplicative log-normal spread so the
        cloud straddles the exercise manifold rather than sitting exactly on it.
        Points whose final coordinate falls outside the domain are clipped.
        """
        lo = np.maximum(np.asarray(self.S_min, dtype=float), 1e-3)
        hi = np.asarray(self.S_max, dtype=float)

        S = np.empty((n, self.n_assets))
        for i in range(self.n_assets - 1):
            S[:, i] = np.exp(self.sampler.rng.uniform(np.log(lo[i]), np.log(hi[i]), n))

        prod_prev = np.prod(S[:, :self.n_assets - 1], axis=1) if self.n_assets > 1 else np.ones(n)
        noise = np.exp(self.sampler.rng.normal(0.0, self.boundary_band, n))
        S[:, -1] = np.clip(self.K / prod_prev * noise, lo[-1], hi[-1])

        return torch.tensor(S, dtype=torch.float32)

    def __sample_interior(self, batch_size):
        """Sample (t, S, V) collocation points, concentrating a fraction near the free boundary prod(S) = K."""
        t = self.sampler.uniform(0, self.T, (batch_size, 1))

        # Concentrate V density near theta so the enlarged [V_min, V_max] domain
        # does not starve the region of interest (V ~ a few * theta) of points.
        V = self.sampler.segmented_uniform_1d(
            self.V_min, self.V_max, centre=self.theta, radius=3 * self.theta,
            weight=self.v_band_frac, shape=(batch_size,)
        ).reshape(-1, 1)

        n_curve = int(batch_size * self.boundary_frac)
        n_bulk = batch_size - n_curve

        atm = self.K ** (1.0 / self.n_assets)
        centres = np.full(self.n_assets, atm)
        radii = np.full(self.n_assets, atm)
        weights = np.full(self.n_assets, 0.5)
        S_bulk = self.sampler.segmented_uniform(self.S_min, self.S_max, centres, radii, weights, n_bulk)

        S = torch.cat([S_bulk, self.__sample_curve(n_curve)], dim=0)
        return t, S, V

    def __sample_boundary(self, batch_size):
        """Sample (t, S, V) points for boundary condition evaluation."""
        return self.__sample_interior(batch_size)

    def __heston_residual(self, t, S, V):
        """Evaluate the multi-asset Heston PDE residual at (t, S, V) via automatic differentiation."""
        f = self._f(t, S, V)

        f_t, f_S, f_V = torch.autograd.grad(
            f, (t, S, V), torch.ones_like(f), create_graph=True
        )

        f_SS = []
        f_SV = []
        for i in range(self.n_assets):
            grads = torch.autograd.grad(
                f_S[:, i:i+1], (S, V),
                torch.ones_like(f_S[:, i:i+1]),
                create_graph=True
            )
            f_SS.append(grads[0].unsqueeze(1))  # (N, 1, n)
            f_SV.append(grads[1])               # (N, 1)
        f_SS = torch.cat(f_SS, dim=1)   # (N, n, n)
        f_SV = torch.cat(f_SV, dim=1)  # (N, n)

        f_VV = torch.autograd.grad(
            f_V, V, torch.ones_like(f_V), create_graph=True
        )[0]

        Lf = self.r * torch.sum(S * f_S, dim=1, keepdim=True)
        Lf += self.kappa * (self.theta - V) * f_V

        S_outer = S.unsqueeze(2) * S.unsqueeze(1)  # (N, n, n)
        diffusion = torch.einsum(
            'ik, bik, bik -> b',
            self.sigma_outer, S_outer, f_SS
        )
        Lf += 0.5 * V * diffusion.unsqueeze(1)

        # Variance diffusion.
        Lf += 0.5 * self.sigma_bar ** 2 * V * f_VV

        cross = V * self.sigma_bar * torch.sum(
            self.sigma_L_rho * S * f_SV, dim=1, keepdim=True
        )
        Lf += cross

        residual = -f_t - Lf + self.r * f

        return residual, f

    def __interior_loss(self, batch_size, t=None, S=None, V=None):
        """Mean-squared variational complementarity loss min(G[f], f-g)^2 on the interior with product-put payoff."""
        if t is None or S is None or V is None:
            t, S, V = self.__sample_interior(batch_size)

        t.requires_grad_(True)
        S.requires_grad_(True)
        V.requires_grad_(True)

        zeros = torch.zeros((batch_size, 1))

        residual, f = self.__heston_residual(t, S, V)
        S_prod = torch.prod(S, dim=1, keepdim=True)
        g = torch.maximum(self.K - S_prod, zeros)

        per_point = torch.minimum(residual, f - g) ** 2
        variational_loss = torch.mean(per_point)
        self._last_interior_max = per_point.max().item()

        return variational_loss

    def __boundary_loss(self, batch_size, t=None, S=None, V=None):
        """MSE losses for the five multi-asset Heston boundary conditions."""
        if t is None or S is None or V is None:
            t, S, V = self.__sample_boundary(batch_size)

        S_list = [S[:, i].unsqueeze(1) for i in range(self.n_assets)]

        zeros = torch.zeros((batch_size, 1))
        ones = torch.ones((batch_size, 1))

        # Terminal condition f(T, S, V) = payoff(S)
        f_T = self._f(self.T * ones, S, V)
        S_prod = torch.prod(S, dim=1, keepdim=True)
        g = torch.maximum(self.K - S_prod, zeros)
        terminal_loss = torch.mean((f_T - g)**2)

        # Smin loss: when any asset is at 0 the product is identically 0, so the
        # American put is worth K. The other assets are sampled away from 0 to
        # avoid double-counting against the Smax boundary.
        Smin_loss = 0
        for i in range(self.n_assets):
            S_boundary = S.clone()
            S_boundary[:, i] = 0
            S_boundary_list = [S_boundary[:, j].unsqueeze(1) for j in range(self.n_assets)]
            Smin_loss += torch.mean((
                self._f(t, *S_boundary_list, V) - self.K
            )**2)

        # Smax loss: for the product payoff the option is only worthless when ALL
        # assets are simultaneously large, so set every coordinate to its upper
        # bound (the in-domain edge) rather than only one at a time.
        S_far = torch.tensor(np.asarray(self.S_max), dtype=torch.float32).expand(batch_size, self.n_assets)
        Smax_loss = torch.mean(self._f(t, S_far, V) ** 2)

        # Vmin loss: at V = 0 every variance-multiplied term vanishes, leaving the
        # degenerate PDE  -f_t - (r * sum_i S_i f_{S_i} + kappa*theta f_V) + r f = 0.
        # The variance characteristic points inward there (kappa*theta > 0), so by
        # the Fichera condition no Dirichlet value is admissible; instead impose the
        # American complementarity min(G0[f], f - g) = 0 with this degenerate G0.
        # Evaluating __heston_residual at V = 0 reproduces G0 automatically.
        t_v0 = t.detach().clone().requires_grad_(True)
        S_v0 = S.detach().clone().requires_grad_(True)
        V_v0 = torch.zeros((batch_size, 1), requires_grad=True)
        residual_0, f_v0 = self.__heston_residual(t_v0, S_v0, V_v0)
        Vmin_loss = torch.mean(torch.minimum(residual_0, f_v0 - g) ** 2)

        # Vmax loss: at the top of the variance domain the value becomes
        # insensitive to V, so impose the far-field Neumann condition dV f = 0 at
        # V_max instead of the (only asymptotically valid) Dirichlet value f = K.
        V_top = (ones * self.V_max).requires_grad_(True)
        f_Vmax = self._f(t, *S_list, V_top)
        f_Vmax_V = torch.autograd.grad(
            f_Vmax, V_top, torch.ones_like(f_Vmax), create_graph=True
        )[0]
        Vmax_loss = torch.mean(f_Vmax_V ** 2)

        return terminal_loss, Smin_loss, Smax_loss, Vmin_loss, Vmax_loss
