"""PINN for solving the n-dimensional Black-Scholes PDE for American put options."""

import torch

from utility.model import PINN


class BlackScholesMultiAssetPINN(PINN):
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
        self.K = K
        self.r = r
        self.sigmas = torch.tensor(sigmas, dtype=torch.float32)  # array of length n_assets
        self.corr = torch.tensor(corr, dtype=torch.float32)  # n_assets x n_assets correlation matrix
        self.T = T
        self.S_mins = S_mins
        self.S_maxs = S_maxs

        self.n_assets = len(sigmas)

    def train(self, batch_size, epochs, early_stopping, anneal_freq=500, alpha=0.9):
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
                    'Smax': Smax_loss,
                }
                self.__anneal_weights(unweighted_losses, alpha)

            loss = self.__process_loss(variational_loss, terminal_loss, Smin_loss, Smax_loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            variational_loss_val = self.__interior_loss(batch_size, t=val_t_interior, S=val_S_interior, create_graph=False)
            terminal_loss_val, Smin_loss_val, Smax_loss_val = self.__boundary_loss(batch_size, t=val_t_boundary, S=val_S_boundary)
            val_loss = variational_loss_val + terminal_loss_val + Smin_loss_val + Smax_loss_val

            if i % 500 == 0:
                weight_str = "  ".join(f"{k}={v:.3f}" for k, v in self.loss_weights.items())
                print(f"Iter {i:>6} | Train: {loss.item():.4e} | Val: {val_loss.item():.4e} | Weights: {weight_str}")

            if early_stopping and early_stopping.step(val_loss.item(), self.model):
                print(f"Early stopping at epoch {i}")
                early_stopping.restore(self.model)
                break

    def __anneal_weights(self, unweighted_losses: dict, alpha: float):
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

    def __process_loss(self, variational_loss, terminal_loss, Smin_loss, Smax_loss, update_dict=True):
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

    def __sample_interior(self, batch_size):
        t = self.sampler.uniform(0, self.T, (batch_size, 1))
        S = self.sampler.uniform(self.S_mins, self.S_maxs, (batch_size, self.n_assets))
        return t, S

    def __sample_boundary(self, batch_size):
        return self.__sample_interior(batch_size)

    def __bs_residual(self, t, S, create_graph=True):
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

    def __interior_loss(self, batch_size, t=None, S=None, create_graph=True):
        if t is None or S is None:
            t, S = self.__sample_interior(batch_size)

        assert t.shape[0] == batch_size and S.shape[0] == batch_size

        t.requires_grad_(True)
        S.requires_grad_(True)

        zeros = torch.zeros((batch_size, 1))

        residual, f = self.__bs_residual(t, S, create_graph=create_graph)
        S_prod = torch.prod(S, dim=1, keepdim=True)
        g = torch.maximum(self.K - S_prod, zeros)

        variational_loss = torch.mean(
            torch.minimum(residual, f - g) ** 2
        )

        return variational_loss

    def __boundary_loss(self, batch_size, t=None, S=None):
        if t is None or S is None:
            t, S = self.__sample_interior(batch_size)

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

        # S_max loss: at the upper boundary, the option value equals the intrinsic value.
        # With S_mins = 0, the product can still be near 0 when other assets are small, so
        # enforcing f = 0 uniformly conflicts with the S_min condition. Use max(K - product, 0).
        Smax_loss = 0
        for i in range(self.n_assets):
            S_ = S.clone()
            S_[:, i] = self.S_maxs[i]
            S_prod_ = torch.prod(S_, dim=1, keepdim=True)
            g_ = torch.maximum(self.K - S_prod_, zeros)
            Smax_loss += torch.mean((
                self.model(t, S_) - g_
            )**2)

        return terminal_loss, Smin_loss, Smax_loss
