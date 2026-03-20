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
        self.sigmas = torch.tensor(sigmas, dtype=torch.float32) # array of length n_assets
        self.corr = torch.tensor(corr, dtype=torch.float32) # n_assets x n_assets correlation matrix
        self.T = T
        self.S_mins = S_mins
        self.S_maxs = S_maxs

        self.n_assets = len(sigmas)

    def train(self, batch_size, epochs, early_stopping):
        val_t_interior, val_S_interior = self.__sample_interior(batch_size)
        val_t_boundary, val_S_boundary = self.__sample_boundary(batch_size)

        for i in range(epochs):
            variational_loss = self.__interior_loss(batch_size)
            terminal_loss, Smin_loss, Smax_loss = self.__boundary_loss(batch_size)

            loss = self.__process_loss(variational_loss, terminal_loss, Smin_loss, Smax_loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            variational_loss_val = self.__interior_loss(batch_size, t=val_t_interior, S=val_S_interior)
            terminal_loss_val, Smin_loss_val, Smax_loss_val = self.__boundary_loss(batch_size, t=val_t_boundary, S=val_S_boundary)
            val_loss = self.__process_loss(variational_loss_val, terminal_loss_val, Smin_loss_val, Smax_loss_val, update_dict=False)

            if i % 500 == 0:
                print(f"Iteration {i} | Training Loss: {loss.item()} | Validation Loss: {val_loss.item()}")

            if early_stopping and early_stopping.step(val_loss.item()):
                print(f"Early stopping at epoch {i}")

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
    
    def __bs_residual(self, t, S):
        batch_size = S.shape[0]

        f = self.model(t, S)

        f_t, f_S = torch.autograd.grad(
            f, (t, S), grad_outputs=torch.ones_like(f), create_graph=True
        )

        rows = []
        for i in range(self.n_assets):
            f_i = f_S[:, i].unsqueeze(-1)
            f_i_S = torch.autograd.grad(
                f_i, S, grad_outputs=torch.ones_like(f_i), create_graph=True
            )[0]
            rows.append(f_i_S.unsqueeze(1))  # (batch, 1, n_assets)
        f_SS = torch.cat(rows, dim=1)        # (batch, n_assets, n_assets) — stays in graph ✅


        cov_matrix = torch.outer(self.sigmas, self.sigmas) * self.corr

        drift = self.r * torch.sum(S * f_S, dim=1, keepdim=True)  # shape (batch_size, 1)

        S_outer = S.unsqueeze(2) * S.unsqueeze(1)  # shape (batch_size, n_assets, n_assets)
        cov_broadcast = cov_matrix.unsqueeze(0).expand(batch_size, self.n_assets, self.n_assets)  # shape (batch_size, n_assets, n_assets)

        elements = cov_broadcast * S_outer * f_SS  # shape (batch_size, n_assets, n_assets)
        diffusion = 0.5 * torch.sum(elements, dim=(1, 2))

        residual = -f_t - drift - diffusion + self.r * f  # shape (batch_size, 1)

        return residual, f

    def __interior_loss(self, batch_size, t=None, S=None):
        if t is None or S is None:
            t, S = self.__sample_interior(batch_size)

        t.requires_grad_(True)
        S.requires_grad_(True)

        zeros = torch.zeros((batch_size, 1))

        residual, f = self.__bs_residual(t, S)
        S_prod = torch.prod(S, dim=1, keepdim=True)
        g = torch.maximum(self.K - S_prod, zeros)

        variational_loss = torch.mean(
            torch.minimum(residual, f - g) ** 2
        )

        return variational_loss

    def __boundary_loss(self, batch_size, t=None, S=None):
        if t is None or S is None:
            t, S = self.__sample_interior(batch_size)

        t.requires_grad_(True)
        S.requires_grad_(True)

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
        
        # S_max loss: if any S_i is very large, then f(t, S) = 0
        Smax_loss = 0
        for i in range(self.n_assets):
            S_ = S.clone()
            S_[:, i] = self.S_maxs[i]
            Smax_loss += torch.mean((
                self.model(t, S_)
            )**2)
        
        return terminal_loss, Smin_loss, Smax_loss
