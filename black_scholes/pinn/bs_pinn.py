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

    def set_params(self, K, r, sigma, T, S_min, S_max):
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.S_min = S_min
        self.S_max = S_max

    def train(self, batch_size, epochs, early_stopping):
        """
        Train the PINN model.

        Args:
            batch_size (int): Number of samples per training batch.
            epochs (int): Maximum number of training epochs.
            early_stopping (EarlyStopping): Early stopping mechanism to prevent overfitting
        """
        # Create held-out validation set for early stopping
        val_t_interior, val_S_interior = self.__sample_interior(batch_size)
        val_t_boundary, val_S_boundary = self.__sample_boundary(batch_size)

        for i in range(epochs):

            t_batch, S_batch = self.__sample_interior(batch_size)

            variational_loss = self.__interior_loss(batch_size, t_batch, S_batch)
            terminal_loss, Smin_loss, Smax_loss = self.__boundary_loss(batch_size, t_batch, S_batch)

            loss = self.__process_loss(variational_loss, terminal_loss, Smin_loss, Smax_loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Compute validation loss for early stopping
            variational_loss_val = self.__interior_loss(batch_size, t=val_t_interior, S=val_S_interior)
            terminal_loss_val, Smin_loss_val, Smax_loss_val = self.__boundary_loss(batch_size, t=val_t_boundary, S=val_S_boundary)
            val_loss = self.__process_loss(variational_loss_val, terminal_loss_val, Smin_loss_val, Smax_loss_val, update_dict=False)

            if i % 500 == 0:
                print(f"Iteration {i} | Training Loss: {loss.item()} | Validation Loss: {val_loss.item()}")

            if early_stopping and early_stopping.step(val_loss.item()):
                print(f"Early stopping at epoch {i}")
                break

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
        S = self.sampler.uniform(self.S_min, self.S_max, (batch_size, 1))
        return t, S

    def __sample_boundary(self, batch_size):
        t = self.sampler.uniform(0, self.T, (batch_size, 1))
        S = self.sampler.uniform(self.S_min, self.S_max, (batch_size, 1))
        return t, S

    def __bs_residual(self, t, S):
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
        # If t and S are provided, use them instead of sampling new points
        if t is None or S is None:
            t, S = self.__sample_boundary(batch_size)

        zeros = torch.zeros((batch_size, 1))
        ones = torch.ones((batch_size, 1))

        # Terminal condition: f(T, S) = max(K - S, 0)
        f_T = self.model(ones * self.T, S)
        g_T = torch.maximum(self.K - S, zeros)
        terminal_loss = torch.mean((f_T - g_T) ** 2)

        # S_max loss: f(t, S_max) = 0
        f_inf = self.model(t, ones * self.S_max)
        Smax_loss = torch.mean(f_inf ** 2)

        # S_min loss: f(t, 0) = K
        f_min = self.model(t, zeros)
        Smin_loss = torch.mean((f_min - self.K) ** 2)

        return terminal_loss, Smin_loss, Smax_loss
