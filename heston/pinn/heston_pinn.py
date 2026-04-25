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
        t = self.sampler.uniform(0, self.T, (batch_size, 1))
        S = self.sampler.uniform(self.S_min, self.S_max, (batch_size, 1))
        V = self.sampler.uniform(self.V_min, self.V_max, (batch_size, 1))
        return t, S, V

    def __sample_boundary(self, batch_size):
        return self.__sample_interior(batch_size)

    def __heston_residual(self, t, S, V):
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
