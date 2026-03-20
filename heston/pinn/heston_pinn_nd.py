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

    def set_params(self):
        self.n_assets = 0
        pass

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

            if early_stopping and early_stopping.step(val_loss.item()):
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
        S = self.sampler.uniform(self.S_min, self.S_max, (batch_size, self.n_assets))
        V = self.sampler.uniform(self.V_min, self.V_max, (batch_size, self.n_assets))
        return t, S, V

    def __sample_boundary(self, batch_size):
        return self.__sample_interior(batch_size)

    def __heston_residual(self, t, S, V):
        raise NotImplementedError("Heston residual computation not implemented for multi-asset case yet.")

    def __interior_loss(self, batch_size, t=None, S=None, V=None):
        if t is None or S is None or V is None:
            t, S, V = self.__sample_interior(batch_size)

        t.requires_grad_(True)
        S.requires_grad_(True)
        V.requires_grad_(True)

        zeros = torch.zeros((batch_size, 1))

        residual, f = self.__heston_residual(t, S, V)
        S_prod = torch.prod(S, dim=1, keepdim=True)
        g = torch.maximum(self.K - S_prod, zeros)

        variational_loss = torch.mean(
            torch.minimum(residual, f - g) ** 2
        )

        return variational_loss

    def __boundary_loss(self, batch_size, t=None, S=None, V=None):
        raise NotImplementedError("Boundary loss computation not implemented for multi-asset case yet.")
