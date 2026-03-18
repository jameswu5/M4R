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
        raise NotImplementedError("Training loop not implemented yet")

    def __process_loss(self, variational_loss, terminal_loss, Smin_loss, Smax_loss, Vmin_loss, Vmax_loss, V_update_dict=True):
        raise NotImplementedError("Loss processing not implemented yet")

    def __sample_interior(self, batch_size):
        raise NotImplementedError("Interior sampling not implemented yet")

    def __sample_boundary(self, batch_size):
        raise NotImplementedError("Boundary sampling not implemented yet")

    def __heston_residual(self, t, S, V):
        raise NotImplementedError("Heston PDE residual not implemented yet")

    def __interior_loss(self, batch_size, t=None, S=None, V=None):
        raise NotImplementedError("Interior loss not implemented yet")

    def __boundary_loss(self, batch_size, t=None, S=None, V=None):
        raise NotImplementedError("Boundary loss not implemented yet")
