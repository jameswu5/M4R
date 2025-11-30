import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class Payoff(ABC):
    @abstractmethod
    def __call__(self, S, K):
        pass

    @abstractmethod
    def boundary_loss(self, model, t_boundary, S_boundary, **kwargs):
        pass


class Put(Payoff):
    def __call__(self, S, K):
        return torch.relu(K - S)

    def boundary_loss(self, model, t_boundary, S_boundary, **kwargs):
        K = kwargs.get('K', None)
        S_max = kwargs.get('S_max', None)[0]
        S_min = kwargs.get('S_min', None)[0]

        if K is None or S_max is None or S_min is None:
            raise ValueError("K, S_max, and S_min must be provided for boundary loss calculation.")

        length = t_boundary.shape[0]
        ones = torch.ones((length, 1))

        v_1 = model(ones, S_boundary)
        payoff = self(S_boundary, K)
        boundary_loss = nn.MSELoss()(v_1, payoff)

        # f(t, S_max) = 0
        v_Smax = model(t_boundary, ones * S_max)
        boundary_Smax_loss = nn.MSELoss()(v_Smax, torch.zeros((length, 1)))

        v_Smin = model(t_boundary, ones * S_min)
        boundary_Smin_loss = nn.MSELoss()(v_Smin, ones * (K - S_min))

        total_boundary_loss = 3 * boundary_loss + boundary_Smax_loss + boundary_Smin_loss

        return total_boundary_loss


class PutProductMultipleAssets(Payoff):
    def __call__(self, S, K):
        product = torch.prod(S, dim=1, keepdim=True)
        return torch.relu(K - product)

    def boundary_loss(self, model, t_boundary, S_boundary, **kwargs):
        K = kwargs.get('K', None)
        S_max = kwargs.get('S_max', None)
        S_min = kwargs.get('S_min', None)

        # S_max shape: (n_assets,)
        # S_min shape: (n_assets,)

        if K is None or S_max is None or S_min is None:
            raise ValueError("K, S_max, and S_min must be provided for boundary loss calculation.")

        length = t_boundary.shape[0]
        ones = torch.ones((length, 1))

        v_1 = model(ones, S_boundary)
        payoff = self(S_boundary, K)
        boundary_loss = nn.MSELoss()(v_1, payoff)

        # f(t, S1_max, ..., SN_max) = 0
        # Create a tensor where each row is S_max
        S_max_array = np.tile(S_max, (length, 1))
        S_max_tensor = torch.tensor(S_max_array, dtype=S_boundary.dtype, device=S_boundary.device)
        v_Smax = model(t_boundary, S_max_tensor)
        boundary_Smax_loss = nn.MSELoss()(v_Smax, torch.zeros((length, 1)))

        # f(t, S1_min, ..., SN_min) = K - prod(S_min)
        S_min_array = np.tile(S_min, (length, 1))
        S_min_tensor = torch.tensor(S_min_array, dtype=S_boundary.dtype, device=S_boundary.device)
        v_Smin = model(t_boundary, S_min_tensor)
        boundary_Smin_loss = nn.MSELoss()(v_Smin, ones * (K - torch.prod(S_min_tensor[0])))

        total_boundary_loss = 3 * boundary_loss + boundary_Smax_loss + boundary_Smin_loss

        return total_boundary_loss
