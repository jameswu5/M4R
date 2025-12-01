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
        exercise_loss = nn.MSELoss()(v_1, payoff)

        # f(t, S1, ..., SN) where any Si = Si_max = 0
        boundary_max_losses = []
        n_assets = S_boundary.shape[1]
        for i in range(n_assets):
            S_boundary_max = S_boundary.clone()
            S_boundary_max[:, i] = S_max[i]
            v_Si_max = model(t_boundary, S_boundary_max)
            boundary_loss_Si_max = nn.MSELoss()(v_Si_max, torch.zeros((length, 1)))
            boundary_max_losses.append(boundary_loss_Si_max)

        # f(t, S1, ..., SN) where any Si = Si_min = max(K - prod(S_min), 0)
        boundary_min_losses = []
        for i in range(n_assets):
            S_boundary_min = S_boundary.clone()
            S_boundary_min[:, i] = S_min[i]
            v_Si_min = model(t_boundary, S_boundary_min)
            boundary_loss_Si_min = nn.MSELoss()(v_Si_min, ones * torch.relu(K - torch.prod(S_boundary_min[0])))
            boundary_min_losses.append(boundary_loss_Si_min)

        total_boundary_loss = 3 * exercise_loss + sum(boundary_max_losses) + sum(boundary_min_losses)

        return total_boundary_loss
