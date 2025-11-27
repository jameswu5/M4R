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
        S_max = kwargs.get('S_max', None)
        S_min = kwargs.get('S_min', None)

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
