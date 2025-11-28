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


class PutMaxTwoAssets(Payoff):
    def __call__(self, S, K):
        S1 = S[:, 0:1]
        S2 = S[:, 1:2]
        return torch.relu(K - torch.max(S1, S2))

    def boundary_loss(self, model, t_boundary, S_boundary, **kwargs):
        K = kwargs.get('K', None)
        S_max = kwargs.get('S_max', None)
        S_min = kwargs.get('S_min', None)

        if K is None or S_max is None or S_min is None:
            raise ValueError("K, S_max, and S_min must be provided for boundary loss calculation.")

        length = t_boundary.shape[0]
        ones = torch.ones((length, 1))

        S1 = S_boundary[:, 0:1]
        S2 = S_boundary[:, 1:2]

        v_1 = model(ones, S_boundary)
        payoff = self(S_boundary, K)
        boundary_loss = nn.MSELoss()(v_1, payoff)

        # f(t, S_1, S_max) = 0
        v_Smax = model(t_boundary, torch.cat((S1, ones * S_max), dim=1))
        boundary_Smax_loss = nn.MSELoss()(v_Smax, torch.zeros((length, 1)))

        # f(t, S_max, S_2) = 0
        v_Smax_2 = model(t_boundary, torch.cat((ones * S_max, S2), dim=1))
        boundary_Smax_2_loss = nn.MSELoss()(v_Smax_2, torch.zeros((length, 1)))

        # f(t, S_min, S_min) = K - S_min
        v_Smin = model(t_boundary, torch.cat((ones * S_min, ones * S_min), dim=1))
        boundary_Smin_loss = nn.MSELoss()(v_Smin, ones * (K - S_min))

        total_boundary_loss = 3 * boundary_loss + boundary_Smax_loss + boundary_Smax_2_loss + boundary_Smin_loss
        return total_boundary_loss


class PutMinTwoAssets(Payoff):
    def __call__(self, S, K):
        S1 = S[:, 0:1]
        S2 = S[:, 1:2]
        return torch.relu(K - torch.min(S1, S2))

    def boundary_loss(self, model, t_boundary, S_boundary, **kwargs):
        K = kwargs.get('K', None)
        S_max = kwargs.get('S_max', None)
        S_min = kwargs.get('S_min', None)

        if K is None or S_max is None or S_min is None:
            raise ValueError("K, S_max, and S_min must be provided for boundary loss calculation.")

        length = t_boundary.shape[0]
        ones = torch.ones((length, 1))

        S1 = S_boundary[:, 0:1]
        S2 = S_boundary[:, 1:2]

        v_1 = model(ones, S_boundary)
        payoff = self(S_boundary, K)
        boundary_loss = nn.MSELoss()(v_1, payoff)

        # f(t, S_1, S_min) = K - S_min
        v_Smax = model(t_boundary, torch.cat((S1, ones * S_min), dim=1))
        boundary_Smax_loss = nn.MSELoss()(v_Smax, ones * (K - S_min))

        # f(t, S_min, S_2) = K - S_min
        v_Smax_2 = model(t_boundary, torch.cat((ones * S_min, S2), dim=1))
        boundary_Smax_2_loss = nn.MSELoss()(v_Smax_2, ones * (K - S_min))

        # f(t, S_max, S_max) = 0
        v_Smin = model(t_boundary, torch.cat((ones * S_max, ones * S_max), dim=1))
        boundary_Smin_loss = nn.MSELoss()(v_Smin, torch.zeros((length, 1)))

        total_boundary_loss = 3 * boundary_loss + boundary_Smax_loss + boundary_Smax_2_loss + boundary_Smin_loss
        return total_boundary_loss


class PutProductTwoAssets(Payoff):
    def __call__(self, S, K):
        S1 = S[:, 0:1]
        S2 = S[:, 1:2]
        return torch.relu(K - (S1 * S2))

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

        # f(t, S_max, S_max) = 0
        v_Smax = model(t_boundary, torch.cat((ones * S_max, ones * S_max), dim=1))
        boundary_Smax_loss = nn.MSELoss()(v_Smax, torch.zeros((length, 1)))

        # f(t, S_min, S_min) = K - S_min^2
        v_Smin = model(t_boundary, torch.cat((ones * S_min, ones * S_min), dim=1))
        boundary_Smin_loss = nn.MSELoss()(v_Smin, ones * (K - S_min*2))

        total_boundary_loss = 3 * boundary_loss + boundary_Smax_loss + boundary_Smin_loss

        return total_boundary_loss
