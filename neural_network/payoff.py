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

    @abstractmethod
    def sobolev_loss(self, model, **kwargs):
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

    def sobolev_loss(self, model, S_interior, t1_interior, t2_interior, S_boundary, **kwargs):
        K = kwargs.get('K', None)
        a = kwargs.get('a', None)

        S_interior.requires_grad_(True)
        length = S_interior.shape[0]
        zeros = torch.zeros((length, 1))
        ones = torch.ones((length, 1))

        # w2 loss ||f(0, .) - g(.)||^2_H1
        v2 = model(zeros, S_interior)
        payoff = self(S_interior, K)
        value_loss = nn.MSELoss()(v2, payoff)
        v_S = torch.autograd.grad(v2 - value_loss, S_interior, grad_outputs=torch.ones_like(v2), create_graph=True)[0]
        derivative_loss = nn.MSELoss()(v_S, zeros)
        w2 = value_loss + derivative_loss

        # w3 loss ||f(t, x) - g(x)||^2_H3/4,3/2
        v3 = model(t1_interior, S_boundary)
        payoff_boundary = self(S_boundary, K)
        value_loss3 = nn.MSELoss()(v3, payoff_boundary)
        # fractional loss
        integrand_num_1 = model(t1_interior, ones * a) - model(t2_interior, ones * a)
        integrand_num_2 = model(t1_interior, ones / a) - model(t2_interior, ones / a)
        integrand_denom = torch.abs(t1_interior - t2_interior) ** (5/2)
        fractional_loss_3 = torch.mean((integrand_num_1 ** 2 + integrand_num_2 ** 2) / integrand_denom)

        w3 = value_loss3 + fractional_loss_3

        # w4 loss ||d/dx (f(t, x) - g(x))||^2_H1/4,1/2
        integrand2_num = (model(t1_interior, ones * a) - model(t1_interior, ones / a) - self(ones * a, K) + self(ones / a, K)) ** 2
        integrand2_denom = 2 * (a**2 + 1/a**2)
        first_fractional_loss_4 = torch.mean(integrand2_num / integrand2_denom)

        integrand_denom_4 = torch.abs(t1_interior - t2_interior) ** (3/2)
        fractional_loss_4 = torch.mean((integrand_num_1 ** 2 + integrand_num_2 ** 2) / integrand_denom_4)

        w4 = value_loss3 + first_fractional_loss_4 + fractional_loss_4

        return w2 + w3 + w4



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

    def sobolev_loss(self, model):
        raise NotImplementedError("Sobolev loss is not implemented for PutProductMultipleAssets.")
