import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from .model import BaseNetwork
from .sampler import Sampler
from .losses import compute_derivatives_nd, pde_residual_nd


class NeuralNetworkTrainer(ABC):
    def __init__(self, model_config, market_params, payoff, seed):
        self.model_config = model_config
        self.market_params = market_params
        self.payoff = payoff
        self.set_seed(seed)

        self.model = BaseNetwork(
            act_fn=model_config.activation,
            input_size=model_config.input_size,
            output_size=model_config.output_size,
            hidden_sizes=model_config.hidden_sizes
        )

        # Number of assets
        self.dimension = model_config.input_size - 1  # assuming first input is time

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=model_config.learning_rate)

        self.sampler = Sampler(
            t_min=0.0,
            t_max=market_params.T,
            S_min=market_params.S_min,
            S_max=market_params.S_max,
            seed=seed
        )

        self.history = {
            'loss': []
        }

    def set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)

    def train(self, num_samples, max_iterations, tol=1e-3):
        for i in range(max_iterations):
            self.optimizer.zero_grad()

            t_interior, S_interior = self.sample_interior_points(num_samples)
            pde_loss = self.get_pde_loss(t_interior, S_interior)

            t_boundary, S_boundary = self.sample_boundary_points(num_samples)
            boundary_losses = self.get_boundary_loss(t_boundary, S_boundary)

            loss = pde_loss + boundary_losses

            loss.backward()
            self.optimizer.step()

            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss.item()}")

            self.history['loss'].append(loss.item())

            if i > 0 and abs(self.history['loss'][-1] - self.history['loss'][-2]) < tol:
                print(f"Converged at iteration {i}")
                break

    @abstractmethod
    def sample_interior_points(self, num_samples):
        pass

    @abstractmethod
    def sample_boundary_points(self, num_samples):
        pass

    @abstractmethod
    def get_pde_loss(self, t_interior, S_interior):
        pass

    @abstractmethod
    def get_boundary_loss(self, t_boundary, S_boundary):
        pass

    def plot_losses(self):
        plt.plot(self.history['loss'])
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss over Iterations')
        plt.show()

    def predict(self, t, *S):
        return self.model(t, *S)


class GeneralTrainer(NeuralNetworkTrainer):
    def __init__(self, model_config, market_params, payoff, seed):
        super().__init__(model_config, market_params, payoff, seed)

    def sample_interior_points(self, num_samples):
        t_interior, S_interior = self.sampler.generate(mode="uniform", n_samples=num_samples)
        return t_interior, S_interior

    def sample_boundary_points(self, num_samples):
        t_boundary, S_boundary = self.sampler.generate(mode="uniform", n_samples=num_samples)
        return t_boundary, S_boundary

    def get_pde_loss(self, t_interior, S_interior):
        v, v_t, v_S, H = compute_derivatives_nd(self.model, t_interior, S_interior)
        r = self.market_params.r
        Sigma = self.market_params.sigma
        residual = pde_residual_nd(v, v_t, v_S, H, S_interior, r, Sigma)
        pde_loss = torch.min(residual, v - self.payoff(S_interior, self.market_params.K))
        pde_loss = torch.mean(pde_loss**2)
        return pde_loss

    def get_boundary_loss(self, t_boundary, S_boundary):
        return self.payoff.boundary_loss(self.model, t_boundary, S_boundary,
                                         K=self.market_params.K,
                                         S_max=self.sampler.S_max,
                                         S_min=self.sampler.S_min)
