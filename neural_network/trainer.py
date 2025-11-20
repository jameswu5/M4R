import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from .model import BaseNetwork
from .sampler import Sampler
from .losses import compute_derivatives, pde_residual
from .config import MarketParams, ModelConfig


class NeuralNetworkTrainer:
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

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=model_config.learning_rate)

        self.sampler = Sampler(
            t_min=0.0,
            t_max=market_params.T,
            S_min=market_params.S_min,
            S_max=market_params.S_max
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

            # Sample interior points
            t_interior, S_interior = self.sampler.generate(mode="segmented_uniform", N=num_samples,
                                                           S_centre=self.market_params.K,
                                                           radius=(self.market_params.S_max - self.market_params.S_min) / 6,
                                                           weight=0.5)

            # Compute derivatives
            v, v_t, v_S, v_SS = compute_derivatives(self.model, t_interior, S_interior)

            # Compute PDE residual
            residual = pde_residual(
                v_t, v_S, v_SS, v,
                S_interior,
                self.market_params.r,
                self.market_params.sigma
            )
            pde_loss = torch.min(residual, v - self.payoff(S_interior, self.market_params.K))
            pde_loss = torch.mean(pde_loss**2)

            # Boundary conditions (these are hardcoded for now)
            t_b, S_b = self.sampler.generate(mode="uniform", N=num_samples)
            ones = torch.ones(num_samples, 1)

            v_b = self.model(ones, S_b)
            payoff = self.payoff(S_b, self.market_params.K)
            boundary_loss = nn.MSELoss()(v_b, payoff)

            # f(t, S_max) = 0
            v_Smax = self.model(t_b, ones * self.sampler.S_max)
            boundary_Smax_loss = nn.MSELoss()(v_Smax, torch.zeros(num_samples, 1))

            # f(t, S_min) = K - S_min
            v_Smin = self.model(t_b, ones * self.sampler.S_min)
            boundary_Smin_loss = nn.MSELoss()(v_Smin, ones * (self.market_params.K - self.sampler.S_min))

            # Loss (weights are 1 for now)
            loss = (pde_loss + boundary_loss + boundary_Smax_loss + boundary_Smin_loss) / 4
            loss.backward()
            self.optimizer.step()

            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss.item()}")

            self.history['loss'].append(loss.item())

            if i > 0 and abs(self.history['loss'][-1] - self.history['loss'][-2]) < tol:
                print(f"Converged at iteration {i}")
                break

    def plot_losses(self):
        plt.plot(self.history['loss'])
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss over Iterations')
        plt.show()

    def predict(self, t, S):
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.float32).view(-1, 1)
        if not torch.is_tensor(S):
            S = torch.tensor(S, dtype=torch.float32).view(-1, 1)
        return self.model(t, S)
