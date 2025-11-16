import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from model import BaseNetwork
from sampler import Sampler
from losses import compute_derivatives, pde_residual


class Parameters:
    def __init__(self, r, sigma, K, T):
        self.r = r
        self.sigma = sigma
        self.K = K
        self.T = T


class NeuralNetworkTrainer:
    def __init__(self, params, payoff, seed):
        self.params = params
        self.payoff = payoff

        self.set_seed(seed)

        self.act_fn = nn.ReLU()
        self.model = BaseNetwork(
            act_fn=self.act_fn,
            input_size=2,
            output_size=1,
            hidden_sizes=[32, 32, 32]
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.sampler = Sampler(
            t_min=0.0,
            t_max=1.0,
            S_min=0.0,
            S_max=2.0
        )

        self.history = {
            'loss': []
        }


    def set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)

    def train(self, max_iterations):
        for i in range(max_iterations):
            self.optimizer.zero_grad()

            # Sample interior points
            t_interior, S_interior = self.sampler.generate(N=1000)

            # Compute derivatives
            v, v_t, v_S, v_SS = compute_derivatives(self.model, t_interior, S_interior)

            # Compute PDE residual
            residual = pde_residual(
                v_t, v_S, v_SS, v,
                S_interior,
                self.params.r,
                self.params.sigma
            )
            pde_loss = torch.min(residual, v - self.payoff(S_interior, self.params.K))
            pde_loss = torch.mean(pde_loss**2)

            # Boundary conditions
            num_samples = 1000
            t_b, S_b = self.sampler.generate(num_samples)
            ones = torch.ones(num_samples, 1)

            v_b = self.model(ones, S_b)
            payoff = self.payoff(S_b, self.params.K)
            boundary_loss = nn.MSELoss()(v_b, payoff)

            #f(t, S_max) = 0
            v_Smax = self.model(t_b, ones * self.sampler.S_max)
            boundary_Smax_loss = nn.MSELoss()(v_Smax, torch.zeros(num_samples, 1))

            #f(t, S_min) = 0
            v_Smin = self.model(t_b, ones * self.sampler.S_min)
            boundary_Smin_loss = nn.MSELoss()(v_Smin, torch.zeros(num_samples, 1))

            # Loss (weights are 1 for now)
            loss = pde_loss + boundary_loss + boundary_Smax_loss + boundary_Smin_loss
            loss.backward()
            self.optimizer.step()

            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss.item()}")

            self.history['loss'].append(loss.item())

            if i > 0 and abs(self.history['loss'][-1] - self.history['loss'][-2]) < 1e-6:
                print(f"Converged at iteration {i}")
                break

        return self.history
    
    def plot_losses(self):
        plt.plot(self.history['loss'])
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss over Iterations')
        plt.show()

    def predict(self, t, S):
        return self.model(t, S)