import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from .model import BaseNetwork
from .sampler import Sampler
from .losses import compute_derivatives_nd, pde_residual_nd, heston_residual


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
            seed=seed
        )

        self.history = {
            'loss': []
        }

    def set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)

    @abstractmethod
    def train(self, batch_size, epochs, tol=1e-3):
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
    def __init__(self, model_config, market_params, payoff, loss_weights=None, seed=None):
        super().__init__(model_config, market_params, payoff, seed)

        self.loss_weights = loss_weights if loss_weights is not None else {
            'pde': 1.0,
            'exercise': 1.0,
            'boundary_Smax': 1.0,
            'boundary_Smin': 1.0
        }

        self.history = {
            'loss': [],
            'pde_loss': [],
            'exercise_loss': [],
            'boundary_Smax_loss': [],
            'boundary_Smin_loss': []
        }

    def train(self, batch_size, epochs, tol=1e-3):
        for i in range(epochs):
            self.optimizer.zero_grad()

            t_interior, S_interior = self.sample_interior_points(batch_size)
            pde_loss = self.get_pde_loss(t_interior, S_interior)

            t_boundary, S_boundary = self.sample_boundary_points(batch_size)
            boundary_loss, boundary_Smax_loss, boundary_Smin_loss = self.get_boundary_loss(t_boundary, S_boundary)

            pde_loss *= self.loss_weights['pde']
            boundary_loss *= self.loss_weights['exercise']
            boundary_Smax_loss *= self.loss_weights['boundary_Smax']
            boundary_Smin_loss *= self.loss_weights['boundary_Smin']

            loss = pde_loss + boundary_loss + boundary_Smax_loss + boundary_Smin_loss

            loss.backward()
            self.optimizer.step()

            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss.item()}")

            self.history['loss'].append(loss.item())
            self.history['pde_loss'].append(pde_loss.item())
            self.history['exercise_loss'].append(boundary_loss.item())
            self.history['boundary_Smax_loss'].append(boundary_Smax_loss.item())
            self.history['boundary_Smin_loss'].append(boundary_Smin_loss.item())

            if i > 0 and abs(self.history['loss'][-1] - self.history['loss'][-2]) < tol:
                print(f"Converged at epoch {i}")
                break

    def sample_interior_points(self, num_samples):
        t_interior = self.sampler.uniform(0, self.market_params.T, (num_samples, 1))

        if self.dimension == 1:
            S_interior = self.sampler.segmented_uniform_1d(
                self.market_params.S_min, self.market_params.S_max,
                centre=self.market_params.S0, radius=0.1 * self.market_params.S0,
                weight=0.4, shape=(num_samples, self.dimension),
            )

            # std = (self.market_params.S_max - self.market_params.S_min) / 3

            # S_interior = self.sampler.truncated_normal_1d(
            #     mean=self.market_params.S0, std=std,
            #     left=self.market_params.S_min,
            #     right=self.market_params.S_max,
            #     batch_size=num_samples
            # )

        else:
            # S_interior = self.sampler.uniform(self.market_params.S_min, self.market_params.S_max, (num_samples, self.dimension))
            S_interior = self.sampler.segmented_uniform(
                left=self.market_params.S_min, right=self.market_params.S_max,
                centres=self.market_params.S0, radii=0.1 * self.market_params.S0,
                weights=0.7, batch_size=num_samples
            )
        return t_interior, S_interior

    def sample_boundary_points(self, num_samples):
        t_boundary = self.sampler.uniform(0, self.market_params.T, (num_samples, 1))
        S_boundary = self.sampler.uniform(self.market_params.S_min, self.market_params.S_max, (num_samples, self.dimension))
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
                                         S_max=self.market_params.S_max,
                                         S_min=self.market_params.S_min)

    def plot_losses_detailed(self):
        # plt.plot(self.history['loss'], label='Total Loss')
        plt.plot(self.history['pde_loss'], label='PDE Loss')
        plt.plot(self.history['exercise_loss'], label='Exercise Loss')
        plt.plot(self.history['boundary_Smax_loss'], label='Boundary Smax Loss')
        plt.plot(self.history['boundary_Smin_loss'], label='Boundary Smin Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss Components over Iterations')
        plt.legend()
        plt.show()


class SobolevTrainer(NeuralNetworkTrainer):
    def __init__(self, model_config, market_params, payoff, loss_weights=None, seed=None):
        super().__init__(model_config, market_params, payoff, seed)

        self.loss_weights = loss_weights if loss_weights is not None else {
            'pde': 1.0,
            'J2': 1.0,
            'J3': 1.0,
            'J4': 1.0
        }

        self.history = {
            'loss': [],
            'pde_loss': [],
            'J2_loss': [],
            'J3_loss': [],
            'J4_loss': []
        }

    def train(self, batch_size, epochs, tol=1e-3):

        K = self.market_params.K
        S_min = self.market_params.S_min
        S_max = self.market_params.S_max
        n_assets = self.market_params.n_assets

        for i in range(epochs):
            self.optimizer.zero_grad()

            t1_interior, t2_interior, _, _ = self.sampler.uniform_pair(0, self.market_params.T, batch_size, 1, epsilon=0.01, boundary=False)

            if n_assets == 1:
                # a = S_max[0]
                S_interior = self.sampler.uniform(S_min, S_max, (batch_size, 1))
                J2, J3, J4 = self.payoff.sobolev_loss(self.model,
                                                      t1_interior=t1_interior, t2_interior=t2_interior,
                                                      S_min=S_min, S_max=S_max,
                                                      S_interior=S_interior,
                                                      K=K)
            else:
                S_interior = self.sampler.uniform(S_min, S_max, (batch_size, n_assets))
                S1_boundary, S2_boundary, face1, face2 = self.sampler.uniform_pair(S_min, S_max, batch_size, n_assets, epsilon=0.01, boundary=True)
                J2, J3, J4 = self.payoff.sobolev_loss(self.model,
                                                      t1_interior=t1_interior, t2_interior=t2_interior,
                                                      S_interior=S_interior,
                                                      S1_boundary=S1_boundary, S2_boundary=S2_boundary,
                                                      S1_face=face1, S2_face=face2,
                                                      a=S_min, b=S_max, K=K)

            pde_loss = self.get_pde_loss(t1_interior, S_interior)

            pde_loss *= self.loss_weights['pde']
            J2 *= self.loss_weights['J2']
            J3 *= self.loss_weights['J3']
            J4 *= self.loss_weights['J4']

            loss = pde_loss + J2 + J3 + J4

            loss.backward()
            self.optimizer.step()

            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss.item()}")

            self.history['loss'].append(loss.item())
            self.history['pde_loss'].append(pde_loss.item())
            self.history['J2_loss'].append(J2.item())
            self.history['J3_loss'].append(J3.item())
            self.history['J4_loss'].append(J4.item())

            if i > 0 and abs(self.history['loss'][-1] - self.history['loss'][-2]) < tol:
                print(f"Converged at epoch {i}")
                break

    def get_pde_loss(self, t_interior, S_interior):
        v, v_t, v_S, H = compute_derivatives_nd(self.model, t_interior, S_interior)
        r = self.market_params.r
        Sigma = self.market_params.sigma
        residual = pde_residual_nd(v, v_t, v_S, H, S_interior, r, Sigma)
        pde_loss = torch.min(residual, v - self.payoff(S_interior, self.market_params.K))
        pde_loss = torch.mean(pde_loss**2)
        return pde_loss

    def plot_losses_detailed(self):
        # plt.plot(self.history['loss'], label='Total Loss')
        plt.plot(self.history['pde_loss'], label='PDE Loss')
        plt.plot(self.history['J2_loss'], label='J2 Loss')
        plt.plot(self.history['J3_loss'], label='J3 Loss')
        plt.plot(self.history['J4_loss'], label='J4 Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss Components over Iterations')
        plt.legend()
        plt.show()


class HestonTrainer(NeuralNetworkTrainer):
    def __init__(self, model_config, heston_params, payoff, loss_weights=None, seed=None):
        super().__init__(model_config, heston_params, payoff, seed)

        self.loss_weights = loss_weights if loss_weights is not None else {
            'pde': 1.0,
            'payoff': 1.0,
            'S_min': 1.0,
            'S_max': 1.0,
            'V_min': 1.0,
            'V_max': 1.0
        }

        self.history = {
            'loss': [],
            'pde_loss': [],
            'payoff_loss': [],
            'S_min_loss': [],
            'S_max_loss': [],
            'V_min_loss': [],
            'V_max_loss': []
        }

    def train(self, batch_size, epochs, tol=1e-5):
        for i in range(epochs):
            self.optimizer.zero_grad()

            t, S, V = self.sample_points(batch_size)
            pde_loss = self.get_pde_loss(t, S, V)
            payoff_loss, S_min_loss, S_max_loss, V_min_loss, V_max_loss = self.get_boundary_loss(t, S, V)

            pde_loss *= self.loss_weights['pde']
            payoff_loss *= self.loss_weights['payoff']
            S_min_loss *= self.loss_weights['S_min']
            S_max_loss *= self.loss_weights['S_max']
            V_min_loss *= self.loss_weights['V_min']
            V_max_loss *= self.loss_weights['V_max']

            loss = pde_loss + payoff_loss + S_min_loss + S_max_loss + V_min_loss + V_max_loss

            loss.backward()
            self.optimizer.step()

            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss.item()}")

            self.history['loss'].append(loss.item())
            self.history['pde_loss'].append(pde_loss.item())
            self.history['payoff_loss'].append(payoff_loss.item())
            self.history['S_min_loss'].append(S_min_loss.item())
            self.history['S_max_loss'].append(S_max_loss.item())
            self.history['V_min_loss'].append(V_min_loss.item())
            self.history['V_max_loss'].append(V_max_loss.item())

            if i > 0 and abs(self.history['loss'][-1] - self.history['loss'][-2]) < tol:
                print(f"Converged at epoch {i}")
                break

    def sample_points(self, num_samples):
        t = self.sampler.uniform(0, self.market_params.T, (num_samples, 1))
        # S = self.sampler.uniform(0, self.market_params.S_max, (num_samples, 1))
        S = self.sampler.segmented_uniform_1d(
            0, self.market_params.S_max, self.market_params.S0, 0.1 * self.market_params.S0,
            0.5, (num_samples, 1)
        )
        V = self.sampler.uniform(0, self.market_params.V_max, (num_samples, 1))
        return t, S, V

    def get_pde_loss(self, t, S, V):
        # t, S, V are all interior points
        residual = heston_residual(self.model, t, S, V,
                                   r=self.market_params.r,
                                   kappa=self.market_params.kappa,
                                   theta=self.market_params.theta,
                                   sigma=self.market_params.sigma,
                                   rho=self.market_params.rho)

        pde_loss = torch.mean(residual**2)
        return pde_loss

    def get_boundary_loss(self, t, S, V):
        # t, S, V are all interior points

        # Hardcode the call option in for now
        K = self.market_params.K
        S_max = self.market_params.S_max
        V_max = self.market_params.V_max
        T = self.market_params.T

        ones = torch.ones_like(t)
        zeros = torch.zeros_like(t)

        # J2
        payoff_loss = torch.mean((
            self.model(ones * T, S, V) - torch.maximum(S - K * ones, zeros)
        )**2)

        # J3
        S_min_loss = torch.mean((
            self.model(t, zeros, V)
        )**2)

        # J4
        S_max_tensor = (ones * S_max).requires_grad_(True)
        f_Smax = self.model(t, S_max_tensor, V)
        df_Smax_dS = torch.autograd.grad(
            f_Smax, S_max_tensor, grad_outputs=torch.ones_like(f_Smax), create_graph=True, retain_graph=True
        )[0]

        S_max_loss = torch.mean((
            df_Smax_dS - ones
        )**2)

        # J5
        V_min = zeros.requires_grad_(True)
        f_Vmin = self.model(t, S, V_min)
        df_dt = torch.autograd.grad(
            f_Vmin, t, grad_outputs=torch.ones_like(f_Vmin), create_graph=True, retain_graph=True
        )[0]
        df_dS = torch.autograd.grad(
            f_Vmin, S, grad_outputs=torch.ones_like(f_Vmin), create_graph=True, retain_graph=True
        )[0]
        df_dV = torch.autograd.grad(
            f_Vmin, V_min, grad_outputs=torch.ones_like(f_Vmin), create_graph=True, retain_graph=True
        )[0]
        r = self.market_params.r
        kappa = self.market_params.kappa
        theta = self.market_params.theta

        V_min_loss = torch.mean((
            r * S * df_dS + kappa * theta * df_dV - r * f_Vmin + df_dt
        )**2)

        V_max_loss = torch.mean((
            self.model(t, S, ones * V_max) - S
        )**2)

        return payoff_loss, S_min_loss, S_max_loss, V_min_loss, V_max_loss

    def plot_losses_detailed(self):
        plt.plot(self.history['pde_loss'], label='PDE Loss')
        plt.plot(self.history['payoff_loss'], label='Payoff Loss')
        plt.plot(self.history['S_min_loss'], label='S min Loss')
        plt.plot(self.history['S_max_loss'], label='S max Loss')
        plt.plot(self.history['V_min_loss'], label='V min Loss')
        plt.plot(self.history['V_max_loss'], label='V max Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss Components over Iterations')
        plt.legend()
        plt.show()
