"""New training script, old one was too complicated"""

import numpy as np
import matplotlib.pyplot as plt
import torch

from .model import BaseNetwork
from .sampler import Sampler
from .losses import bs_residual


class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def reset(self):
        self.counter = 0
        self.best_loss = float('inf')

    def step(self, loss):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


class BlackScholesPINN:
    def __init__(self, model_config, seed):
        """
        PINN for solving the 1D Black-Scholes PDE for American put options.
        """

        self.model = BaseNetwork(
            act_fn=model_config.activation,
            input_size=model_config.input_size,
            hidden_sizes=model_config.hidden_sizes,
            output_size=model_config.output_size,
            dropout=model_config.dropout
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.model_config.learning_rate
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=500,
            gamma=0.5
        )

        self.sampler = Sampler(seed=seed)

        self.history = {
            'loss': [],
            'variational_loss': [],
            'terminal_loss': [],
            'Smin_loss': [],
            'Smax_loss': []
        }

    def set_params(self, K, r, sigma, T, S_min, S_max):
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.S_min = S_min
        self.S_max = S_max

    def set_loss_weights(self, loss_weights):
        # Normalise loss weights to sum to 1
        total_weight = sum(loss_weights.values())
        self.loss_weights = {key: weight / total_weight for key, weight in loss_weights.items()}

    def train(self, batch_size, epochs, early_stopping):
        """
        Train the PINN model.

        Args:
            batch_size (int): Number of samples per training batch.
            epochs (int): Maximum number of training epochs.
            min_delta (float): Minimum change in loss to qualify as an improvement for early stopping.
        """
        for i in range(epochs):
            variational_loss = self.__interior_loss(batch_size)
            terminal_loss, Smin_loss, Smax_loss = self.__boundary_loss(batch_size)

            loss = self.__process_loss(variational_loss, terminal_loss, Smin_loss, Smax_loss)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Compute validation loss for early stopping
            variational_loss_val = self.__interior_loss(batch_size)
            terminal_loss_val, Smin_loss_val, Smax_loss_val = self.__boundary_loss(batch_size)
            val_loss = self.__process_loss(variational_loss_val, terminal_loss_val, Smin_loss_val, Smax_loss_val, update_dict=False)

            if i % 100 == 0:
                print(f"Iteration {i} | Training Loss: {loss.item()} | Validation Loss: {val_loss.item()}")

            if early_stopping.step(val_loss.item()):
                print(f"Early stopping at epoch {i}")
                break

    def __process_loss(self, variational_loss, terminal_loss, Smin_loss, Smax_loss, update_dict=True):
        variational_loss *= self.loss_weights['variational']
        terminal_loss *= self.loss_weights['terminal']
        Smin_loss *= self.loss_weights['Smin']
        Smax_loss *= self.loss_weights['Smax']

        loss = variational_loss + terminal_loss + Smin_loss + Smax_loss

        if update_dict:
            self.history['loss'].append(loss.item())
            self.history['variational_loss'].append(variational_loss.item())
            self.history['terminal_loss'].append(terminal_loss.item())
            self.history['Smin_loss'].append(Smin_loss.item())
            self.history['Smax_loss'].append(Smax_loss.item())

        return loss

    def __interior_loss(self, batch_size):
        t, S = self.__sample_interior(batch_size)

        pde_residual = bs_residual(self.model, t, S, self.K, self.r, self.sigma)
        f = self.model(t, S)
        g = self.payoff(S, self.K)

        if self.exercise_type == 'american':
            variational_loss = torch.mean(
                torch.minimum(pde_residual, f - g) ** 2
            )
        else:
            variational_loss = torch.mean(pde_residual ** 2)

        return variational_loss

    def __boundary_loss(self, batch_size):
        t, S = self.__sample_boundary(batch_size)

        S_inf = self.S_max * 10

        zeros = torch.zeros((batch_size, 1))
        ones = torch.ones((batch_size, 1))

        # Terminal condition: f(T, S) = max(K - S, 0)
        f_T = self.model(ones * self.T, S)
        g_T = torch.maximum(self.K - S, zeros)
        terminal_loss = torch.mean((f_T - g_T) ** 2)

        # S_inf loss: f(t, S_inf) = 0
        f_inf = self.model(t, ones * S_inf)
        Smax_loss = torch.mean(f_inf ** 2)

        # S_min loss: f(t, 0) = K
        f_min = self.model(t, zeros)
        Smin_loss = torch.mean((f_min - self.K) ** 2)

        return terminal_loss, Smin_loss, Smax_loss

    def __sample_interior(self, batch_size):
        t = self.sampler.uniform(0, self.T, (batch_size, 1))
        S = self.sampler.uniform(self.S_min, self.S_max, (batch_size, 1))
        return t, S

    def __sample_boundary(self, batch_size):
        t = self.sampler.uniform(0, self.T, (batch_size, 1))
        S = self.sampler.uniform(self.S_min, self.S_max, (batch_size, 1))
        return t, S

    def plot_losses(self, start_epoch=0, detailed=False):
        x = range(start_epoch, len(self.history['loss']))
        for key in self.history:
            if (key == 'loss') ^ (detailed):  # one or the other but not both = xor
                plt.plot(x, self.history[key][start_epoch:], label=key)

        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        title = 'Total Loss' if not detailed else 'Loss Components'
        plt.title(title)
        plt.legend()
        plt.show()
