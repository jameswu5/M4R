import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

class Parameters:
    def __init__(self, r, sigma, K, T):
        self.r = r
        self.sigma = sigma
        self.K = K
        self.T = T

class BaseNetwork(nn.Module):
    def __init__(self, act_fn, input_size, output_size, hidden_sizes):
        super().__init__()

        layers = []
        layers += [nn.Linear(input_size, hidden_sizes[0]), act_fn]
        for layer_index in range(1, len(hidden_sizes)):
            layers += [nn.Linear(hidden_sizes[layer_index - 1], hidden_sizes[layer_index]), act_fn]
        layers += [nn.Linear(hidden_sizes[-1], output_size)]
        self.layers = nn.Sequential(*layers)

    def forward(self, t, S):
        if isinstance(t, (int, float)):
            t = torch.tensor([[t]], dtype=torch.float32)
        if isinstance(S, (int, float)):
            S = torch.tensor([[S]], dtype=torch.float32)
        x = torch.cat((t, S), dim=-1)
        return self.layers(x)


class Sampler:
    def __init__(self, t_min, t_max, S_min, S_max):
        self.t_min = t_min
        self.t_max = t_max
        self.S_min = S_min
        self.S_max = S_max

    def generate(self, N):
        # Uniform sampling for both t and S for now
        t = torch.rand(N, 1) * (self.t_max - self.t_min) + self.t_min
        S = torch.rand(N, 1) * (self.S_max - self.S_min) + self.S_min
        return t, S

    def sample_boundary(self, N):
        # t is fixed, S is uniformly sampled
        t = torch.full((N, 1), self.t_max)
        S = torch.rand(N, 1) * (self.S_max - self.S_min) + self.S_min
        return t, S


def payoff_put(S, K):
    return torch.relu(K - S)

def pde_residual(v_t, v_S, v_SS, v, S, r, sigma):
    return -v_t - r * S * v_S - 0.5 * sigma**2 * S**2 * v_SS + r * v

def train(model, optimiser, params, sampler, max_iterations):

    # Store losses for plotting
    losses = []

    for i in range(max_iterations):
        optimiser.zero_grad()

        # Sample interior points
        t, S = sampler.sample_interior(1000)
        t.requires_grad_(True)
        S.requires_grad_(True)

        v = model(t, S)
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_S = torch.autograd.grad(v, S, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_SS = torch.autograd.grad(v_S, S, grad_outputs=torch.ones_like(v_S), create_graph=True)[0]

        residual = pde_residual(v_t, v_S, v_SS, v, S, params.r, params.sigma) # need to norm?
        pde_loss = torch.min(residual, v - payoff_put(S, params.K)) # inequality
        pde_loss = torch.mean(pde_loss**2) # Take mean squared error

        # Boundary conditions
        num_samples = 1000

        t_b, S_b = sampler.sample_interior(num_samples)
        ones = torch.ones(num_samples, 1)

        v_b = model(ones, S_b)
        payoff = payoff_put(S_b, params.K)
        boundary_loss = nn.MSELoss()(v_b, payoff)

        #f(t, S_max) = 0
        v_Smax = model(t_b, ones * sampler.S_max)
        boundary_Smax_loss = nn.MSELoss()(v_Smax, torch.zeros(num_samples, 1))

        #f(t, S_min) = 0
        v_Smin = model(t_b, ones * sampler.S_min)
        boundary_Smin_loss = nn.MSELoss()(v_Smin, torch.zeros(num_samples, 1))

        # Loss (weights are 1 for now)
        loss = pde_loss + boundary_loss + boundary_Smax_loss + boundary_Smin_loss
        loss.backward()
        optimiser.step()

        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss.item()}")

        losses.append(loss.item())

        if i > 0 and abs(losses[-1] - losses[-2]) < 1e-6:
            print(f"Converged at iteration {i}")
            break

    return model, losses


def plot_losses(losses):
    plt.plot(losses)
    # plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss over Iterations')
    plt.show()


def main():
    set_seed(42)
    
    # Parameters
    r = 0.05
    sigma = 0.2
    K = 1.0
    T = 1.0
    params = Parameters(r, sigma, K, T)

    # Model
    act_fn = nn.ReLU()
    input_size = 2 
    output_size = 1
    hidden_sizes = [32, 32, 32]
    model = BaseNetwork(act_fn, input_size, output_size, hidden_sizes)

    # Sampler
    t_min = 0.0
    t_max = T
    S_min = 0.0
    S_max = 2 * K
    sampler = Sampler(t_min, t_max, S_min, S_max)

    # Optimiser
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train
    max_iterations = 10000
    trained_model, losses = train(model, optimiser, params, sampler, max_iterations)

    # Test
    print(trained_model(0, 0.5).item())
    # Plot losses
    plot_losses(losses)

if __name__ == "__main__":
    main()