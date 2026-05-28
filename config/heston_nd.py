import numpy as np
import torch.nn as nn

from utility.model import ModelConfig

n_assets = 2
K = 1.0
T = 1.0
r = 0.05
kappa = 2.0
theta = 0.04
sigma_bar = 0.3

sigmas = np.array([0.2, 0.25])

rho_asset = 0  # correlation between assets
corr = np.full((n_assets, n_assets), float(rho_asset))
np.fill_diagonal(corr, 1.0)

rho_cross = [0.3, 0.4]  # stock-variance correlation per asset

S0 = 1.0
v0 = 0.04

S_min = np.full(n_assets, 0.0)
V_min = 0.01
S_max = np.full(n_assets, 3 * S0)
V_max = 5 * v0

model_config = ModelConfig(
    input_size=4,
    hidden_sizes=[64, 64, 64, 64],
    output_size=1,
    # activation=nn.Sigmoid(),
    activation=nn.Tanh(),
    learning_rate=0.001,
    step_size=2000,
    gamma=0.7,
)

loss_weights = {
    'variational': 5,
    'terminal': 5,
    'Smin': 3,
    'Smax': 3,
    'Vmin': 3,
    'Vmax': 3,
}
