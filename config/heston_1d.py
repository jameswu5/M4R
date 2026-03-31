import torch.nn as nn

from utility.model import ModelConfig

K = 1.0
r = 0.1
T = 1.0
kappa = 2.0
theta = 0.04
sigma = 0.3
rho = -0.7

S_min = 0.0
S_max = 3 * K

V_min = 0.0
V_max = 0.8

model_config = ModelConfig(
    input_size=3,
    hidden_sizes=[64, 64, 64],
    output_size=1,
    activation=nn.Sigmoid(),
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
