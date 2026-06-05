import numpy as np
import torch.nn as nn

from utility.model import ModelConfig

r = 0.1
sigmas = np.array([0.2, 0.3])
rho = 0.5
corr = np.array([
    [1.0, rho],
    [rho, 1.0]
])
K = 1.0
T = 1.0
S_mins = np.array([0.0, 0.0])
S_maxs = np.array([3.0, 3.0])

model_config = ModelConfig(
    input_size=3,
    hidden_sizes=[64, 64, 64, 64],
    output_size=1,
    activation=nn.Tanh(),
    learning_rate=0.001,
    step_size=1000,
    gamma=0.8,
)

loss_weights = {
    'pde': 8,
    'J2': 5,
    'J3': 3,
    'J4': 1,
}
