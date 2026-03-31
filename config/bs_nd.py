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
    activation=nn.Sigmoid(),
    learning_rate=0.001,
    step_size=2000,
    gamma=0.7,
)

loss_weights = {
    'variational': 1,
    'terminal': 1,
    'Smax': 1,
    'Smin': 1,
}
