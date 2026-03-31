import torch.nn as nn

from utility.model import ModelConfig

r = 0.1
sigma = 0.3
K = 1.0
T = 1.0
S_min = 0.0
S_max = 3 * K

model_config = ModelConfig(
    input_size=2,
    hidden_sizes=[32, 32, 32],
    output_size=1,
    activation=nn.Sigmoid(),
    learning_rate=0.001,
    step_size=2000,
    gamma=0.7,
)
