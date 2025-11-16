import torch
import torch.nn as nn

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
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if S.dim() == 1:
            S = S.unsqueeze(-1)
        x = torch.cat((t, S), dim=-1)
        return self.layers(x)
