import torch
import torch.nn as nn


class BaseNetwork(nn.Module):
    def __init__(self, act_fn, input_size, output_size, hidden_sizes, dropout):
        super().__init__()

        layers = []
        layers += [nn.Linear(input_size, hidden_sizes[0]), act_fn]
        for layer_index in range(1, len(hidden_sizes)):
            layers += [nn.Linear(hidden_sizes[layer_index - 1], hidden_sizes[layer_index]), act_fn]
            if dropout > 0:
                layers += [nn.Dropout(p=dropout)]
        layers += [nn.Linear(hidden_sizes[-1], output_size), act_fn]
        self.layers = nn.Sequential(*layers)

    def forward(self, t, *S_args):
        # Accept python scalars, numpy arrays or tensors for `t`.
        if not torch.is_tensor(t):
            t = torch.as_tensor(t, dtype=torch.float32)
        if t.dim() == 0:
            t = t.view(1, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)

        # Accept either:
        # - a single tensor S of shape (batch, n_assets) passed as the only
        #   positional arg, or
        # - multiple tensors S1, S2, ... each of shape (batch,) or (batch,1).
        if len(S_args) == 0:
            raise ValueError("No asset input provided to the network")

        # Normalize inputs: convert non-tensors to tensors and ensure column shape
        normalized = []
        for s in S_args:
            if not torch.is_tensor(s):
                s = torch.as_tensor(s, dtype=torch.float32)
            if s.dim() == 0:
                s = s.view(1, 1)
            elif s.dim() == 1:
                s = s.unsqueeze(-1)
            normalized.append(s)

        # If a single tensor was passed and it has multiple asset columns,
        # split it into per-asset column tensors.
        if len(normalized) == 1 and normalized[0].dim() == 2 and normalized[0].size(1) > 1:
            S = normalized[0]
            assets = [S[:, i:i+1] for i in range(S.size(-1))]
        else:
            assets = normalized

        # Broadcast scalar rows (batch size 1) to match `t` if needed.
        batch = t.size(0)
        for i, a in enumerate(assets):
            if a.size(0) == 1 and batch > 1:
                assets[i] = a.expand(batch, -1)
            elif a.size(0) != batch:
                raise ValueError(f"Asset tensor batch size {a.size(0)} does not match time batch size {batch}")

        x = torch.cat([t] + assets, dim=-1)
        return self.layers(x)
