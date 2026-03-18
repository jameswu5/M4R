import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from utility.sampler import Sampler


class ModelConfig:
    def __init__(self, input_size: int, hidden_sizes: list, output_size: int, activation: nn.Module, learning_rate: float, dropout=0.0, step_size=500, gamma=0.5):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.step_size = step_size
        self.gamma = gamma


class BaseNetwork(nn.Module):
    def __init__(self, act_fn: nn.Module, input_size: int, output_size: int, hidden_sizes: list, dropout: float):
        super().__init__()

        layers = []
        layers += [nn.Linear(input_size, hidden_sizes[0]), act_fn]
        for layer_index in range(1, len(hidden_sizes)):
            layers += [nn.Linear(hidden_sizes[layer_index - 1], hidden_sizes[layer_index]), act_fn]
            if dropout > 0:
                layers += [nn.Dropout(p=dropout)]
        layers += [nn.Linear(hidden_sizes[-1], output_size)]
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


class PINN(ABC):
    def __init__(self, model_config, seed):
        torch.manual_seed(seed)

        self.model = BaseNetwork(
            act_fn=model_config.activation,
            input_size=model_config.input_size,
            hidden_sizes=model_config.hidden_sizes,
            output_size=model_config.output_size,
            dropout=model_config.dropout
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=model_config.learning_rate
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=model_config.step_size,
            gamma=model_config.gamma
        )

        self.sampler = Sampler(seed=seed)

    @abstractmethod
    def set_params(self, *args, **kwargs):
        pass

    def set_loss_weights(self, loss_weights):
        total_weight = sum(loss_weights.values())
        self.loss_weights = {key: weight / total_weight for key, weight in loss_weights.items()}

    @abstractmethod
    def train(self, batch_size, epochs, early_stopping):
        pass

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

    def predict(self, t, *S):
        self.model.eval()
        with torch.no_grad():
            return self.model(t, *S)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
