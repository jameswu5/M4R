import numpy as np
from numbers import Number


class ModelConfig:
    def __init__(self, input_size, hidden_sizes, output_size, activation, learning_rate):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation  # needs to be a torch.nn activation function
        self.learning_rate = learning_rate


class MarketParams:
    def __init__(self, n_assets, S0, r, sigma, K, T, S_min, S_max):
        self.n_assets = n_assets
        self.S0 = self.process(S0)
        self.r = r
        self.sigma = self.process_sigma(sigma)
        self.K = K
        self.T = T
        self.S_min = self.process(S_min)
        self.S_max = self.process(S_max)

    def process(self, parameter):
        if isinstance(parameter, Number):
            return np.full(self.n_assets, parameter)
        elif isinstance(parameter, (list, np.ndarray)):
            parameter = np.array(parameter)
            if parameter.shape[0] != self.n_assets:
                raise ValueError(f"Parameter length {parameter.shape[0]} does not match number of assets {self.n_assets}.")
            return parameter
        else:
            raise TypeError("Parameter must be a number or a list/array of numbers.")
        
    def process_sigma(self, sigma):
        if isinstance(sigma, Number):
            # return 2D array with variances on diagonal
            return np.diag(np.full(self.n_assets, sigma**2))
        elif isinstance(sigma, (list, np.ndarray)):
            sigma = np.array(sigma)
            if sigma.ndim == 1:
                if sigma.shape[0] != self.n_assets:
                    raise ValueError(f"Sigma length {sigma.shape[0]} does not match number of assets {self.n_assets}.")
                return np.diag(sigma**2)
            elif sigma.ndim == 2:
                if sigma.shape != (self.n_assets, self.n_assets):
                    raise ValueError(f"Sigma shape {sigma.shape} is not ({self.n_assets}, {self.n_assets}).")
                return sigma
            else:
                raise ValueError("Sigma must be a 1D or 2D array.")
        else:
            raise TypeError("Sigma must be a number or a list/array of numbers.")
