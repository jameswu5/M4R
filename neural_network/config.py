
class ModelConfig:
    def __init__(self, input_size, hidden_sizes, output_size, activation, learning_rate):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation  # needs to be a torch.nn activation function
        self.learning_rate = learning_rate


class MarketParams:
    def __init__(self, r, sigma, K, T, S_min, S_max):
        self.r = r
        self.sigma = sigma
        self.K = K
        self.T = T
        self.S_min = S_min
        self.S_max = S_max
