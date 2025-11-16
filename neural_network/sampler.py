import torch

class Sampler:
    def __init__(self, t_min, t_max, S_min, S_max):
        self.t_min = t_min
        self.t_max = t_max
        self.S_min = S_min
        self.S_max = S_max

    def generate(self, N):
        # Uniform sampling for both t and S for now
        t = torch.rand(N, 1) * (self.t_max - self.t_min) + self.t_min
        S = torch.rand(N, 1) * (self.S_max - self.S_min) + self.S_min
        return t, S