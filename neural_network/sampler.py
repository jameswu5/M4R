import numpy as np
import matplotlib.pyplot as plt
import torch


class Sampler:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def uniform(self, left, right, shape):
        sample = self.rng.uniform(left, right, shape)
        return torch.tensor(sample, dtype=torch.float32)

    def segmented_uniform(self, left, right, centre, radius, weight, shape):
        left_high_density = max(left, centre - radius)
        right_high_density = min(right, centre + radius)
        true_width = right_high_density - left_high_density

        high_density_samples = self.rng.uniform(left_high_density, right_high_density, shape)
        low_density_samples = self.rng.uniform(left, right - true_width, shape)

        low_density_samples = np.where(
            low_density_samples >= left_high_density,
            low_density_samples + true_width,
            low_density_samples
        )

        signal = self.rng.uniform(0, 1, shape)
        samples = np.where(signal < weight, high_density_samples, low_density_samples)
        return torch.tensor(samples, dtype=torch.float32)

    def plot_samples(self, samples):
        plt.hist(samples, bins=100, density=True)
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.show()


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    print(rng.uniform(0, 1, (3, 4)))
