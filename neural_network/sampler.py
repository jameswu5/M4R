import numpy as np
import matplotlib.pyplot as plt
import torch
from numbers import Number


class Sampler:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def uniform(self, left, right, shape):
        sample = self.rng.uniform(left, right, shape)
        return torch.tensor(sample, dtype=torch.float32)

    def segmented_uniform_1d(self, left, right, centre, radius, weight, shape):
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

    def segmented_uniform(self, left, right, centres, radii, weights, batch_size):
        dimensions = len(centres)
        if isinstance(left, Number):
            left = np.ones(dimensions) * left
        if isinstance(right, Number):
            right = np.ones(dimensions) * right
        if isinstance(weights, Number):
            weights = np.ones(dimensions) * weights

        assert len(centres) == len(radii) == len(weights)

        dimensions = len(centres)
        samples = np.zeros((batch_size, dimensions))

        for d in range(dimensions):
            samples[:, d] = self.segmented_uniform_1d(
                left[d], right[d],
                centres[d], radii[d], weights[d],
                (batch_size,)
            ).numpy()

        return torch.tensor(samples, dtype=torch.float32)

    def truncated_normal_1d(self, mean, std, left, right, batch_size):
        samples = []
        while len(samples) < batch_size:
            sample = self.rng.normal(mean, std, batch_size - len(samples))
            sample = sample[(sample >= left) & (sample <= right)]
            samples.extend(sample.tolist())

        result = torch.tensor(samples[:batch_size], dtype=torch.float32)
        return result.view(-1, 1)

    def sample_from_points(self, points, shape):
        indices = self.rng.integers(0, len(points), size=shape)
        sampled_points = points[indices]
        return torch.tensor(sampled_points, dtype=torch.float32)

    def uniform_pair(self, left, right, batch_size, dimensions, epsilon, boundary=False):
        """
        Samples uniformly pairs of points (x1, x2) such that |x1 - x2| < epsilon.

        boundary: if true, for each sample, one of the points is at the boundary
        """

        # Sample bigger batch to account for rejections
        big_batch_size = int(batch_size * 1.2)

        if isinstance(left, Number):
            left = np.full((dimensions,), left)
        if isinstance(right, Number):
            right = np.full((dimensions,), right)

        assert len(left) == dimensions
        assert len(right) == dimensions

        def sample_point():
            x = np.column_stack([
                self.rng.uniform(left[d], right[d], big_batch_size)
                for d in range(dimensions)
            ])

            face = None

            if boundary:
                face = self.rng.integers(0, dimensions, size=big_batch_size)
                side = self.rng.integers(0, 2, size=big_batch_size)
                replace = np.where(side == 0, left[face], right[face])
                x[np.arange(big_batch_size), face] = replace

            return x, face

        x1, face1 = sample_point()
        x2, face2 = sample_point()

        valid_indices = np.where(
            np.linalg.norm(x1 - x2, axis=1) >= epsilon
        )[0]

        valid_indices = valid_indices[:batch_size]

        x1 = torch.tensor(x1[valid_indices], dtype=torch.float32)
        x2 = torch.tensor(x2[valid_indices], dtype=torch.float32)

        face1 = face1[valid_indices] if face1 is not None else None
        face2 = face2[valid_indices] if face2 is not None else None

        return x1, x2, face1, face2

    def plot_samples(self, samples):
        plt.hist(samples, bins=100, density=True)
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.show()

    def sample_near_free_boundary_put(self, K, r, T, t_samples, width_fraction=0.15):
        """
        Sample points near the estimated free boundary for American put options.

        Parameters:
        -----------
        K : float
            Strike price
        r : float
            Risk-free rate
        T : float
            Time to maturity
        t_samples : torch.Tensor
            Time points at which to sample, shape (batch_size, 1)
        width_fraction : float
            Width of the sampling band as fraction of K (default 0.15 = 15%)

        Returns:
        --------
        S_samples : torch.Tensor
            Stock prices sampled near the free boundary, shape (batch_size, 1)
        """
        # Rough estimate of free boundary for American put
        # More accurate: S_f(t) ≈ K * min(1, r/(r + lambda)) where lambda depends on parameters
        # Simpler approximation: S_f(t) ≈ K * (1 - exp(-r*(T-t))) for small r*(T-t)
        # Better approximation: S_f(t) is between K*r/(r+0.5*sigma^2) and K

        # Use conservative estimate that works well in practice
        tau = T - t_samples  # Time to maturity

        # Free boundary is typically below K and increases with time
        # Rough approximation for put: S_f ≈ K * (0.7 + 0.3 * (1 - tau/T))
        S_boundary_estimate = K * (0.7 + 0.3 * (1 - tau / T))

        # Sample in a band around this estimate
        width = width_fraction * K
        S_low = torch.maximum(S_boundary_estimate - width, torch.tensor(0.1 * K))
        S_high = S_boundary_estimate + width

        # Uniform sampling in this band
        S_samples = S_low + (S_high - S_low) * torch.rand_like(t_samples)

        return S_samples


if __name__ == "__main__":
    sampler = Sampler(seed=42)
    samples = sampler.truncated_normal_1d(mean=0, std=0.5, left=-1, right=1, batch_size=1000000)
    sampler.plot_samples(samples.numpy())
