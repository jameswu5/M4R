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
        Samples uniformly pairs of points (x1, x2) such that |x1 - x2| >= epsilon.

        boundary: if true, BOTH points of each pair are placed on the SAME face
        (same coordinate pinned to the same min/max value). This gives the pair a
        single shared outward normal, so the normal-derivative comparison in the
        J4 spatial Gagliardo seminorm is well-defined, and ||x1 - x2|| reduces to
        the distance within that face (the face coordinate cancels).
        """

        if isinstance(left, Number):
            left = np.full((dimensions,), left)
        if isinstance(right, Number):
            right = np.full((dimensions,), right)

        assert len(left) == dimensions
        assert len(right) == dimensions

        # Rejection-sample in chunks until `batch_size` valid pairs are collected.
        # A fixed over-sample factor is not enough when the epsilon-separation
        # acceptance rate is low (e.g. large epsilon on a small domain such as
        # time in [0, T]); looping guarantees a full batch at any epsilon.
        chunk = int(batch_size * 1.5) + 1
        x1_parts, x2_parts, face_parts = [], [], []
        n_collected = 0

        while n_collected < batch_size:
            x1 = np.column_stack([
                self.rng.uniform(left[d], right[d], chunk) for d in range(dimensions)
            ])
            x2 = np.column_stack([
                self.rng.uniform(left[d], right[d], chunk) for d in range(dimensions)
            ])

            face = None
            if boundary:
                # One shared face (coordinate + side) per pair.
                face = self.rng.integers(0, dimensions, size=chunk)
                side = self.rng.integers(0, 2, size=chunk)
                boundary_val = np.where(side == 0, left[face], right[face])

                rows = np.arange(chunk)
                x1[rows, face] = boundary_val
                x2[rows, face] = boundary_val

            valid = np.where(np.linalg.norm(x1 - x2, axis=1) >= epsilon)[0]
            x1_parts.append(x1[valid])
            x2_parts.append(x2[valid])
            if boundary:
                face_parts.append(face[valid])
            n_collected += len(valid)

        x1 = torch.tensor(np.concatenate(x1_parts)[:batch_size], dtype=torch.float32)
        x2 = torch.tensor(np.concatenate(x2_parts)[:batch_size], dtype=torch.float32)

        face1 = face2 = None
        if boundary:
            # Both points lie on the same face, so they share its index.
            faces = np.concatenate(face_parts)[:batch_size]
            face1 = faces
            face2 = faces

        return x1, x2, face1, face2

    def plot_samples(self, samples):
        plt.hist(samples, bins=100, density=True)
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.show()


if __name__ == "__main__":
    sampler = Sampler(seed=42)
    samples = sampler.truncated_normal_1d(mean=0, std=0.5, left=-1, right=1, batch_size=1000000)
    sampler.plot_samples(samples.numpy())
