import numpy as np


def build_covariance_matrix(stds, correlations):
    Sigma = correlations * np.outer(stds, stds)
    return Sigma


if __name__ == "__main__":
    stds = np.array([0.4, 0.5])
    correlations = np.array([
        [1.0, 0.3],
        [0.3, 1.0]
    ])
    Sigma = build_covariance_matrix(stds, correlations)
    print("Covariance Matrix:\n", Sigma)
