"""Simulate asset price paths using Geometric Brownian Motion."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp


def simulate_gbm(S0, r, sigma, T, N, n_paths=1, seed=None):
    """
    Simulate a single asset price path using Geometric Brownian Motion (GBM).

    Parameters:
    - S0: Initial asset price (scalar).
    - r: Risk-free interest rate (scalar).
    - sigma: Volatility of the asset (scalar).
    - T: Time horizon (scalar).
    - N: Number of time steps (scalar).
    - n_paths: Number of paths to simulate (scalar).
    - seed: Random seed for reproducibility (scalar or None).
    """
    rng = np.random.default_rng(seed)

    dt = T / N
    Z = rng.standard_normal(size=(n_paths, N))
    dW = np.sqrt(dt) * Z
    S = np.zeros((n_paths, N + 1))
    S[:, 0] = S0
    drift = (r - 0.5 * sigma**2) * dt

    for t in range(N):
        S[:, t+1] = S[:, t] * np.exp(drift + sigma * dW[:, t])

    return S


def correlated_brownian_increments(T, N, corr, n_paths, seed=None):
    rng = np.random.default_rng(seed)

    k = corr.shape[0]
    dt = T / N
    L = np.linalg.cholesky(corr)

    Z = rng.standard_normal(size=(n_paths, N, k))
    dW = np.sqrt(dt) * Z @ L.T
    return dW


def simulate_correlated_gbm(S0, r, sigma, corr, T, N, n_paths=1, seed=None):
    S0 = np.asarray(S0)
    sigma = np.asarray(sigma)
    k = len(S0)
    dt = T / N

    dW = correlated_brownian_increments(T, N, corr, n_paths, seed)

    S = np.zeros((n_paths, N + 1, k))
    S[:, 0, :] = S0
    drift = (r - 0.5 * sigma**2) * dt

    for t in range(N):
        S[:, t+1, :] = S[:, t, :] * np.exp(drift + sigma * dW[:, t, :])

    return S


def test(S0, r, sigma, corr, T=1.0, N=252, n_paths=100, seed=None):
    S = simulate_correlated_gbm(S0, r, sigma, corr, T, N, n_paths, seed)
    S_product = np.prod(S, axis=2)
    S_final = S_product[:, -1]

    # reduce to 1D case
    k = S0.shape[0]
    cov_matrix = corr * np.outer(sigma, sigma)
    S0_1d = np.prod(S0)
    r_1d = r * k + np.sum(np.tril(cov_matrix, k=-1))
    sigma_1d = np.sqrt(np.sum(cov_matrix))
    corr_1d = np.array([[1.0]])

    S_1d = simulate_correlated_gbm([S0_1d], [r_1d], [sigma_1d], corr_1d, T, N, n_paths, seed)
    S_product_1d = S_1d[:, :, 0]
    S_final_1d = S_product_1d[:, -1]

    # ks test on S_product and S_product_1d
    ks_stat, p_value = ks_2samp(S_final, S_final_1d)
    print(f"KS Statistic: {ks_stat}, P-value: {p_value}")

    separate = False
    if separate:
        plt.figure(figsize=(13, 6))
        plt.subplot(1, 2, 1)
        plt.hist(S_final, bins=50, alpha=0.7, color='blue', density=True)
        plt.title('Histogram of Final Product Prices')
        plt.xlabel('Final Product Price')
        plt.ylabel('Density')

        plt.subplot(1, 2, 2)
        plt.hist(S_final_1d, bins=50, alpha=0.7, color='green', density=True)
        plt.title('Histogram of Final Prices (1D)')
        plt.xlabel('Final Price')
        plt.ylabel('Density')
        plt.tight_layout()
    else:
        plt.figure(figsize=(8, 5))
        plt.hist(S_final, bins=50, alpha=0.5, label='Multi-asset Price', color='blue', density=True)
        plt.hist(S_final_1d, bins=50, alpha=0.5, label='1D Price', color='green', density=True)
        plt.title('Histogram of Final Prices')
        plt.xlabel('Final Price')
        plt.ylabel('Density')
        plt.legend()
    plt.show()


if __name__ == "__main__":
    k = 3
    corr = np.array([
        [1.0, 0.6, -0.2],
        [0.6, 1.0, 0.4],
        [-0.2, 0.4, 1.0],
    ])

    S0 = np.array([10, 5, 3])
    sigma = np.array([0.2, 0.3, 0.25])
    r = 0.05
    T = 1.0
    N = 252
    n_paths = 100000
    seed = 42

    test(S0, r, sigma, corr, T=T, N=N, n_paths=n_paths, seed=seed)
