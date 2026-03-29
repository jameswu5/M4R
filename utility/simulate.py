"""Simulate asset price paths using Geometric Brownian Motion."""

import numpy as np


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
    """
    Simulate correlated asset price paths using Geometric Brownian Motion (GBM).

    Parameters:
    - S0: Initial asset prices (array-like of shape (k,)).
    - r: Risk-free interest rate (scalar).
    - sigma: Volatility of the assets (array-like of shape (k,)).
    - corr: Correlation matrix of the assets (array-like of shape (k, k)).
    - T: Time horizon (scalar).
    - N: Number of time steps (scalar).
    - n_paths: Number of paths to simulate (scalar).
    - seed: Random seed for reproducibility (scalar or None).
    """
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


def simulate_heston(S0, V0, r, T, kappa, theta, sigma, rho, N, n_paths=1, seed=None):
    """
    Simulate asset price paths using the Heston model.

    Parameters:
    - S0: Initial asset price (scalar).
    - V0: Initial variance (scalar).
    - r: Risk-free interest rate (scalar).
    - T: Time horizon (scalar).
    - kappa: Speed of mean reversion (scalar).
    - theta: Long-term variance (scalar).
    - sigma: Volatility of variance (scalar).
    - rho: Correlation between asset and variance (scalar).
    - N: Number of time steps (scalar).
    - n_paths: Number of paths to simulate (scalar).
    - seed: Random seed for reproducibility (scalar or None).
    """
    rng = np.random.default_rng(seed)

    dt = T / N
    S = np.zeros((n_paths, N + 1))
    V = np.zeros((n_paths, N + 1))
    S[:, 0] = S0
    V[:, 0] = V0

    for t in range(N):
        Z1 = rng.standard_normal(size=n_paths)
        Z2 = rng.standard_normal(size=n_paths)
        dW1 = np.sqrt(dt) * Z1
        dW2 = np.sqrt(dt) * (rho * Z1 + np.sqrt(1 - rho**2) * Z2)

        V[:, t+1] = np.maximum(V[:, t] + kappa * (theta - V[:, t]) * dt + sigma * np.sqrt(V[:, t]) * dW2, 0)
        S[:, t+1] = S[:, t] * np.exp((r - 0.5 * V[:, t]) * dt + np.sqrt(V[:, t]) * dW1)

    return S, V
