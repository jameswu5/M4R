"""Simulate asset price paths using Geometric Brownian Motion."""

import numpy as np


def simulate_gbm(S0, r, sigma, T, N, n_paths=1, seed=None):
    """
    Simulate a single asset price path using Geometric Brownian Motion (GBM).

    Parameters
    ----------
    S0 : float
        Initial asset price.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility of the asset.
    T : float
        Time horizon.
    N : int
        Number of time steps.
    n_paths : int, optional
        Number of paths to simulate.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    S : ndarray of shape (n_paths, N + 1)
        Simulated asset price paths.
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
    """
    Generate correlated Brownian motion increments.

    Parameters
    ----------
    T : float
        Time horizon.
    N : int
        Number of time steps.
    corr : array_like of shape (k, k)
        Correlation matrix of the Brownian motions.
    n_paths : int
        Number of paths to simulate.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    dW : ndarray of shape (n_paths, N, k)
        Correlated Brownian increments.
    """
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

    Parameters
    ----------
    S0 : array_like of shape (k,)
        Initial asset prices.
    r : float
        Risk-free interest rate.
    sigma : array_like of shape (k,)
        Volatility of the assets.
    corr : array_like of shape (k, k)
        Correlation matrix of the assets.
    T : float
        Time horizon.
    N : int
        Number of time steps.
    n_paths : int, optional
        Number of paths to simulate.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    S : ndarray of shape (n_paths, N + 1, k)
        Simulated asset price paths.
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

    Parameters
    ----------
    S0 : float
        Initial asset price.
    V0 : float
        Initial variance.
    r : float
        Risk-free interest rate.
    T : float
        Time horizon.
    kappa : float
        Speed of mean reversion.
    theta : float
        Long-term variance.
    sigma : float
        Volatility of variance.
    rho : float
        Correlation between asset and variance.
    N : int
        Number of time steps.
    n_paths : int, optional
        Number of paths to simulate.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    S : ndarray of shape (n_paths, N + 1)
        Simulated asset price paths.
    V : ndarray of shape (n_paths, N + 1)
        Simulated variance paths.
    """

    rng = np.random.default_rng(seed)
    dt = T / N

    S = np.zeros((n_paths, N + 1))
    V = np.zeros((n_paths, N + 1))
    S[:, 0] = S0
    V[:, 0] = V0

    sqrt_dt = np.sqrt(dt)

    for t in range(N):
        Z1 = rng.standard_normal(n_paths)
        Z2 = rng.standard_normal(n_paths)

        dW1 = sqrt_dt * Z1
        dW2 = sqrt_dt * (rho * Z1 + np.sqrt(1 - rho**2) * Z2)

        # Truncate V to ensure non-negativity
        V_pos = np.maximum(V[:, t], 0)
        sqrt_V = np.sqrt(V_pos)

        # Milstein scheme for variance
        V[:, t+1] = np.maximum(
            V[:, t]
            + kappa * (theta - V_pos) * dt
            + sigma * sqrt_V * dW2
            + 0.25 * sigma**2 * (dW2**2 - dt),
            0
        )

        # Log-Euler for asset price
        S[:, t+1] = S[:, t] * np.exp((r - 0.5 * V_pos) * dt + sqrt_V * dW1)

    return S, V


def simulate_heston_multi(
    S0, V0, r, T,
    kappa, theta, sigma_bar,
    sigmas, corr, rho_cross,
    N, n_paths=1, seed=None
):
    """
    Simulate multi-asset price paths under the multi-asset Heston model.

    Parameters
    ----------
    S0 : float or array_like of shape (n,)
        Initial asset prices.
    V0 : float
        Initial variance (shared).
    r : float
        Risk-free rate.
    T : float
        Time horizon.
    kappa : float
        Mean reversion speed.
    theta : float
        Long-run variance.
    sigma_bar : float
        Vol-of-vol.
    sigmas : array_like of shape (n,)
        Per-asset volatility scaling.
    corr : array_like of shape (n, n)
        Asset correlation matrix (corr_ij = rho for i != j).
    rho_cross : array_like of shape (n,)
        Correlation of each W_{1,j} with W_2.
    N : int
        Number of time steps.
    n_paths : int, optional
        Number of Monte Carlo paths.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    S : ndarray of shape (n_paths, N + 1, n)
        Asset price paths.
    V : ndarray of shape (n_paths, N + 1)
        Variance paths.
    """
    n = len(sigmas)
    sigmas = np.asarray(sigmas)
    corr = np.asarray(corr)
    rho_cross = np.asarray(rho_cross)
    dt = T / N

    full_corr = np.eye(n + 1)
    full_corr[:n, n] = rho_cross
    full_corr[n, :n] = rho_cross

    dW  = correlated_brownian_increments(T, N, full_corr, n_paths, seed)
    dW1 = dW[:, :, :n]
    dW2 = dW[:, :,  n]

    ito_diag = np.diag(corr @ corr.T)
    ito_coef = 0.5 * sigmas**2 * ito_diag

    S = np.zeros((n_paths, N + 1, n))
    V = np.zeros((n_paths, N + 1))
    S[:, 0, :] = S0
    V[:, 0]    = V0

    for t in range(N):
        V_pos  = np.maximum(V[:, t], 0)
        sqrt_V = np.sqrt(V_pos)

        # Variance (Milstein)
        V[:, t+1] = np.maximum(
            V[:, t]
            + kappa * (theta - V_pos) * dt
            + sigma_bar * sqrt_V * dW2[:, t]
            + 0.25 * sigma_bar**2 * (dW2[:, t]**2 - dt),
            0
        )

        # Asset diffusion
        diffusion = sqrt_V[:, None] * sigmas[None, :] * (dW1[:, t, :] @ corr.T)

        # Log-Euler for S
        S[:, t+1, :] = S[:, t, :] * np.exp(
            (r - V_pos[:, None] * ito_coef[None, :]) * dt + diffusion
        )

    return S, V
