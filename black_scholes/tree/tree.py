import numpy as np


def binomial_tree(S, K, r, sigma, T, n, option_type="put", exercise_type="american"):
    """
    Price an option using a Cox-Ross-Rubinstein binomial tree.

    Parameters
    ----------
    S : float
        Current asset price.
    K : float
        Strike price.
    r : float
        Risk-free rate (annualised).
    sigma : float
        Volatility (annualised).
    T : float
        Time to expiry (in years).
    n : int
        Number of time steps.
    option_type : {'put', 'call'}, optional
        Type of option (default 'put').
    exercise_type : {'american', 'european'}, optional
        Exercise style (default 'american').

    Returns
    -------
    price : float
        Option price at time 0.
    price_tree : ndarray, shape (n+1, n+1)
        Asset prices at each node.
    option_tree : ndarray, shape (n+1, n+1)
        Option values after backward induction.
    """

    assert option_type in ['call', 'put'], "option_type must be 'call' or 'put'"
    assert exercise_type in ['european', 'american'], "exercise_type must be 'european' or 'american'"

    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    assert 0 < p < 1, f"Risk-neutral probability p must be between 0 and 1 [params: S={S}, K={K}, r={r}, sigma={sigma}, T={T}, n={n}]"

    # Compute binomial price tree
    price_tree = np.zeros((n+1, n+1))
    price_tree[0, 0] = S
    for i in range(1, n+1):
        price_tree[i, :i] = price_tree[i-1, :i] * d
        price_tree[i, i] = price_tree[i-1, i-1] * u

    # Compute option value at maturity
    option_tree = np.zeros((n+1, n+1))
    if option_type == "call":
        option_tree[n, :] = np.maximum(0, price_tree[n, :] - K)
    else:
        option_tree[n, :] = np.maximum(0, K - price_tree[n, :])

    # Backwards induction to calculate option price
    for i in range(n-1, -1, -1):
        # Binomial value
        option_tree[i, :i+1] = np.exp(-r * dt) * (p * option_tree[i+1, 1:i+2] + (1 - p) * option_tree[i+1, 0:i+1])

        # Early exercise for American options
        if exercise_type == "american":
            if option_type == "call":
                exercise_value = np.maximum(0, price_tree[i, :i+1] - K)
            else:
                exercise_value = np.maximum(0, K - price_tree[i, :i+1])

            option_tree[i, :i+1] = np.maximum(option_tree[i, :i+1], exercise_value)

    price = option_tree[0, 0]

    return price, price_tree, option_tree


def binomial_tree_batch(S, K, r, sigma, T, n, option_type="put", exercise_type="american", continuation_value=False):
    """
    Vectorised CRR binomial tree for pricing a batch of options.

    S and T are broadcast against each other; B = max(len(S), len(T)).

    Parameters
    ----------
    S : float or array-like of shape (B,)
        Asset prices, broadcast with T.
    K : float
        Strike price.
    r : float
        Risk-free rate (annualised).
    sigma : float
        Volatility (annualised).
    T : float or array-like of shape (B,)
        Times to expiry (in years), broadcast with S.
    n : int
        Number of time steps.
    option_type : {'put', 'call'}, optional
        Type of option (default 'put').
    exercise_type : {'american', 'european'}, optional
        Exercise style (default 'american').
    continuation_value : bool, optional
        If True, suppress early exercise at the root node and return the
        continuation value (default False).

    Returns
    -------
    prices : ndarray, shape (B,)
        Option prices or continuation values.
    """

    S = np.atleast_1d(np.asarray(S, dtype=float))
    T = np.atleast_1d(np.asarray(T, dtype=float))
    B = max(len(S), len(T))
    if len(S) == 1:
        S = np.broadcast_to(S, (B,))
    if len(T) == 1:
        T = np.broadcast_to(T, (B,))

    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)

    assert np.all((p > 0) & (p < 1))

    # Reshape to (B, 1) for broadcasting against (B, i) tree slices
    d_bc = d[:, None]
    p_bc = p[:, None]
    discount_bc = np.exp(-r * dt)[:, None]

    # Compute binomial price tree
    price_tree = np.zeros((B, n+1, n+1))
    price_tree[:, 0, 0] = S

    for i in range(1, n+1):
        price_tree[:, i, :i] = price_tree[:, i-1, :i] * d_bc
        price_tree[:, i, i] = price_tree[:, i-1, i-1] * u

    # Compute option value at maturity
    option_tree = np.zeros_like(price_tree)
    if option_type == "call":
        option_tree[:, n, :] = np.maximum(0, price_tree[:, n, :] - K)
    else:
        option_tree[:, n, :] = np.maximum(0, K - price_tree[:, n, :])

    # Backwards induction to calculate option price
    for i in range(n-1, -1, -1):
        option_tree[:, i, :i+1] = discount_bc * (
            p_bc * option_tree[:, i+1, 1:i+2]
            + (1 - p_bc) * option_tree[:, i+1, :i+1]
        )

        # Bermudan option
        if continuation_value and i == 0:
            continue

        if exercise_type == "american":
            if option_type == "call":
                exercise_value = np.maximum(0, price_tree[:, i, :i+1] - K)
            else:
                exercise_value = np.maximum(0, K - price_tree[:, i, :i+1])

            option_tree[:, i, :i+1] = np.maximum(
                option_tree[:, i, :i+1], exercise_value
            )

    return option_tree[:, 0, 0]


class BinomialTree:
    """Stateful CRR binomial tree pricer for batch evaluations at varying (t, S)."""

    def __init__(self, K, r, sigma, T, n_steps, option_type="put", exercise_type="american"):
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma

        self.n_steps = n_steps
        self.option_type = option_type
        self.exercise_type = exercise_type

    def predict(self, t, S, continuation_value=False):
        """
        Price the option at calendar time t and asset price S.

        Parameters
        ----------
        t : float or array-like of shape (B,)
            Calendar time in years; time-to-expiry is T - t. Broadcast with S.
        S : float or array-like of shape (B,)
            Asset prices, broadcast with t.
        continuation_value : bool, optional
            If True, return continuation values rather than option prices (default False).

        Returns
        -------
        prices : ndarray, shape (B,)
            Option prices or continuation values.
        """
        tau = self.T - np.asarray(t)
        return binomial_tree_batch(
            S, self.K, self.r, self.sigma,
            tau, self.n_steps, self.option_type, self.exercise_type,
            continuation_value
        )


def test_batch():
    S = 100
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1
    n = 100
    option_type = 'put'
    exercise_type = 'american'

    S_vals = np.array([70, 80, 90, 100, 110, 120, 130])
    t_vals = np.linspace(0, T-1e-2, 20)

    # Test S_vals batch
    single_prices_S = []
    for S_i in S_vals:
        price, _, _ = binomial_tree(S_i, K, r, sigma, T, n, option_type, exercise_type)
        single_prices_S.append(price)
    batch_prices_S = binomial_tree_batch(S_vals, K, r, sigma, T, n, option_type, exercise_type)
    assert np.allclose(single_prices_S, batch_prices_S), "Batch prices do not match single prices"
    print("S Batch pricing test passed.")

    # Test t_vals batch
    single_prices_t = []
    for t_i in t_vals:
        price, _, _ = binomial_tree(S, K, r, sigma, T - t_i, n, option_type, exercise_type)
        single_prices_t.append(price)
    batch_prices_t = binomial_tree_batch(S, K, r, sigma, T - t_vals, n, option_type, exercise_type)
    assert np.allclose(single_prices_t, batch_prices_t), "Batch prices do not match single prices"
    print("t Batch pricing test passed.")


if __name__ == "__main__":
    test_batch()
