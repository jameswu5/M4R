import numpy as np


def binomial_tree(S, K, r, sigma, T, n, option_type="put", exercise_type="american"):
    """
    Price an option using a binomial tree.

    Parameters:
    S : float
        Current stock price
    K : float
        Strike price
    r : float
        Risk-free interest rate
    sigma : float
        Volatility of the underlying stock
    T : float
        Time to expiration in years
    n : int
        Number of time steps in the binomial tree
    option_type : str
        'call' for call option, 'put' for put option
    exercise_type : str
        'european' for European option, 'american' for American option
    """

    assert option_type in ['call', 'put'], "option_type must be 'call' or 'put'"
    assert exercise_type in ['european', 'american'], "exercise_type must be 'european' or 'american'"

    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    assert 0 < p < 1, f"Risk-neutral probability p must be between 0 and 1 [params: S={S}, K={K}, r={r}, sigma={sigma}, T={T}, n={n}]"

    # Compute binomial price tree
    # Here price_tree[i, j] = S * u^j * d^(i-j)
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

    # Backward induction to calculate option price
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


class BinomialTree:
    def __init__(self, market_params, n_steps, option_type="put", exercise_type="american"):
        self.market_params = market_params
        self.n_steps = n_steps
        self.option_type = option_type
        self.exercise_type = exercise_type

    def predict(self, t, S):
        """
        Predict the option price at time t and stock price S=S_t using the binomial tree method.

        We can shift by time t, so we price the option with time to maturity T - t and S0=S.
        """

        price, _, _ = binomial_tree(
            S,
            self.market_params.K,
            self.market_params.r,
            self.market_params.sigma,
            self.market_params.T - t,
            self.n_steps,
            self.option_type,
            self.exercise_type
        )
        return price


if __name__ == "__main__":
    S = 100
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1
    n = 100
    option_type = 'put'
    exercise_type = 'american'

    price, pt, ot = binomial_tree(S, K, r, sigma, T, n, option_type, exercise_type)

    print(f"Option Price: {price:.4f}")
