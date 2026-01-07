import numpy as np
from scipy.stats import norm


def black_scholes(S, K, r, sigma, T, option_type="put"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    if option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    raise ValueError("option_type must be 'call' or 'put'")


class BlackScholes:
    def __init__(self, market_params, option_type):
        self.K = market_params.K
        self.r = market_params.r
        self.sigma = market_params.sigma
        self.T = market_params.T
        self.option_type = option_type

    def price(self, t, S):
        tau = self.T - t
        return black_scholes(S, self.K, self.r, self.sigma, tau, self.option_type)


if __name__ == "__main__":
    S = 1
    r = 0.1
    sigma = 0.55
    K = 1.0
    T = np.linspace(0, 1, 100)

    price = black_scholes(S, K, r, sigma, T, option_type="call")

    print(price.shape)
