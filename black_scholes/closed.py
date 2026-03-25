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


def delta(S, K, r, sigma, T, option_type="put"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    if option_type == "call":
        return norm.cdf(d1)
    if option_type == "put":
        return norm.cdf(d1) - 1

    raise ValueError("option_type must be 'call' or 'put'")


def gamma(S, K, r, sigma, T, option_type="put"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def theta(S, K, r, sigma, T, option_type="put"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) -
                r * K * np.exp(-r * T) * norm.cdf(d2))
    if option_type == "put":
        return (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) +
                r * K * np.exp(-r * T) * norm.cdf(-d2))

    raise ValueError("option_type must be 'call' or 'put'")


def implied_volatility(price, S, K, r, T, option_type="put", tol=1e-6, max_iterations=1000):
    sigma = 0.2
    for _ in range(max_iterations):
        price_estimate = black_scholes(S, K, r, sigma, T, option_type)
        vega = (S * norm.pdf((np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))) *
                np.sqrt(T))

        price_diff = price_estimate - price
        if abs(price_diff) < tol:
            return sigma

        sigma -= price_diff / vega

    raise ValueError("Implied volatility not found within the maximum number of iterations")


class BlackScholes:
    def __init__(self, K, r, sigma, T, option_type):
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
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
