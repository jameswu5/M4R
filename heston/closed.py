import numpy as np
from scipy.integrate import quad


def characteristic_function(u, S0, r, T, kappa, theta, sigma, rho, v0, j):
    i = 1j
    a = kappa * theta
    b = kappa - rho * sigma if j == 1 else kappa
    d = np.sqrt((rho * sigma * i * u - b)**2 - sigma**2 * ((3 - 2*j) * i * u - u**2))
    g = (b - rho * sigma * i * u + d) / (b - rho * sigma * i * u - d)

    C = r * i * u * T + a / sigma**2 * ((b - rho * sigma * i * u + d) * T - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g)))
    D = (b - rho * sigma * i * u + d) / sigma**2 * ((1 - np.exp(d * T)) / (1 - g * np.exp(d * T)))

    return np.exp(C + D * v0 + i * u * np.log(S0))


def heston_call_price(S0, K, T, r, kappa, theta, sigma, rho, v0):
    def integrand(u, j):
        phi = characteristic_function(u, S0, r, T, kappa, theta, sigma, rho, v0, j)
        return np.real(np.exp(-1j * u * np.log(K)) * phi / (1j * u))

    P1 = 0.5 + (1 / np.pi) * quad(lambda u: integrand(u, 1), 0, 100)[0]
    P2 = 0.5 + (1 / np.pi) * quad(lambda u: integrand(u, 2), 0, 100)[0]

    return S0 * P1 - K * np.exp(-r * T) * P2


def heston_closed_price(S0, K, T, r, kappa, theta, sigma, rho, v0, option_type='call'):
    call_price = heston_call_price(S0, K, T, r, kappa, theta, sigma, rho, v0)
    if option_type == "call":
        return call_price
    elif option_type == "put":
        return call_price - S0 + K * np.exp(-r * T)
    else:
        raise ValueError(f"Payoff type ({option_type}) is not valid")


class HestonClosed:
    def __init__(self, K, T, r, kappa, theta, sigma, rho):
        self.K = K
        self.T = T
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho

    def price(self, V, S, t=0, option_type='put'):
        """
        Prices a European option under the Heston model using the closed-form solution.

        V: variance at time t
        S: underlying price at time t
        t: time elapsed since time 0 (default 0)
        """
        T = self.T - t
        return heston_closed_price(S, self.K, T, self.r, self.kappa, self.theta, self.sigma, self.rho, V, option_type)


if __name__ == "__main__":
    S0 = 100.0    # Initial stock price
    K = 100.0     # Strike price
    T = 1.0       # Time to maturity
    r = 0.05      # Risk-free rate
    kappa = 2.0   # Mean reversion rate
    theta = 0.04  # Long-term variance
    sigma = 0.3   # Volatility of variance
    rho = -0.7    # Correlation
    v0 = 0.04     # Initial variance
    price = heston_call_price(S0, K, T, r, kappa, theta, sigma, rho, v0)
    print(f"Heston model European call option price: {price:.4f}")
