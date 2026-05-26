import numpy as np
from scipy.integrate import quad


def characteristic_function(omega, S0, r, T, kappa, theta, sigma, rho, v0, j):
    """
    Heston characteristic function phi_j(u) for j=1 or j=2.
    """
    i = 1j
    u = 1/2 if j == 1 else -1/2
    a = kappa * theta
    b = kappa - rho * sigma if j == 1 else kappa
    d = np.sqrt((rho * sigma * i * omega - b)**2 - sigma**2 * (2 * u * i * omega - omega**2))
    g = (b - rho * sigma * i * omega - d) / (b - rho * sigma * i * omega + d)  # heston trap

    C = r * i * omega * T + a / sigma**2 * ((b - rho * sigma * i * omega - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
    D = (b - rho * sigma * i * omega - d) / sigma**2 * ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))

    return np.exp(C + D * v0 + i * omega * np.log(S0))


def heston_call_price(S0, K, T, r, kappa, theta, sigma, rho, v0):
    """
    Heston European call price by semi-analytical integration of the characteristic function.
    """
    def integrand(omega, j):
        phi = characteristic_function(omega, S0, r, T, kappa, theta, sigma, rho, v0, j)
        return np.real(np.exp(-1j * omega * np.log(K)) * phi / (1j * omega))

    P1 = 0.5 + (1 / np.pi) * quad(lambda omega: integrand(omega, 1), 0, 100)[0]
    P2 = 0.5 + (1 / np.pi) * quad(lambda omega: integrand(omega, 2), 0, 100)[0]

    return S0 * P1 - K * np.exp(-r * T) * P2


def heston_closed_price(S0, K, T, r, kappa, theta, sigma, rho, v0, option_type='call'):
    """
    Heston European call or put price, deriving the put via put-call parity.

    Parameters
    ----------
    S0 : float
        Current asset price.
    K : float
        Strike price.
    T : float
        Time to expiry (in years).
    r : float
        Risk-free rate (annualised).
    kappa : float
        Mean reversion speed of the variance process.
    theta : float
        Long-run mean of the variance process.
    sigma : float
        Volatility of variance (vol-of-vol).
    rho : float
        Correlation between asset and variance Brownian motions.
    v0 : float
        Initial variance.
    option_type : {'call', 'put'}, optional
        Type of option (default 'call').

    Returns
    -------
    price : float
        European option price.
    """
    call_price = heston_call_price(S0, K, T, r, kappa, theta, sigma, rho, v0)
    if option_type == "call":
        return call_price
    elif option_type == "put":
        return call_price - S0 + K * np.exp(-r * T)
    else:
        raise ValueError(f"Payoff type ({option_type}) is not valid")


class HestonClosed:
    """Heston model pricer for European options with fixed parameters."""

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
        Evaluate the Heston European option price at variance V, asset price S, and calendar time t.

        Parameters
        ----------
        V : float
            Variance at time t.
        S : float
            Asset price at time t.
        t : float, optional
            Calendar time in years; time-to-expiry is T - t (default 0).
        option_type : {'put', 'call'}, optional
            Type of option (default 'put').

        Returns
        -------
        price : float
            European option price.
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
