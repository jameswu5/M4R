import numpy as np
from scipy.stats import norm


def black_scholes(S, K, r, sigma, T, option_type="put"):
    """
    Compute the Black-Scholes analytical price for a European option.

    Parameters
    ----------
    S : float or array-like
        Current price of the underlying asset.
    K : float
        Strike price of the option.
    r : float
        Risk-free interest rate (annualised).
    sigma : float
        Volatility of the underlying asset (annualised).
    T : float or array-like
        Time to expiry (in years).
    option_type : {'put', 'call'}, optional
        Type of the option (default ``'put'``).

    Returns
    -------
    price : float or ndarray
        Black-Scholes option price.

    Raises
    ------
    ValueError
        If ``option_type`` is not ``'call'`` or ``'put'``.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    if option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    raise ValueError("option_type must be 'call' or 'put'")


def delta(S, K, r, sigma, T, option_type="put"):
    """
    Compute the Black-Scholes delta (first derivative of price w.r.t. S).

    Parameters
    ----------
    S : float or array-like
        Current price of the underlying asset.
    K : float
        Strike price of the option.
    r : float
        Risk-free interest rate (annualised).
    sigma : float
        Volatility of the underlying asset (annualised).
    T : float or array-like
        Time to expiry (in years).
    option_type : {'put', 'call'}, optional
        Type of the option (default ``'put'``).

    Returns
    -------
    delta : float or ndarray
        Rate of change of option price with respect to the underlying price.

    Raises
    ------
    ValueError
        If ``option_type`` is not ``'call'`` or ``'put'``.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    if option_type == "call":
        return norm.cdf(d1)
    if option_type == "put":
        return norm.cdf(d1) - 1

    raise ValueError("option_type must be 'call' or 'put'")


def gamma(S, K, r, sigma, T, option_type="put"):
    """
    Compute the Black-Scholes gamma (second derivative of price w.r.t. S).

    Gamma is identical for calls and puts; ``option_type`` is accepted for
    interface consistency but has no effect on the result.

    Parameters
    ----------
    S : float or array-like
        Current price of the underlying asset.
    K : float
        Strike price of the option.
    r : float
        Risk-free interest rate (annualised).
    sigma : float
        Volatility of the underlying asset (annualised).
    T : float or array-like
        Time to expiry (in years).
    option_type : {'put', 'call'}, optional
        Ignored; present for interface consistency (default ``'put'``).

    Returns
    -------
    gamma : float or ndarray
        Rate of change of delta with respect to the underlying price.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def theta(S, K, r, sigma, T, option_type="put"):
    """
    Compute the Black-Scholes theta (first derivative of price w.r.t. time).

    Returns the rate of change of the option price with respect to calendar
    time (not time-to-expiry), so theta is typically negative.

    Parameters
    ----------
    S : float or array-like
        Current price of the underlying asset.
    K : float
        Strike price of the option.
    r : float
        Risk-free interest rate (annualised).
    sigma : float
        Volatility of the underlying asset (annualised).
    T : float or array-like
        Time to expiry (in years).
    option_type : {'put', 'call'}, optional
        Type of the option (default ``'put'``).

    Returns
    -------
    theta : float or ndarray
        Rate of change of option price with respect to time (per year).

    Raises
    ------
    ValueError
        If ``option_type`` is not ``'call'`` or ``'put'``.
    """
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
    """
    Compute implied volatility via Newton-Raphson root finding.

    Inverts the Black-Scholes formula to find the volatility ``sigma`` that
    reproduces the observed market ``price``.

    Parameters
    ----------
    price : float
        Observed market price of the option.
    S : float
        Current price of the underlying asset.
    K : float
        Strike price of the option.
    r : float
        Risk-free interest rate (annualised).
    T : float
        Time to expiry (in years).
    option_type : {'put', 'call'}, optional
        Type of the option (default ``'put'``).
    tol : float, optional
        Convergence tolerance on the price difference (default ``1e-6``).
    max_iterations : int, optional
        Maximum number of Newton-Raphson iterations (default 1000).

    Returns
    -------
    sigma : float
        Implied volatility (annualised).

    Raises
    ------
    ValueError
        If the algorithm does not converge within ``max_iterations``.
    """
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
        """
        Compute the Black-Scholes price at calendar time ``t`` and asset price ``S``.

        Parameters
        ----------
        t : float or array-like
            Current calendar time (in years); time-to-expiry is ``T - t``.
        S : float or array-like
            Current price of the underlying asset.

        Returns
        -------
        price : float or ndarray
            Black-Scholes option price.
        """
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
