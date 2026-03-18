import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def p_cont(d, tau1, tau2):
    tau = np.where(d < 0, tau1, tau2)
    return sigmoid(d / tau)


def compute_continuation_probs(prices, intrinsics, eps1, eps2, one=0.99, shift=0):
    """
    Compute continuation probabilities for a set of prices and intrinsic values. These are performed on the
    relative difference between them.

    Prices and intrinsics must be numpy arrays of the same shape.

    We have eps1 and eps2 as parameters that control the shape of the continuation probability function, such that
    sigmoid(-eps1 / tau1) = 1 - one and sigmoid(eps2 / tau2) = one

    `shift` is added to prices to adjust the probabilities, defaulted to 0. This is useful for tree / closed form
    prices where we necessarily have price >= intrinsic, for neural networks this may not be the case
    """

    assert prices.shape == intrinsics.shape, "Prices and intrinsics must have the same shape."
    assert 0.5 < one < 1, "one must be between 0.5 and 1."

    tau1 = eps1 / np.log(one / (1 - one))
    tau2 = eps2 / np.log(one / (1 - one))

    d = (prices - intrinsics + shift) / intrinsics

    cont_probs = p_cont(d, tau1, tau2)
    cont_probs = np.where(intrinsics <= 0, 1.0, cont_probs)  # If the option is out of the money, we always continue

    return cont_probs


def continuation_normal(prices, stds, intrinsics):
    """
    Compute the continuation probability with the Gaussian model.
    All three inputs must be numpy arrays of the same shape.
    """
    # If option is not in the money, we always continue
    cont_probs = np.where(
        intrinsics <= 0, 1.0,
        np.where(
            stds > 0, norm.cdf((prices - intrinsics) / stds),  # std > 0 use CDF
            (prices > intrinsics).astype(float)  # std = 0 use step function
        )
    )

    return cont_probs


def plot_p_cont(tau1, tau2):
    x = np.linspace(-4, 4, 500)
    y = p_cont(x, tau1, tau2)
    plt.plot(x, y)
    plt.title(f'p_cont with tau1={tau1} and tau2={tau2}')
    plt.xlabel('x')
    plt.ylabel('p_cont(x)')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    tau1 = 0.1
    tau2 = 1
    plot_p_cont(tau1, tau2)
