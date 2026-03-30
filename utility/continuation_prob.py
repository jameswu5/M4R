import numpy as np
from utility.simulate import simulate_gbm, simulate_correlated_gbm, simulate_heston, simulate_heston_multi
from scipy.stats import norm


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_continuation_probs(continuation, intrinsics, eps, one=0.99, shift=0):
    """
    Compute continuation probabilities for a set of continuation prices and intrinsic values. These are performed on the
    relative difference between them.

    `continuation` and `intrinsics` must be numpy arrays of the same shape.

    We have eps as a parameter that controls the shape of the continuation probability function, such that
    sigmoid(-eps / tau) = 1 - one and sigmoid(eps / tau) = one

    `shift` is added to prices to adjust the probabilities, defaulted to 0. This is useful for tree / closed form
    prices where we necessarily have price >= intrinsic, for neural networks this may not be the case
    """

    assert continuation.shape == intrinsics.shape, "Continuation and intrinsics must have the same shape."
    assert 0.5 < one < 1, "one must be between 0.5 and 1."

    tau = eps / np.log(one / (1 - one))

    d = (continuation - intrinsics + shift) / intrinsics

    cont_probs = sigmoid(d / tau)
    cont_probs = np.where(intrinsics <= 0, 1.0, cont_probs)  # If the option is out of the money, we always continue

    return cont_probs


def continuation_normal(continuation, stds, intrinsics):
    """
    Compute the continuation probability with the Gaussian model.
    All three inputs must be numpy arrays of the same shape.
    """
    # If option is not in the money, we always continue
    cont_probs = np.where(
        intrinsics <= 0, 1.0,
        np.where(
            stds > 0, norm.cdf((continuation - intrinsics) / stds),  # std > 0 use CDF
            (continuation > intrinsics).astype(float)  # std = 0 use step function
        )
    )

    return cont_probs


def estimate_continuation_value(model, t, S, r, sigma, n_paths=100, h=0.01, seed=None):
    # This is for black scholes model
    S_forward = simulate_gbm(S0=S, r=r, sigma=sigma, T=h, N=1, n_paths=n_paths, seed=seed)[:, -1]
    t_forward = np.full_like(S_forward, t + h)

    f_forward = model(t_forward, S_forward).detach().numpy()
    continuation_values = np.exp(-r * h) * np.mean(f_forward)
    return continuation_values


def estimate_continuation_value_nd(model, t, S, r, sigmas, corr, n_paths=100, h=0.01, seed=None):
    S_forward = simulate_correlated_gbm(S0=S, r=r, sigma=sigmas, corr=corr, T=h, N=1, n_paths=n_paths, seed=seed)[:, -1]
    t_forward = np.full_like(S_forward[:, 0], t + h)

    f_forward = model(t_forward, S_forward).detach().numpy()
    continuation_values = np.exp(-r * h) * np.mean(f_forward)
    return continuation_values


def estimate_contination_value_heston(model, t, S, V, r, kappa, theta, sigma, rho, n_paths=100, h=0.01, seed=None):
    S_forward, V_forward = simulate_heston(S0=S, V0=V, r=r, T=h, kappa=kappa, theta=theta, sigma=sigma, rho=rho, N=1, n_paths=n_paths, seed=seed)
    S_forward = S_forward[:, -1]
    V_forward = V_forward[:, -1]
    t_forward = np.full_like(S_forward, t + h)

    f_forward = model(t_forward, S_forward, V_forward).detach().numpy()
    continuation_values = np.exp(-r * h) * np.mean(f_forward)
    return continuation_values


def estimate_continuation_value_heston_nd(model, t, S, V, r, kappa, theta, sigma_bar, sigmas, corr, rho_cross, n_paths=100, h=0.01, seed=None):
    S_forward, V_forward = simulate_heston_multi(S0=S, V0=V, r=r, T=h, kappa=kappa,
                                                 theta=theta, sigma_bar=sigma_bar, sigmas=sigmas,
                                                 corr=corr, rho_cross=rho_cross, N=1, n_paths=n_paths, seed=seed)
    S_forward = S_forward[:, -1]
    V_forward = V_forward[:, -1]
    t_forward = np.full_like(S_forward[:, 0], t + h)

    f_forward = model(t_forward, S_forward, V_forward).detach().numpy()
    continuation_values = np.exp(-r * h) * np.mean(f_forward)
    return continuation_values
