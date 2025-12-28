import numpy as np
import matplotlib.pyplot as plt


def build_covariance_matrix(stds, correlations):
    Sigma = correlations * np.outer(stds, stds)
    return Sigma


def plot_relative_error(true_values, predicted_values, eps, **kwargs):
    mask = np.abs(true_values) > eps
    relative_errors = np.where(mask, np.abs((true_values - predicted_values) / true_values), np.nan)
    im = plt.imshow(relative_errors, extent=[kwargs.get('x_min', 0), kwargs.get('x_max', 1), kwargs.get('y_min', 0), kwargs.get('y_max', 1)],
                    aspect='auto', origin='lower', cmap='RdBu_r', vmin=kwargs.get('vmin', None), vmax=kwargs.get('vmax', None))
    plt.colorbar(im, label=kwargs.get('label', 'Relative Error'))
    plt.xlabel(kwargs.get('xlabel', None))
    plt.ylabel(kwargs.get('ylabel', None))
    plt.title(kwargs.get('title', None))
    plt.show()


if __name__ == "__main__":
    stds = np.array([0.4, 0.5])
    correlations = np.array([
        [1.0, 0.3],
        [0.3, 1.0]
    ])
    Sigma = build_covariance_matrix(stds, correlations)
    print("Covariance Matrix:\n", Sigma)
