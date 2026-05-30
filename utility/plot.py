"""
Useful plotting functions
"""

import numpy as np
import matplotlib.pyplot as plt
from config.plot_settings import figsize3, cmap


def plot_price_heatmaps(prices1, prices2, **kwargs):
    """
    Plots 3 heatmaps: prices1, prices2 and the difference.
    """
    xlabel = kwargs.get("xlabel", None)
    ylabel = kwargs.get("ylabel", None)
    title1 = kwargs.get("title1", None)
    title2 = kwargs.get("title2", None)
    title_diff = kwargs.get("title_diff", None)
    label1 = kwargs.get("label1", None)
    label2 = kwargs.get("label2", None)
    label_diff = kwargs.get("label_diff", None)
    extent = kwargs.get("extent", None)
    save_path = kwargs.get("save_path", None)

    plt.figure(figsize=figsize3)

    plt.subplot(1, 3, 1)
    vmax = np.abs(prices1).max()
    vmin = -vmax
    im = plt.imshow(prices1, extent=extent, aspect='auto', origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, label=label1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title1)

    plt.subplot(1, 3, 2)
    vmax = np.abs(prices2).max()
    vmin = -vmax
    im2 = plt.imshow(prices2, extent=extent, aspect='auto', origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im2, label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title2)

    plt.subplot(1, 3, 3)
    diff = prices1 - prices2

    vmax = np.abs(diff).max()
    vmin = -vmax
    im3 = plt.imshow(diff, extent=extent, aspect='auto', origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im3, label=label_diff)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title_diff)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()
