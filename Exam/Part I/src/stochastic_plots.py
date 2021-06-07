# /usr/bin/python
# -*- coding: utf-8 -*-

"""
Plotting utilities for the trajectories of stochastic processes.

@author: <alberto.suarez@uam.es>
         Antonio Coín Castro
"""

from typing import Tuple, Callable, Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def plot_trajectories(
    t: np.ndarray,
    X: np.ndarray,
    max_trajectories: int = 20,
    ax: Optional[matplotlib.axes.Axes] = None,
    fig_num: int = 1,
    fig_size: Tuple[int, int] = (8, 4),
    fontsize: int = 14,
    mean_color: str = 'k',
) -> matplotlib.axes.Axes:
    """
    Plots a sample of simulated trajectories and their mean.

    Parameters
    ----------
    t:
        Array of discretized times.
    X:
        Array of trajectories with shape (num_trajectories, len(t))
    max_trajectories:
        Maximum number of trajectories to plot.
    ax:
        Axis in which to plot.
    fig_num:
        Matplotlib internal parameter for figure number.
    fig_size:
        Size of figure.
    fontsize:
        Font size.
    mean_color:
        Color for the mean function on the plot.

    Returns
    --------
    ax:
        Axis with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, num=fig_num, figsize=fig_size)
    M, _ = np.shape(X)

    # Plot trajectories
    M = np.min((M, max_trajectories))
    ax.plot(t, X[:M, :].T, linewidth=1)
    ax.set_xlabel('t', fontsize=fontsize)
    ax.set_ylabel('X(t)', fontsize=fontsize)
    ax.set_title('Sample of simulated trajectories', fontsize=fontsize)

    # Plot mean
    ax.plot(t, np.mean(X, axis=0), linewidth=3, color=mean_color)

    return ax


def plot_pdf(
    X: np.ndarray,
    pdf: Callable,
    max_bins: int = 50,
    ax: Optional[matplotlib.axes.Axes] = None,
    fig_num: int = 1,
    fig_size: Tuple[int, int] = (4, 4),
    fontsize: int = 14
) -> matplotlib.axes.Axes:
    """
    Compare theoretical pdf with the normalized histogram.

    The normalized histogram is an empirical estimate of the pdf.

    Parameters
    ----------
    X:
        Array of trajectories with shape (num_trajectories, num_points)
    max_bins:
        Maximum number of bins in the histogram.
    ax:
        Axis in which to plot.
    fig_num:
        Matplotlib internal parameter for figure number.
    fig_size:
        Size of figure.
    fontsize:
        Font size.

    Returns
    --------
    ax:
        Axis with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, num=fig_num, figsize=fig_size)
    n_samples = len(X)

    # Plot histogram
    n_bins = np.min((np.int(np.sqrt(n_samples)),
                     max_bins))
    ax.hist(X, bins=n_bins, density=True)
    ax.set_xlabel('x', fontsize=fontsize)
    ax.set_ylabel('pdf(x)', fontsize=fontsize)

    # Compare with exact distribution
    n_plot = 1000
    x_plot = np.linspace(np.min(X), np.max(X), n_plot)
    y_plot = pdf(x_plot)
    ax.plot(x_plot, y_plot, linewidth=2, color='r')

    return ax
