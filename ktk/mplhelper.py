#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020 Félix Chénier
#
# This file is not for redistribution.
"""Helper functions for Matplotlib."""

def plot_mean_std(x, y, axis, color, label):
    """
    Plot a mean ± std curve with shaded std area.

    Parameters
    ----------
    x : array
        1-dimension array corresponding to the x axis.

    y : array
        2-dimension array corresponding to the y axis.

    axis : int
        Axis on which the data to average is aligned.

    color : str or list-like
        Color of the curve and shaded area.

    label : str
        Label of the curve, for the legend.

    """
    plt.plot(x, np.nanmean(y, axis=axis), color=color, linewidth=2,
             label=label)

    plt.fill_between(
        x,
        np.nanmean(y, axis=axis) -
        np.nanstd(y, axis=axis),
        np.nanmean(y, axis=axis) +
        np.nanstd(y, axis=axis),
        color=color, alpha=0.1)
