#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020 Félix Chénier

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Temporary helper functions for Matplotlib. This is not settled at all yet."""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import matplotlib.pyplot as plt
import numpy as np


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
