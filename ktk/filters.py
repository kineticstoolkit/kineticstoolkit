#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply standard filters on TimeSeries.

Author: Félix Chénier
Started on Aug 1st, 2019.
"""

import numpy as np
import scipy.signal as sgl


def savgol(tsin, window_length, poly_order, deriv=0):
    """
    Apply a Savitzky-Golay filter on a TimeSeries.

    Parameters
    ----------
    tsin : ktk.TimeSeries
        Input TimeSeries
    window_length : int
        The length of the filter window. window_length must be a positive
        odd integer less or equal than the length of the TimeSeries.
    poly_order : int
        The order of the polynomial used to fit the samples. polyorder must be
        less than window_length.
    deriv : int, optional
        The order of the derivative to compute. This must be a nonnegative
        integer. The default is 0, which means to filter the data without
        differentiating.

    Returns
    -------
    tsout : ktk.TimeSeries
        The filtered TimeSeries

    """
    tsout = tsin.copy()

    delta = tsin.time[1] - tsin.time[0]

    for key in tsout.data.keys():
        filtered_data = sgl.savgol_filter(tsout.data[key],
                                          window_length, poly_order, deriv,
                                          delta=delta)
        tsout.data[key] = filtered_data

    return tsout


def smooth(tsin, window_length):
    """
    Apply a smoothing (moving average) filter on a TimeSeries.

    Parameters
    ----------
    tsin : ktk.TimeSeries
        Input TimeSeries
    window_length : int
        The length of the filter window. window_length must be a positive
        odd integer less or equal than the length of the TimeSeries.

    Returns
    -------
    tsout : ktk.TimeSeries
        The filtered TimeSeries

    """
    tsout = savgol(tsin, window_length, 0)
    return tsout
