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

"""
Provide standard filters for TimeSeries.
"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import numpy as np
import scipy as sp
import scipy.signal as sgl
import scipy.ndimage as ndi
import warnings
from ktk import TimeSeries
from typing import *

import ktk  # for doctests


def savgol(tsin: TimeSeries, /, *, window_length: int, poly_order: int,
           deriv: int = 0) -> TimeSeries:
    """
    Apply a Savitzky-Golay filter on a TimeSeries.

    Note
    ----
    If the TimeSeries contains missing samples, a warning is issued, missing
    samples are interpolated using a first-order interpolation before
    filtering, and then replaced by NaNs in the filtered signal.

    Parameters
    ----------
    tsin
        Input TimeSeries
    window_length
        The length of the filter window. window_length must be a positive
        odd integer less or equal than the length of the TimeSeries.
    poly_order
        The order of the polynomial used to fit the samples. polyorder must be
        less than window_length.
    deriv
        Optional. The order of the derivative to compute. The default is 0,
        which means to filter the data without differentiating.

    """
    tsout = tsin.copy()

    delta = tsin.time[1] - tsin.time[0]

    for key in tsout.data.keys():

        input_signal = tsout.data[key]
        # Resample NaNs if they exist:

        # Find NaNs
        signal_shape = np.shape(input_signal)

        n_data = signal_shape[0]
        nan_index = tsout.isnan(key)

        if np.sum(~nan_index) < poly_order + 1:
            # We can't do anything without more points
            warnings.warn('Not enough non-missing samples to filter.')
            continue

        if not np.all(~nan_index):
            # There were NaNs, issue a warning.
            warning_message = ('NaNs found in the signal. They have been ' +
                               'interpolated before filtering, and then put ' +
                               'back in the filtered data.')
            warnings.warn(warning_message)

        original_x = np.arange(n_data)[~nan_index]
        original_y = input_signal[~nan_index]
        new_x = np.arange(n_data)

        # Resample
        f = sp.interpolate.interp1d(original_x, original_y, axis=0,
                                    fill_value='extrapolate')
        input_signal = f(new_x)

        # Filter
        filtered_data = sgl.savgol_filter(input_signal,
                                          window_length, poly_order, deriv,
                                          delta=delta, axis=0)

        # Put back NaNs
        filtered_data[nan_index] = np.nan

        # Assign it to the output
        tsout.data[key] = filtered_data

    return tsout


def smooth(tsin: TimeSeries, /, window_length: int) -> TimeSeries:
    """
    Apply a smoothing (moving average) filter on a TimeSeries.

    Parameters
    ----------
    tsin
        Input TimeSeries.
    window_length
        The length of the filter window. window_length must be a positive
        odd integer less or equal than the length of the TimeSeries.

    """
    tsout = savgol(tsin, window_length=window_length, poly_order=0)
    return tsout


def butter(tsin: TimeSeries, /, fc: float, *, order: int = 2,
           btype: str = 'lowpass', filtfilt: bool = True) -> TimeSeries:
    """
    Apply a Butterworth filter to a TimeSeries.

    Note
    ----
    The sampling rate must be constant.

    Parameters
    ----------
    tsin
        Input TimeSeries.
    fc
        Cut-off frequency in Hz.
    order
        Optional. Order of the filter.
    btype
        Optional. {'lowpass', 'highpass', 'bandpass', 'bandstop'}.
    filtfilt
        Optional. If True, the filter is applied two times in reverse direction
        to eliminate time lag. If False, the filter is applied only in forward
        direction.

    """
    ts = tsin.copy()

    # Create the filter
    fs = (1 / (tsin.time[1] - tsin.time[0]))
    if np.isnan(fs):
        raise ValueError("The TimeSeries' time vector must not contain NaNs.")

    sos = sgl.butter(order, fc, btype, analog=False,
                     output='sos', fs=fs)

    for data in ts.data:

        # Subset
        subts = ts.get_subset(data)

        # Save nans and interpolate
        missing = subts.isnan(data)
        subts.fill_missing_samples(0, method='pchip')

        # Filter
        if filtfilt is True:
            subts.data[data] = sgl.sosfiltfilt(sos, subts.data[data],
                                               axis=0)
        else:
            subts.data[data] = sgl.sosfilt(sos, subts.data[data],
                                           axis=0)

        # Put back nans
        subts.data[data][missing] = np.nan

        # Put back in main TimeSeries
        ts.data[data] = subts.data[data]

    return ts


def deriv(ts: TimeSeries, /, n: int = 1) -> TimeSeries:
    """
    Calculate the nth numerical derivative.

    Note
    ----
    The sample rate must be constant.

    Parameters
    ----------
    ts
        Input timeseries

    n
        Order of the derivative.

    Example
    -------
    >>> ts = ktk.TimeSeries(time=np.arange(0, 0.5, 0.1))
    >>> ts.data['data'] = np.array([0.0, 0.0, 1.0, 1.0, 0.0])

    >>> # Source data
    >>> ts.time
    array([0. , 0.1, 0.2, 0.3, 0.4])
    >>> ts.data['data']
    array([0., 0., 1., 1., 0.])

    >>> # First derivative
    >>> ts1 = ktk.filters.deriv(ts)

    >>> ts1.time
    array([0.05, 0.15, 0.25, 0.35])
    >>> ts1.data['data']
    array([  0.,  10.,   0., -10.])

    >>> # Second derivative
    >>> ts2 = ktk.filters.deriv(ts, n=2)

    >>> ts2.time
    array([0.1, 0.2, 0.3])
    >>> ts2.data['data']
    array([ 100., -100., -100.])

    """
    out_ts = ts.copy()

    for i in range(n):
        out_ts.time = (out_ts.time[1:] + out_ts.time[0:-1]) / 2

    for key in ts.data:
        out_ts.data[key] = sp.diff(
            ts.data[key], n=n, axis=0) / (ts.time[1] - ts.time[0]) ** n

    return out_ts


def median(ts: TimeSeries, /, window_length: int = 3) -> TimeSeries:
    """
    Calculate a moving median.

    Parameters
    ----------
    ts
        Input TimeSeries

    window_length
        Optional. Kernel size, must be odd. The default is 3.

    Example
    -------
    >>> ts = ktk.TimeSeries(time=np.arange(0, 0.5, 0.1))

    >>> # Works on 1-dimension data
    >>> ts.data['data1'] = np.array([10., 11., 11., 20., 14., 15.])

    >>> # and also on n-dimension data
    >>> ts.data['data2'] = np.array( \
            [[0., 10.], \
             [0., 11.], \
             [1., 11.], \
             [1., 20.], \
             [2., 14.], \
             [2., 15.]])

    >>> # Filter
    >>> ts = ktk.filters.median(ts)

    >>> ts.data['data1']
    array([10., 11., 11., 14., 15., 15.])

    >>> ts.data['data2']
    array([[ 0., 10.],
           [ 0., 11.],
           [ 1., 11.],
           [ 1., 14.],
           [ 2., 15.],
           [ 2., 15.]])

    """
    out_ts = ts.copy()
    for key in ts.data:
        window_shape = [1 for i in range(len(ts.data[key].shape))]
        window_shape[0] = window_length
        out_ts.data[key] = ndi.median_filter(
            ts.data[key], size=window_shape)

    return out_ts


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
