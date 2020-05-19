#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020 Félix Chénier
#
# This file is not for redistribution.
"""
Standard filters for TimeSeries.
"""

import numpy as np
import scipy as sp
import scipy.signal as sgl
import scipy.ndimage as ndi
import warnings


def savgol(tsin, window_length, poly_order, deriv=0):
    """
    Apply a Savitzky-Golay filter on a TimeSeries.

    Parameters
    ----------
    tsin : TimeSeries
        Input TimeSeries
    window_length : int
        The length of the filter window. window_length must be a positive
        odd integer less or equal than the length of the TimeSeries.
    poly_order : int
        The order of the polynomial used to fit the samples. polyorder must be
        less than window_length.
    deriv : int (optional)
        The order of the derivative to compute. This must be a nonnegative
        integer. The default is 0, which means to filter the data without
        differentiating.

    Returns
    -------
    tsout : TimeSeries
        The filtered TimeSeries

    If the TimeSeries contains missing samples, a warning is issued, missing
    samples are interpolated using a first-order interpolation before
    filtering, and then replaced by NaNs in the filtered signal.

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


def smooth(tsin, window_length):
    """
    Apply a smoothing (moving average) filter on a TimeSeries.

    Parameters
    ----------
    tsin : TimeSeries
        Input TimeSeries.
    window_length : int
        The length of the filter window. window_length must be a positive
        odd integer less or equal than the length of the TimeSeries.

    Returns
    -------
    tsout : TimeSeries
        The filtered TimeSeries

    """
    tsout = savgol(tsin, window_length, 0)
    return tsout


def butter(tsin, fc, order=2, btype='lowpass', filtfilt=True):
    """
    Apply a Butterworth filter to a TimeSeries.

    The sampling rate must be constant.

    Parameters
    ----------
    tsin : TimeSeries
        Input TimeSeries.
    fc : float
        Cut-off frequency.
    order : int (optional)
        Order of the filter. The default is 2.
    btype : {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}, optional
        The default is 'lowpass'.
    filtfilt : bool (optional)
        If True, the filter is applied two times in reverse direction to
        eliminate time lag. If False, the filter is applied only in forward
        direction. The default is True.

    Returns
    -------
    tsout : TimeSeries
        The filtered TimeSeries

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


def deriv(ts, n=1):
    """
    Calculate the nth numerical derivative.

    Parameters
    ----------
    ts : TimeSeries
        Input timeseries

    n : int (optional)
        Order of the derivative. The default is 1.

    Returns
    -------
    ts : TimeSeries
        A copy of the TimeSeries where each data key has been derivated n
        times.

    Notes
    -----
    The sample rate must be constant.

    Example
    -------
        >>> import ktk, numpy as np
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


def median(ts, window_length=3):
    """
    Calculate a moving median.

    Parameters
    ----------
    ts : TimeSeries
        Input TimeSeries

    window_length : int (optinal)
        Kernel size, must be odd. The default is 3.

    Returns
    -------
    ts : TimeSeries
        A copy of the input TimeSeries with filtered data.

    Example
    -------
        >>> import ktk, numpy as np
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
