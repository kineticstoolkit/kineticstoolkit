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
        # Resample NaNs if the exist:

        # Find NaNs
        signal_shape = np.shape(input_signal)

        n_data = signal_shape[0]
        nan_index = tsout.isnan(key)

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
    # Create the filter
    fs = (1 / (tsin.time[1] - tsin.time[0]))
    sos = sgl.butter(order, fc, btype, analog=False,
                     output='sos', fs=fs)

    ts = tsin.copy()
    for data in ts.data:

        # Subset
        subts = ts.get_subset(data)

        # Save nans and interpolate
        missing = subts.isnan(data)
        subts.fill_missing_samples(0, method='cubic')

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
