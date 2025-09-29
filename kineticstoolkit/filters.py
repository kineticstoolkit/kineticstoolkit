#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020-2025 Félix Chénier

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Provide standard filters for TimeSeries."""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2025 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import numpy as np
import scipy.signal as sgl
import scipy.ndimage as ndi
import warnings
from kineticstoolkit import TimeSeries
from kineticstoolkit.typing_ import check_param
from typing import cast

import kineticstoolkit as ktk  # For doctests


def __dir__():
    return ["savgol", "smooth", "butter", "deriv", "median"]


def _interpolate(ts: TimeSeries, key: str) -> tuple[TimeSeries, np.ndarray]:
    """Interpolate NaNs in a given data key in a TimeSeries."""
    ts = ts.get_subset(key)
    nan_index = ts.isnan(key)

    if not np.all(~nan_index):
        # There were NaNs, issue a warning.
        ts = ts.fill_missing_samples(0)
        warnings.warn(
            "NaNs found in the signal. They have been "
            "interpolated before filtering, and then put "
            "back in the filtered data."
        )
    return (ts, nan_index)


def _validate_input(ts):
    """Check that time is not null, that sample rate is constant, and that time
    unit is s."""
    if ts.time.shape[0] == 0:
        raise ValueError("There is no data to filter.")
    if np.isnan(ts.get_sample_rate()):
        raise ValueError("Sample rate must be constant.")
    try:
        assert ts.info["Time"]["Unit"] == "s"
    except KeyError:
        warnings.warn("I assume, but cannot tell if the time unit is 's'.")
    except ValueError:
        warnings.warn("It seems that unit is not 's'.")


def savgol(
    ts: TimeSeries, /, *, window_length: int, poly_order: int, deriv: int = 0
) -> TimeSeries:
    """
    Apply a Savitzky-Golay filter on a TimeSeries.

    Filtering occurs on the first axis (time). If the TimeSeries contains
    missing samples, a warning is issued, missing samples are interpolated
    using a first-order interpolation before filtering, and then replaced by
    np.nan in the filtered signal.

    Parameters
    ----------
    ts
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

    Returns
    -------
    TimeSeries
        A copy of the input TimeSeries, which each data being filtered.

    Raises
    ------
    ValueError
        If sample rate is not constant, or if there is no data to
        filter.

    See Also
    --------
    ktk.filters.smooth

    """
    check_param("ts", ts, TimeSeries)
    check_param("window_length", window_length, int)
    check_param("poly_order", poly_order, int)
    check_param("deriv", deriv, int)
    _validate_input(ts)

    tsout = ts.copy()

    delta = ts.time[1] - ts.time[0]

    for key in tsout.data.keys():
        (subts, nan_index) = _interpolate(tsout, key)

        if np.sum(~nan_index) < poly_order + 1:
            # We can't do anything without more points
            warnings.warn(f"Not enough non-missing samples to filter {key}.")
            continue

        input_signal = subts.data[key]

        # Filter
        filtered_data = sgl.savgol_filter(
            input_signal, window_length, poly_order, deriv, delta=delta, axis=0
        )

        # Put back NaNs
        filtered_data[nan_index] = np.nan

        # Assign it to the output
        tsout.data[key] = filtered_data

    return tsout


def smooth(ts: TimeSeries, /, window_length: int) -> TimeSeries:
    """
    Apply a smoothing (moving average) filter on a TimeSeries.

    Filtering occurs on the first axis (time). If the TimeSeries contains
    missing samples, a warning is issued, missing samples are interpolated
    using a first-order interpolation before filtering, and then replaced by
    np.nan in the filtered signal.

    Parameters
    ----------
    ts
        Input TimeSeries.
    window_length
        The length of the filter window. window_length must be a positive
        odd integer less or equal than the length of the TimeSeries.

    Returns
    -------
    TimeSeries
        A copy of the input TimeSeries, which each data being filtered.

    Raises
    ------
    ValueError
        If sample rate is not constant, or if there is no data to
        filter.

    See Also
    --------
    ktk.filters.savgol

    """
    check_param("ts", ts, TimeSeries)
    check_param("window_length", window_length, int)
    _validate_input(ts)

    tsout = savgol(ts, window_length=window_length, poly_order=0)
    return tsout


def butter(
    ts: TimeSeries,
    /,
    fc: float | tuple[float, float],
    *,
    order: int = 2,
    btype: str = "lowpass",
    filtfilt: bool = True,
) -> TimeSeries:
    """
    Apply a Butterworth filter to a TimeSeries.

    Filtering occurs on the first axis (time). If the TimeSeries contains
    missing samples, a warning is issued, missing samples are interpolated
    using a first-order interpolation before filtering, and then replaced by
    np.nan in the filtered signal.

    Parameters
    ----------
    ts
        Input TimeSeries.
    fc
        Cut-off frequency in Hz. This is a float for single-frequency filters
        (lowpass, highpass), or a tuple of two floats (e.g., (10., 13.)
        for two-frequency filters (bandpass, bandstop)).
    order
        Optional. Order of the filter. Default is 2.
    btype
        Optional. Can be either "lowpass", "highpass", "bandpass" or
        "bandstop". Default is "lowpass".
    filtfilt
        Optional. If True, the filter is applied two times in reverse direction
        to eliminate time lag. If False, the filter is applied only in forward
        direction. Default is True.

    Returns
    -------
    TimeSeries
        A copy of the input TimeSeries, which each data being filtered.

    Raises
    ------
    ValueError
        If sample rate is not constant, or if there is no data to
        filter.

    """
    check_param("ts", ts, TimeSeries)
    try:
        check_param("fc", fc, float)
    except TypeError:
        try:
            fc = cast(tuple[float, float], fc)
            fc = cast(tuple[float, float], tuple(fc))
            check_param("fc", fc, tuple, length=2, contents_type=float)
        except TypeError:
            raise TypeError("fc must be an integer or a tuple or 2 floats.")
    check_param("order", order, int)
    check_param("btype", btype, str)
    check_param("filtfilt", filtfilt, bool)
    _validate_input(ts)

    ts = ts.copy()

    # Create the filter
    fs = 1 / (ts.time[1] - ts.time[0])
    if np.isnan(fs):
        raise ValueError("The TimeSeries' time vector must not contain NaNs.")

    sos = sgl.butter(order, fc, btype, analog=False, output="sos", fs=fs)

    for data in ts.data:
        (subts, missing) = _interpolate(ts, data)

        # Filter
        if filtfilt is True:
            subts.data[data] = sgl.sosfiltfilt(sos, subts.data[data], axis=0)
        else:
            subts.data[data] = sgl.sosfilt(sos, subts.data[data], axis=0)

        # Put back nans
        subts.data[data][missing] = np.nan

        # Put back in main TimeSeries
        ts.data[data] = subts.data[data]

    return ts


def deriv(ts: TimeSeries, /, n: int = 1) -> TimeSeries:
    """
    Calculate the nth numerical derivative.

    Filtering occurs on the first axis (time). The sample rate must be
    constant.

    Parameters
    ----------
    ts
        Input timeseries

    n
        Order of the derivative.

    Returns
    -------
    TimeSeries
        A copy of the input TimeSeries, which each data being derived. The
        length of the resulting TimeSeries is one less than `ts`.

    Raises
    ------
    ValueError
        If sample rate is not constant, or if there is no data to
        filter.

    Example
    -------
    >>> ts = ktk.TimeSeries(time=np.arange(0, 0.5, 0.1))
    >>> ts = ts.add_data("test", np.array([0.0, 0.0, 1.0, 1.0, 0.0]))

    >>> # Source data
    >>> ts.time
    array([0. , 0.1, 0.2, 0.3, 0.4])
    >>> ts.data["test"]
    array([0., 0., 1., 1., 0.])

    >>> # First derivative
    >>> ts1 = ktk.filters.deriv(ts)

    >>> ts1.time
    array([0.05, 0.15, 0.25, 0.35])
    >>> ts1.data["test"]
    array([  0.,  10.,   0., -10.])

    >>> # Second derivative
    >>> ts2 = ktk.filters.deriv(ts, n=2)

    >>> ts2.time
    array([0.1, 0.2, 0.3])
    >>> ts2.data["test"]
    array([ 100., -100., -100.])

    """
    check_param("ts", ts, TimeSeries)
    check_param("n", n, int)
    _validate_input(ts)

    out_ts = ts.copy()

    for i in range(n):
        out_ts.time = (out_ts.time[1:] + out_ts.time[0:-1]) / 2

    for key in ts.data:
        out_ts.data[key] = (
            np.diff(ts.data[key], n=n, axis=0) / (ts.time[1] - ts.time[0]) ** n
        )

    return out_ts


def median(ts: TimeSeries, /, window_length: int = 3) -> TimeSeries:
    """
    Calculate a moving median.

    Filtering occurs on the first axis (time).

    Parameters
    ----------
    ts
        Input TimeSeries

    window_length
        Optional. Kernel size, must be odd. The default is 3.

    Example
    -------
    >>> ts = ktk.TimeSeries(time=np.arange(0, 6))
    >>> ts = ts.add_data("test", [10., 11., 11., 20., 14., 15.])
    >>> ts2 = ktk.filters.median(ts)
    >>> ts2.data["test"]
    array([10., 11., 11., 14., 15., 15.])

    """
    check_param("ts", ts, TimeSeries)
    check_param("window_length", window_length, int)

    out_ts = ts.copy()
    for key in ts.data:
        window_shape = [1 for i in range(len(ts.data[key].shape))]
        window_shape[0] = window_length
        out_ts.data[key] = ndi.median_filter(ts.data[key], size=window_shape)

    return out_ts


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
