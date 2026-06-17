#!/usr/bin/env python3
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
"""Provide time management methods for TimeSeries."""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2025 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import warnings
from numbers import Real

import numpy as np
import scipy as sp

from kineticstoolkit.typing_ import TYPE_CHECKING, ArrayLike, check_param

if TYPE_CHECKING:
    from kineticstoolkit import TimeSeries


def _get_time_unit(self) -> str:
    try:
        return self.info["Time"]["Unit"]
    except KeyError:
        return "no unit"


def shift(self, time: float, *, in_place: bool = False) -> "TimeSeries":
    """
    Shift time and events.time.

    Parameters
    ----------
    time
        Time to be added to time and events.time.
    in_place
        Optional. True to modify and return the original TimeSeries. False
        to return a modified copy of the TimeSeries while leaving the
        original TimeSeries intact. Default is False.

    Returns
    -------
    TimeSeries
        The TimeSeries with the time being shifted.

    See Also
    --------
    ktk.TimeSeries.ui_sync

    Example
    -------
    >>> ts = ktk.TimeSeries(time=np.arange(10) / 10)
    >>> ts = ts.add_event(0.35, "start")
    >>> ts.time
    array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    >>> ts.events
    [TimeSeriesEvent(time=0.35, name='start')]

    >>> ts = ts.shift(0.2)
    >>> ts.time
    array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. , 1.1])

    >>> ts.events
    [TimeSeriesEvent(time=0.55, name='start')]

    """
    check_param("time", time, float)
    check_param("in_place", in_place, bool)
    self._check_well_shaped()

    ts = self if in_place else self.copy()
    for event in ts.events:
        event.time += time
    ts.time += time
    return ts


def get_sample_rate(self) -> float:
    """
    Get the sample rate in samples/s.

    Returns
    -------
    float
        The sample rate in samples per second. If time is empty or has only
        one data, or if sample rate is variable, or if time is not
        monotonously increasing, a value of np.nan is returned.

    Warning
    -------
    This feature, which has been introduced in version 0.9, is still
    experimental and may change in the future. In particular, the value
    returned if the sample rate is not constant: it is np.nan in all cases
    for now, but it could change in the future based on discussions and
    particular use cases.

    See Also
    --------
    ktk.TimeSeries.resample

    Example
    -------
    >>> ts = ktk.TimeSeries(time=np.arange(100) / 10)  # 100 samples at 10 Hz
    >>> ts.get_sample_rate()
    10.0

    """
    self._check_well_shaped()

    if self.time.shape[0] <= 1:
        return np.nan

    deltas = self.time[1:] - self.time[0:-1]
    if np.allclose(deltas, [deltas[0]]):
        return 1.0 / deltas.mean()
    else:
        return np.nan


def resample(
    self,
    target: ArrayLike | float,
    kind: str = "linear",
    *,
    extrapolate: bool = False,
    in_place: bool = False,
    **kwargs,
) -> "TimeSeries":
    """
    Resample the TimeSeries.

    Resample every data of the TimeSeries over a new frequency or new
    series of times, using the interpolation method provided by parameter
    `kind`. This method does not fill missing data. Consequently, time
    ranges with nans in the original TimeSeries will also contain nans in
    the resampled TimeSeries.

    Parameters
    ----------
    target
        To resample to a target frequency, use a float that represents
        the sample rate of the output TimeSeries, in Hz. To resample to
        specific times, use an array of float that will become the time
        property of the output TimeSeries.
    kind
        Optional. The interpolation method. This input may take any value
        supported by scipy.interpolate.interp1d, such as "linear",
        "nearest", "zero", "slinear", "quadratic", "cubic", "previous",
        "next". Additionally, kind can be "pchip". Default is "linear".
    extrapolate
        Optional. True to extrapolate outside the original time range.
        Default is False.
    in_place
        Optional. True to modify and return the original TimeSeries. False
        to return a modified copy of the TimeSeries while leaving the
        original TimeSeries intact. Default is False.

    Returns
    -------
    TimeSeries
        The TimeSeries with a new sample rate.

    Caution
    -------
    Attempting to resample a series of homogeneous matrices would likely
    produce non-homogeneous matrices, and as a result, transforms would not
    be rigid anymore. This function can't detect if you attempt to resample
    series of homogeneous matrices, and therefore won't generate an
    error or warning.

    See Also
    --------
    ktk.TimeSeries.get_sample_rate
    ktk.TimeSeries.fill_missing_samples

    Examples
    --------
    >>> ts = ktk.TimeSeries(time=np.arange(10.0))
    >>> ts = ts.add_data("data", ts.time**2)
    >>> ts.time
    array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
    >>> ts.data["data"]
    array([ 0.,  1.,  4.,  9., 16., 25., 36., 49., 64., 81.])

    Example 1: Resampling at 2 Hz

    >>> ts1 = ts.resample(2.0)

    >>> ts1.time
    array([0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. , 5.5, 6. , 6.5, 7. , 7.5, 8. , 8.5, 9. ])

    >>> ts1.data["data"]
    array([ 0. ,  0.5,  1. ,  2.5,  4. ,  6.5,  9. , 12.5, 16. , 20.5, 25. , 30.5, 36. , 42.5, 49. , 56.5, 64. , 72.5, 81. ])

    Example 2: Resampling on new times

    >>> ts2 = ts.resample([0.0, 0.5, 1.0, 1.5, 2.0])

    >>> ts2.time
    array([0. , 0.5, 1. , 1.5, 2. ])

    >>> ts2.data["data"]
    array([0. , 0.5, 1. , 2.5, 4. ])

    Example 3: Resampling at 2 Hz with missing data in the original ts

    >>> ts.data["data"][[0, 1, 5, 8, 9]] = np.nan
    >>> ts.data["data"]
    array([nan, nan,  4.,  9., 16., nan, 36., 49., nan, nan])

    >>> ts3 = ts.resample(2.0)

    >>> ts3.time
    array([0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. , 5.5, 6. , 6.5, 7. , 7.5, 8. , 8.5, 9. ])

    >>> ts3.data["data"]
    array([ nan,  nan,  nan,  nan,  4. ,  6.5,  9. , 12.5, 16. ,  nan,  nan, nan, 36. , 42.5, 49. ,  nan,  nan,  nan,  nan])

    """
    check_param("kind", kind, str)
    check_param("in_place", in_place, bool)

    if "fill_value" in kwargs:
        warnings.warn(
            "fill_value parameter has been removed in version 0.12 "
            "because its behavior was unclear and it was ignored in many "
            "situations "
            "(https://github.com/felixchenier/kineticstoolkit/issues/174)."
        )

    self._check_well_shaped()

    ts = self if in_place else self.copy()

    # --------------------------------------------------------------
    # Create the new time if a frequency was provided instead
    if isinstance(target, Real):
        # We specifically use arange instead of linspace, because what
        # is defined is a frequency, not a number of points.
        new_time = np.arange(
            ts.time[0],
            ts.time[-1] + 1 / target,
            1 / target,
        )
        # Work around the numerical instability of using arange with floats
        # by ensuring that the time point is not higher than the original
        # last time point
        if new_time[-1] > ts.time[-1]:
            new_time = new_time[:-1]
    else:
        new_time = np.array(target)  # type: ignore
    # --------------------------------------------------------------

    if np.any(np.isnan(new_time)):
        raise ValueError("new_time must not contain nans")

    # We will progressively fill these data
    new_data = {}  # type: dict[str, np.ndarray]

    for key in ts.data.keys():
        index = ~ts.isnan(key)

        if sum(index) < 3:  # Only Nans, cannot interpolate.
            # We generate an array of nans of the expected size.
            new_shape = [len(new_time)]
            for i in range(1, len(self.data[key].shape)):
                new_shape.append(self.data[key].shape[i])
            new_data[key] = np.empty(new_shape)
            new_data[key][:] = np.nan
            continue

        # Express nans as a range of times to
        # remove from the final, interpolated TimeSeries
        nan_indexes = np.argwhere(~index)

        # initialize with times outside of the original time range
        time_ranges_to_remove: list[tuple[float, float]] = []
        if not extrapolate:
            time_ranges_to_remove.append((-np.inf, ts.time[0]))
            time_ranges_to_remove.append((ts.time[-1], np.inf))

        length = ts.time.shape[0]
        for i in nan_indexes[:, 0]:
            if i > 0 and i < length - 1:
                time_range = (ts.time[i - 1], ts.time[i + 1])
            elif i == 0:
                time_range = (-np.inf, ts.time[i + 1])
            else:
                time_range = (ts.time[i - 1], np.inf)
            time_ranges_to_remove.append(time_range)

        if kind == "pchip":
            P = sp.interpolate.PchipInterpolator(
                ts.time[index],
                ts.data[key][index],
                axis=0,
                extrapolate=True,
            )
            new_data[key] = P(new_time)
        else:
            f = sp.interpolate.interp1d(
                ts.time[index],
                ts.data[key][index],
                axis=0,
                fill_value="extrapolate",
                kind=kind,
            )
            new_data[key] = f(new_time)

        # Put back nans in the originally missing data
        for j in time_ranges_to_remove:
            new_data[key][(new_time > j[0]) & (new_time < j[1])] = np.nan

    ts.time = new_time
    ts.data = new_data
    return ts
