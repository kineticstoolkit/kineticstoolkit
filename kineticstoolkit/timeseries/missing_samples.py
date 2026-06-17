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
"""Provide missing sample management methods for TimeSeries."""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2025 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import numpy as np

from kineticstoolkit.typing_ import TYPE_CHECKING, check_param

if TYPE_CHECKING:
    from kineticstoolkit import TimeSeries


def isnan(self, data_key: str) -> np.ndarray:
    """
    Return a boolean array of missing samples.

    Parameters
    ----------
    data_key
        Key value of the data signal to analyze.

    Returns
    -------
    np.ndarray
        A boolean array of the same size as the time attribute, where True
        values represent missing samples (samples that contain at least
        one nan value).

    See Also
    --------
    ktk.TimeSeries.fill_missing_samples

    Example
    -------
    >>> ts = ktk.TimeSeries(time=np.arange(4))
    >>> ts = ts.add_data("data", np.zeros((4, 2)))
    >>> ts.data["data"][2, :] = np.nan
    >>> ts.data
    {'data': array([[ 0.,  0.], [ 0.,  0.], [nan, nan], [ 0.,  0.]])}

    >>> ts.isnan("data")
    array([False, False,  True, False])

    """
    check_param("data_key", data_key, str)
    self._check_well_shaped()

    values = self.data[data_key].copy()
    # Reduce the dimension of values while keeping the time dimension.
    while len(values.shape) > 1:
        values = np.sum(values, 1)  # type: ignore
    return np.isnan(values)


def fill_missing_samples(
    self,
    max_missing_samples: int,
    *,
    method: str = "linear",
    in_place: bool = False,
) -> "TimeSeries":
    """
    Fill missing samples using a given method.

    Parameters
    ----------
    max_missing_samples
        Maximal number of consecutive missing samples to fill. Set to
        zero to fill all missing samples.
    method
        Optional. The interpolation method. This input may take any value
        supported by scipy.interpolate.interp1d, such as "linear",
        "nearest", "zero", "slinear", "quadratic", "cubic", "previous" or
        "next". Default is "linear".
    in_place
        Optional. True to modify and return the original TimeSeries. False
        to return a modified copy of the TimeSeries while leaving the
        original TimeSeries intact. Default is False.

    Returns
    -------
    TimeSeries
        The TimeSeries with the missing samples filled.

    Raises
    ------
    ValueError
        If the sample rate is not constant.

    See Also
    --------
    ktk.TimeSeries.isnan

    """
    check_param("max_missing_samples", max_missing_samples, int)
    check_param("method", method, str)
    check_param("in_place", in_place, bool)
    self._check_well_shaped()

    if np.isnan(self.get_sample_rate()):
        raise ValueError("The sample rate must be constant.")

    ts_out = self if in_place else self.copy()

    for data in ts_out.data:
        # Fill missing samples
        is_visible = ~ts_out.isnan(data)
        ts = ts_out.get_subset(data)
        ts.data[data] = ts.data[data][is_visible]
        ts.time = ts.time[is_visible]
        ts = ts.resample(ts_out.time, method, extrapolate=True)

        # Put back missing samples in holes longer than max_missing_samples
        if max_missing_samples > 0:
            still_visible_index = -1
            to_keep = np.ones(self.time.shape)
            for current_index in range(ts.time.shape[0]):
                if is_visible[current_index]:
                    still_visible_index = current_index
                elif current_index - still_visible_index > max_missing_samples:
                    to_keep[still_visible_index + 1 : current_index + 1] = 0

            ts.data[data][to_keep == 0] = np.nan

        ts_out.data[data] = ts.data[data]

    return ts_out
