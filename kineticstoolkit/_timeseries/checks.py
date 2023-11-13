#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020-2023 Félix Chénier

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
Checks for TimeSeries.

"""


__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2023 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import numpy as np
from kineticstoolkit._timeseries.classes import TimeSeriesEvent


def _is_equivalent(
    self, ts, *, equal: bool = True, atol: float = 1e-8, rtol: float = 1e-5
):
    """
    Test is two TimeSeries are equal or equivalent.

    Parameters
    ----------
    ts
        The TimeSeries to compare to.
    equal
        Optional. True to test for complete equality, False to compare
        withint a given tolerance.
    atol
        Optional. Absolute tolerance if using equal=False.
    rtol
        Optional. Relative tolerance if using equal=False.

    Returns
    -------
    bool
        True if the TimeSeries are equivalent.

    """
    if equal:
        atol = 0
        rtol = 0

    def compare(var1, var2, atol, rtol):
        if var1.size == 0 and var2.size == 0:
            return np.equal(var1.shape, var2.shape)
        elif var1.size == 0 and var2.size != 0:
            return False
        elif var1.size != 0 and var2.size == 0:
            return False
        else:
            return np.allclose(
                var1, var2, atol=atol, rtol=rtol, equal_nan=True
            )

    try:
        ts._check_well_typed()
    except AttributeError:
        print("The variable begin compared is not a TimeSeries.")

    if not compare(self.time, ts.time, atol=atol, rtol=rtol):
        print("Time is not equal")
        return False

    for data in [self.data, ts.data]:
        for one_data in data:
            try:
                if not compare(
                    self.data[one_data],
                    ts.data[one_data],
                    atol=atol,
                    rtol=rtol,
                ):
                    print(f"{one_data} is not equal")
                    return False
            except KeyError:
                print(f"{one_data} is missing in one of the TimeSeries")
                return False
            except ValueError:
                print(
                    f"{one_data} does not have the same size in both "
                    "TimeSeries"
                )
                return False

    if self.time_info != ts.time_info:
        print("time_info is not equal")
        return False

    if self.data_info != ts.data_info:
        print("data_info is not equal")
        return False

    if self.events != ts.events:
        print("events is not equal")
        return False

    return True


def _check_well_typed(self) -> None:
    """
    Check that every element of every attribute has correct type.

    This is the most basic check: Every component of a TimeSeries must
    be of the correct type at each step of a code. Therefore, any other
    check* starts by calling this function. This is not a performance hit
    because apart from interactive functions (which are not affected by
    test overhead), the check functions are run in case of error only,
    to help the users in fixing their code.

    *except _check_increasing_time, which is run in proprocessing and not
    only in failures.

    Raises
    ------
    AttributeError
        If the TimeSeries' miss some attributes.

    TypeError
        If the TimeSeries' attributes are of wrong type.

    """
    # Ensure that the TimeSeries has all its attributes
    try:
        self.time
    except AttributeError:
        raise AttributeError(
            "This TimeSeries does not have a time attribute anymore."
        )

    try:
        self.data
    except AttributeError:
        raise AttributeError(
            "This TimeSeries does not have a data attribute anymore."
        )

    try:
        self.events
    except AttributeError:
        raise AttributeError(
            "This TimeSeries does not have a events attribute anymore."
        )

    try:
        self.time_info
    except AttributeError:
        raise AttributeError(
            "This TimeSeries does not have a time_info attribute anymore."
        )

    try:
        self.data_info
    except AttributeError:
        raise AttributeError(
            "This TimeSeries does not have a data_info attribute anymore."
        )

    # Ensure that time is a numpy array of dimension 1.
    if not isinstance(self.time, np.ndarray):
        raise TypeError(
            "A TimeSeries' time attribute must be a numpy array. "
            f"However, the current time type is {type(self.time)}."
        )

    if not np.all(~np.isnan(self.time)):
        raise TypeError(
            "A TimeSeries' time attribute must not contain nans. "
            f"However, a total of {np.sum(~np.isnan(self.time.shape))} "
            f"nans were found among the {self.time.shape[0]} samples of "
            "the TimeSeries."
        )

    if not np.array_equal(np.unique(self.time), np.sort(self.time)):
        raise TypeError(
            "A TimeSeries' time attribute must not contain duplicates. "
            f"However, while the TimeSeries has {len(self.time)} samples, "
            f"only {len(np.unique(self.time))} are unique."
        )

    # Ensure that the data attribute is a dict
    if not isinstance(self.data, dict):
        raise TypeError(
            "The TimeSeries data attribute must be a dict. However, "
            "this TimeSeries' data attribute is of type "
            f"{type(self.data)}."
        )

    # Ensure that each data are numpy arrays
    for key in self.data:
        data = self.data[key]

        if not isinstance(data, np.ndarray):
            raise TypeError(
                "A TimeSeries' data attribute must contain only numpy "
                "arrays. However, at least one of the TimeSeries data "
                f"is not an array: the data named {key} contains a "
                f"value of type {type(data)}."
            )

    # Ensure that events is a list of TimeSeriesEvent
    if not isinstance(self.events, list):
        raise TypeError(
            "The TimeSeries' events attribute must be a list. "
            "However, this TimeSeries' events attribute is of type "
            f"{type(self.events)}."
        )

    # Ensure that all events are an instance of TimeSeriesEvent
    for i_event, event in enumerate(self.events):
        if not isinstance(event, TimeSeriesEvent):
            raise TypeError(
                "The TimeSeries' events attribute must be a list of "
                "TimeSeriesEvent. However, at least one element of this "
                f"list is not: element {i_event} is "
                f"of type {type(event)}."
            )

    # Ensure that TimeInfo is a dict
    if not isinstance(self.time_info, dict):
        raise TypeError(
            "The TimeSeries' time_info attribute must be a dict. "
            "However, this TimeSeries' time_info attribute is of type "
            f"{type(self.time_info)}."
        )

    # Ensure that DataInfo is a dict
    if not isinstance(self.data_info, dict):
        raise TypeError(
            "The TimeSeries' data_info attribute must be a dict. "
            "However, this TimeSeries' data_info attribute is of type "
            f"{type(self.data_info)}."
        )

    # Ensure that every element of DataInfo is a dict
    for key in self.data_info:
        if not isinstance(self.data_info[key], dict):
            raise TypeError(
                "Each element of a TimeSeries' data_info attribute must "
                f"be a dict. However, the element '{key}' of this "
                "TimeSeries' data_info attribute is of type "
                f"{type(self.data_info[key])}."
            )


def _check_well_shaped(self) -> None:
    """
    Check that the TimeSeries' time and data shapes concord.

    Raises
    ------
    ValueError
        If the TimeSeries' time and data do not concord in shape.

    """
    self._check_well_typed()
    if len(self.time.shape) != 1:
        raise TypeError(
            "A TimeSeries' time attribute must be a numpy array of "
            "dimension 1. However, the current time shape is "
            f"{self.time.shape}, which is a dimension of "
            f"{len(self.time.shape)}."
        )

    for key in self.data:
        data = self.data[key]
        # Ensure that it's coherent in shape with time
        if data.shape[0] != self.time.shape[0]:
            raise ValueError(
                "Every data of a TimeSeries must have its first "
                "dimension corresponding to time. At least one of the "
                "TimeSeries data has a dimension problem: the data "
                f"named '{key}' has a shape of {data.shape} while the "
                f"time's shape is {self.time.shape}."
            )


def _check_not_empty_time(self) -> None:
    """
    Check that the TimeSeries' time vector is not empty.

    Raises
    ------
    ValueError
        If the TimeSeries time is empty

    """
    if self.time.shape[0] == 0:
        raise ValueError(
            "The TimeSeries is empty: the length of its time "
            "attribute is 0."
        )


def _check_increasing_time(self) -> None:
    """
    Check that the TimeSeries' time vector is always increasing.

    Raises
    ------
    ValueError
        If the TimeSeries' time is not always increasing.

    """
    if not np.array_equal(self.time, np.sort(self.time)):
        raise ValueError(
            "The TimeSeries' time attribute is not always increasing, "
            "which is required by the requested function. You can "
            "resample the TimeSeries on an always increasing time vector "
            "using ts = ts.resample(np.sort(ts.time))."
        )


def _check_constant_sample_rate(self) -> None:
    """
    Check that the TimeSeries's sampling rate is constant.

    Raises
    ------
    ValueError
        If the TimeSeries's sampling rate is not constant.

    """
    if np.isnan(self.get_sample_rate()):
        raise ValueError(
            "The TimeSeries's sample rate is not constant, which is "
            "required by the requested function. You can resample the "
            "TimeSeries on a constant sample rate using "
            "ts = ts.resample(np.linspace("
            "np.min(ts.time), np.max(ts.time), len(ts.time)))."
        )


def _check_not_empty_data(self) -> None:
    """
    Check that the TimeSeries's data dict is not empty.

    Raises
    ------
    ValueError:
        If the TimeSeries as no time

    """
    if len(self.data) == 0:
        raise ValueError(
            "The TimeSeries is empty: it does not contain any data."
        )


def _raise_data_key_error(self, data_key) -> None:
    raise KeyError(
        f"The key '{data_key}' was not found among the "
        f"{len(self.data)} key(s) of the TimeSeries' "
        "data_info attribute."
    )


def _raise_data_info_key_error(self, data_key, info_key) -> None:
    raise KeyError(
        f"The key '{info_key}' was not found among the "
        f"{len(self.data_info[data_key])} key(s) of the TimeSeries' "
        f"data_info[{data_key}] attribute."
    )
