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
"""Provide private check functions for TimeSeries."""

import numpy as np

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2025 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


def _is_equivalent(
    self,
    ts,
    *,
    equal: bool = True,
    atol: float = 1e-8,
    rtol: float = 1e-5,
    debug: bool = False,
):
    """
    Test is two TimeSeries are equal or equivalent.

    Parameters
    ----------
    ts
        The TimeSeries to compare to.
    equal
        Optional. True to test for complete equality, False to compare
        within a given tolerance.
    atol
        Optional. Absolute tolerance if using equal=False.
    rtol
        Optional. Relative tolerance if using equal=False.
    debug
        Optional. Prints what parameter is not equal. Default is False.

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
        ts._check_valid_time()
    except AttributeError:
        if debug:
            print("The variable begin compared is not a TimeSeries.")

    if not compare(self.time, ts.time, atol=atol, rtol=rtol):
        if debug:
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
                    if debug:
                        print(f"{one_data} is not equal")
                    return False
            except KeyError:
                if debug:
                    print(f"{one_data} is missing in one of the TimeSeries")
                return False
            except ValueError:
                if debug:
                    print(
                        f"{one_data} does not have the same size in both "
                        "TimeSeries"
                    )
                return False

    if self.info != ts.info:
        if debug:
            print("info is not equal")
        return False

    if self.events != ts.events:
        if debug:
            print("events is not equal")
        return False

    return True


def _check_valid_time(self) -> None:
    """
    Check that time doesn't have nans or duplicate values.

    Raises
    ------
    ValueError
        If the time attribute contains invalid values.

    """
    if not np.all(~np.isnan(self.time)):
        raise ValueError(
            "A TimeSeries' time attribute must not contain nans. "
            f"However, a total of {np.sum(~np.isnan(self.time.shape))} "
            f"nans were found among the {self.time.shape[0]} samples of "
            "the TimeSeries."
        )

    if not np.array_equal(np.unique(self.time), np.sort(self.time)):
        raise ValueError(
            "A TimeSeries' time attribute must not contain duplicates. "
            f"However, while the TimeSeries has {len(self.time)} samples, "
            f"only {len(np.unique(self.time))} are unique."
        )


def _check_well_shaped(self) -> None:
    """
    Check that the TimeSeries' time and data shapes concord.

    Raises
    ------
    ValueError
        If the TimeSeries' time and data do not concord in shape.

    """
    self._check_valid_time()
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
    Check that the TimeSeries' time attribute is not empty.

    Raises
    ------
    ValueError
        If the TimeSeries' time is empty

    """
    if self.time.shape[0] == 0:
        raise ValueError(
            "The TimeSeries is empty: the length of its time attribute is 0."
        )


def _check_increasing_time(self) -> None:
    """
    Check that the TimeSeries' time attribute is always increasing.

    Raises
    ------
    ValueError
        If the TimeSeries' time is not always increasing.

    """
    if not np.array_equal(self.time, np.sort(self.time)):
        raise ValueError(
            "The TimeSeries' time attribute is not always increasing, "
            "which is required by the requested function. You can "
            "resample the TimeSeries on an always increasing time attribute "
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
        If the TimeSeries has no time.

    """
    if len(self.data) == 0:
        raise ValueError(
            "The TimeSeries is empty: it does not contain any data."
        )


def _raise_data_key_error(self, data_key) -> None:
    raise KeyError(
        f"The key '{data_key}' was not found among the "
        f"{len(self.data)} key(s) of the TimeSeries' "
        "data attribute."
    )


def _raise_info_outer_key_error(self, outer_key) -> None:
    raise KeyError(
        f"The key '{outer_key}' was not found among the "
        f"{len(self.info)} key(s) of the TimeSeries' "
        f"info attribute."
    )


def _raise_info_inner_key_error(self, outer_key, inner_key) -> None:
    raise KeyError(
        f"The key '{inner_key}' was not found among the "
        f"{len(self.info[outer_key])} key(s) of the TimeSeries' "
        f"info[{outer_key}] attribute."
    )
