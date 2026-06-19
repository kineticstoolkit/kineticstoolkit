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
"""Provide deprecated methods for TimeSeries."""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2025 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import warnings
from typing import Any

from kineticstoolkit.decorators import deprecated
from kineticstoolkit.typing_ import TYPE_CHECKING, check_param

if TYPE_CHECKING:
    from kineticstoolkit import TimeSeries


# %% Deprecated methods
@deprecated(
    since="0.15",
    until="2027",
    details=(
        "Events are now always sorted in the events attribute. "
        "There is no need to run the sort_events method anymore."
    ),
)
def sort_events(
    self, *, unique: bool = False, in_place: bool = False
) -> "TimeSeries":
    """
    Sort the TimeSeries' events (deprecated).

    Parameters
    ----------
    unique
        Optional. True to make events unique so that no two events can
        have both the same name and the same time.
    in_place
        Optional. True to modify and return the original TimeSeries. False
        to return a modified copy of the TimeSeries while leaving the
        original TimeSeries intact. Default is False.

    Returns
    -------
    TimeSeries
        The TimeSeries with the sorted events.

    """
    check_param("unique", unique, bool)
    check_param("in_place", in_place, bool)
    self._check_valid_time()

    ts = self if in_place else self.copy()
    if unique:
        ts.remove_duplicate_events(in_place=True)
    ts.events = sorted(ts.events)
    return ts


# To deprecate on v1.0
def add_data_info(
    self,
    data_key: str,
    info_key: str,
    value: Any,
    *,
    overwrite: bool = False,
    in_place: bool = False,
) -> "TimeSeries":
    """
    Add metadata to TimeSeries' data.

    Warning
    -------
    This function will be deprecated when Kinetics Toolkit will reach
    version 1.0. Please use add_info instead.

    Parameters
    ----------
    data_key
        The data key the info corresponds to.
    info_key
        The key of the info dict.
    value
        The info.
    overwrite
        Optional. True to overwrite the data info if it is already present
        in the TimeSeries. Default is False.
    in_place
        Optional. True to modify the original TimeSeries. False
        to return a modified copy of the TimeSeries while leaving the
        original TimeSeries intact. Default is False.

    Returns
    -------
    TimeSeries
        The TimeSeries with the added data info.

    """
    check_param("data_key", data_key, str)
    check_param("info_key", info_key, str)
    check_param("overwrite", overwrite, bool)
    check_param("in_place", in_place, bool)
    self._check_valid_time()

    try:
        ts = self.add_info(
            data_key,
            info_key,
            value,
            overwrite=overwrite,
            in_place=in_place,
        )
        return ts
    except ValueError as e:
        warnings.warn(str(e))
        return self if in_place is True else self.copy()


def remove_data_info(
    self, data_key: str, info_key: str, *, in_place: bool = False
) -> "TimeSeries":
    """
    Remove metadata from a TimeSeries' data.

    Warning
    -------
    This function will be deprecated when Kinetics Toolkit will reach
    version 1.0. Please use add_info instead.

    Parameters
    ----------
    data_key
        The data key the info corresponds to.
    info_key
        The key of the info dict.
    in_place
        Optional. True to modify the original TimeSeries. False
        to return a modified copy of the TimeSeries while leaving the
        original TimeSeries intact.

    Returns
    -------
    TimeSeries
        The TimeSeries with the removed data info.

    Raises
    ------
    KeyError
        If this data_info could not be found.

    """
    check_param("data_key", data_key, str)
    check_param("info_key", info_key, str)
    check_param("in_place", in_place, bool)
    self._check_valid_time()

    ts = self if in_place else self.copy()

    ts = ts.remove_info(data_key, info_key, in_place=in_place)
    return ts
