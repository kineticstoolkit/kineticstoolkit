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
"""Provide info management methods for TimeSeries."""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2025 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


from typing import Any

from kineticstoolkit.typing_ import TYPE_CHECKING, check_param

if TYPE_CHECKING:
    from kineticstoolkit import TimeSeries


def add_info(
    self,
    outer_key: str,
    inner_key: str,
    value: Any,
    *,
    overwrite: bool = False,
    in_place: bool = False,
) -> "TimeSeries":
    """
    Add new info the to TimeSeries.

    Although we can directly assign new values to the `info` property::

        ts.info["Data"]["Forces"] = {"Unit": "N"}

    the method provides an alternative ::

        ts = ts.add_info("Forces", "Unit", "N")

    with the following advantages:

    **Overwrite prevention**: Setting the overwrite argument determines
    explicitly if you want existing info with the same name to be
    overwritten or not.

    **Parent creation**: The function creates the required hierarchy of
    nested dictionaries.

    Parameters
    ----------
    outer_key
        The key for the first level of nested dictionaries of ts.info.
        This is the generally what the information refers to (e.g.,
        "Time", or the related data key such as "Forces".
    inner_key
        The key for the second level of nested dictionaries of ts.info.
        This is generally the nature of the information (e.g., "Unit").
    value
        The information.
    overwrite
        Optional. True to overwrite if there is already an info key of this
        name. Default is False.
    in_place
        Optional. True to modify and return the original TimeSeries. False
        to return a modified copy of the TimeSeries while leaving the
        original TimeSeries intact. Default is False.

    Returns
    -------
    TimeSeries
        The TimeSeries with the added info.

    Raises
    ------
    ValueError
        If an info with these keys already exists and overwrite is False.

    See Also
    --------
    ktk.TimeSeries.rename_info
    ktk.TimeSeries.remove_info

    Example
    -------
    >>> ts = ktk.TimeSeries()
    >>> ts = ts.add_info("Forces", "Unit", "N")
    >>> ts
    TimeSeries with attributes:
          time: array([], dtype=float64)
          data: {}
        events: []
          info: {'Time': {'Unit': 's'}, 'Forces': {'Unit': 'N'}}

    """
    check_param("outer_key", outer_key, str)
    check_param("inner_key", inner_key, str)
    check_param("overwrite", overwrite, bool)
    check_param("in_place", in_place, bool)
    ts = self if in_place else self.copy()

    if outer_key not in ts.info:
        ts.info[outer_key] = {}

    if (overwrite is False) and (inner_key in ts.info[outer_key]):
        raise ValueError(
            f"An info with key '{inner_key}' already exists in this "
            f"TimeSeries' info[{outer_key}] attribute. Either use another "
            "key name or set overwrite to True."
        )

    ts.info[outer_key][inner_key] = value

    return ts


def rename_info(
    self,
    outer_key: str,
    inner_key: str,
    new_outer_key: str,
    new_inner_key: str,
    *,
    in_place: bool = False,
) -> "TimeSeries":
    """
    Rename info keys.

    Parameters
    ----------
    outer_key
        The key for the first level of nested dictionaries of ts.info.
        This is the generally what the information refers to (e.g.,
        "Time", or the related data key such as "Forces".
    inner_key
        The key for the second level of nested dictionaries of ts.info.
        This is generally the nature of the information (e.g., "Unit").
    new_outer_key
        The new key for the first level of nested dictionaries of ts.info.
    new_inner_key
        The new key for the second level of nested dictionaries of ts.info.
    in_place
        Optional. True to modify and return the original TimeSeries. False
        to return a modified copy of the TimeSeries while leaving the
        original TimeSeries intact. Default is False.

    Returns
    -------
    TimeSeries
        The TimeSeries with the renamed info keys.

    Raises
    ------
    KeyError
        If there is no in ts.info[outer_key][inner_key].

    See Also
    --------
    ktk.TimeSeries.add_info
    ktk.TimeSeries.remove_info

    Example
    -------
    >>> ts = ktk.TimeSeries()
    >>> ts = ts.add_info("Forces", "Unit", "N")
    >>> ts
    TimeSeries with attributes:
          time: array([], dtype=float64)
          data: {}
        events: []
          info: {'Time': {'Unit': 's'}, 'Forces': {'Unit': 'N'}}

    >>> ts = ts.rename_info("Forces", "Unit", "Power", "ForceUnit")
    >>> ts
    TimeSeries with attributes:
          time: array([], dtype=float64)
          data: {}
        events: []
          info: {'Time': {'Unit': 's'}, 'Power': {'ForceUnit': 'N'}}

    """
    check_param("outer_key", outer_key, str)
    check_param("inner_key", inner_key, str)
    check_param("new_outer_key", outer_key, str)
    check_param("new_inner_key", inner_key, str)
    check_param("in_place", in_place, bool)
    ts = self if in_place else self.copy()

    if outer_key not in ts.info:
        self._raise_info_outer_key_error(outer_key)
    if inner_key not in ts.info[outer_key]:
        self._raise_info_inner_key_error(outer_key, inner_key)

    # Get the value
    value = ts.info[outer_key][inner_key]

    # Add the value with its new name
    ts.add_info(new_outer_key, new_inner_key, value, in_place=True)

    # Remove the old value
    ts.remove_info(outer_key, inner_key, in_place=True)

    return ts


def remove_info(
    self,
    outer_key: str,
    inner_key: str,
    *,
    in_place: bool = False,
) -> "TimeSeries":
    """
    Remove info from a TimeSeries.

    Parameters
    ----------
    outer_key
        The key for the first level of nested dictionaries of ts.info.
        This is the generally what the information refers to (e.g.,
        "Time", or the related data key such as "Forces".
    inner_key
        The key for the second level of nested dictionaries of ts.info.
        This is generally the nature of the information (e.g., "Unit").
    in_place
        Optional. True to modify and return the original TimeSeries. False
        to return a modified copy of the TimeSeries while leaving the
        original TimeSeries intact. Default is False.

    Returns
    -------
    TimeSeries
        The TimeSeries with the removed info.

    Raises
    ------
    KeyError
        If there is no in ts.info[outer_key][inner_key].

    See Also
    --------
    ktk.TimeSeries.add_info
    ktk.TimeSeries.rename_info

    Example
    -------
    >>> ts = ktk.TimeSeries()
    >>> ts = ts.add_info("Forces", "Unit", "N")
    >>> ts
    TimeSeries with attributes:
          time: array([], dtype=float64)
          data: {}
        events: []
          info: {'Time': {'Unit': 's'}, 'Forces': {'Unit': 'N'}}

    >>> ts = ts.remove_info("Forces", "Unit")
    >>> ts
    TimeSeries with attributes:
          time: array([], dtype=float64)
          data: {}
        events: []
          info: {'Time': {'Unit': 's'}}

    """
    check_param("outer_key", outer_key, str)
    check_param("inner_key", inner_key, str)
    check_param("in_place", in_place, bool)
    ts = self if in_place else self.copy()

    if outer_key not in ts.info:
        self._raise_info_outer_key_error(outer_key)
    if inner_key not in ts.info[outer_key]:
        self._raise_info_inner_key_error(outer_key, inner_key)

    ts.info[outer_key].pop(inner_key)
    if len(ts.info[outer_key]) == 0:
        ts.info.pop(outer_key)
    return ts
