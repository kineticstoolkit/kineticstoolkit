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
"""Provide copy method for TimeSeries."""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2025 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from kineticstoolkit.typing_ import ArrayLike, check_param

if TYPE_CHECKING:
    from kineticstoolkit import TimeSeries


def add_data(
    self,
    data_key: str,
    data_value: ArrayLike,
    *,
    overwrite: bool = False,
    in_place: bool = False,
) -> "TimeSeries":
    """
    Add new data to the TimeSeries.

    Although we can directly assign values to the `data` property::

        ts.data["name"] = value

    this method provides an alternative way to add data to the TimeSeries::

        ts = ts.add_data(name, value, ...)

    with the following advantages:

    **Overwrite prevention**: Setting the overwrite argument determines
    explicitly if you want existing data with the same name to be
    overwritten or not.

    **Size check**: Additional data is compared to the contents of the
    TimeSeries to ensure that it has the correct dimensions. See Raises
    section for more information.

    **Size matching**: Constant "series" such as [3.0], which is a
    one-sample series of 3.0, are automatically expanded to match the size
    of the TimeSeries. For example, if the TimeSeries has 4 samples, then
    the input data is expanded to [3.0, 3.0, 3.0, 3.0].

    Parameters
    ----------
    data_key
        Name of the data key.
    data_value
        Any data that can be converted to a NumPy array
    overwrite
        Optional. True to overwrite if there is already a data key of this
        name in the TimeSeries. Default is False.
    in_place
        Optional. True to modify and return the original TimeSeries. False
        to return a modified copy of the TimeSeries while leaving the
        original TimeSeries intact. Default is False.

    Returns
    -------
    TimeSeries
        The TimeSeries with the added data.

    Raises
    ------
    ValueError
        If data with this key already exists and overwrite is False,
        if the size of the data (first dimension) does not match the size
        of existing data or the existing time, or
        if data is a pandas DataFrame and its index does not match the
        existing time.

    See Also
    --------
    ktk.TimeSeries.rename_data
    ktk.TimeSeries.remove_data

    Examples
    --------
    >>> ts = ktk.TimeSeries()
    >>> ts = ts.add_data("data1", [1.0, 2.0, 3.0])
    >>> ts = ts.add_data("data2", [4.0, 5.0, 6.0])
    >>> ts
    TimeSeries with attributes:
          time: array([], dtype=float64)
          data: {'data1': array([1., 2., 3.]), 'data2': array([4., 5., 6.])}
          events: []
          info: {'Time': {'Unit': 's'}}

    >>> # Size matching example
    >>> ts = ktk.TimeSeries(time=[0.0, 0.1, 0.2, 0.3])
    >>> ts = ts.add_data("data1", [9.9])
    >>> ts
    TimeSeries with attributes:
          time: array([0. , 0.1, 0.2, 0.3])
          data: {'data1': array([9.9, 9.9, 9.9, 9.9])}
          events: []
          info: {'Time': {'Unit': 's'}}

    """
    check_param("data_key", data_key, str)
    check_param("overwrite", overwrite, bool)
    check_param("in_place", in_place, bool)
    ts = self if in_place else self.copy()

    # Cast data
    data_to_add = np.array(data_value)  # Will be set at the very end

    # Check the size of the TimeSeries
    if ts.time.shape[0] != 0:
        n_samples = ts.time.shape[0]
    elif len(ts.data) > 0:
        n_samples = ts.data[list(ts.data.keys())[0]].shape[0]
    else:
        n_samples = 0

    # Expand the input to n_sample if it's a constant series
    if data_to_add.shape[0] == 1 and n_samples > 0:
        data_to_add = np.repeat(data_to_add, n_samples, axis=0)

    # Check that the data fits with the TimeSeries' time (if it exists)
    if ts.time.shape[0] != 0:
        # If this is a Pandas DataFrame, check that its index is fully
        # compatible with time
        if isinstance(data_value, pd.DataFrame):
            if (ts.time.shape[0] != data_to_add.shape[0]) or (
                not np.allclose(ts.time, np.array(data_value.index))
            ):
                raise ValueError(
                    "The index of the provided DataFrame does not match "
                    "this TimeSeries' time attribute. This error was raised "
                    "to prevent merging unsynchronized data. If you are "
                    "confident that this DataFrame's data does match this "
                    "TimeSeries, then set its index to this TimeSeries' time "
                    "before adding it: "
                    "the_dataframe.index = the_timeseries.time"
                )

        # For every other type, check that the dimensions fit at least.
        elif ts.time.shape[0] != data_to_add.shape[0]:
            raise ValueError(
                f"This data has {data_to_add.shape[0]} samples while "
                f"this TimeSeries' time has {ts.time.shape[0]} samples."
            )

    # Check that the data fits with other data (if it exists)
    for key in ts.data:
        if ts.data[key].shape[0] != data_to_add.shape[0]:
            raise ValueError(
                f"This data has {data_to_add.shape[0]} samples while "
                f"this TimeSeries' data {key} has {ts.data[key].shape[0]} "
                "samples."
            )

    # Check that we would not overwrite by mistake
    if (data_key in self.data) and (overwrite is False):
        raise ValueError(
            f"A data with key '{data_key}' already exists in this "
            "TimeSeries. Either use another key name or set overwrite to "
            "True."
        )

    # Add the data
    ts.data[data_key] = data_to_add
    return ts


def rename_data(
    self, old_data_key: str, new_data_key: str, *, in_place: bool = False
) -> "TimeSeries":
    """
    Rename a key in data.

    Parameters
    ----------
    old_data_key
        Name of the current data key.
    new_data_key
        New name of the data key.
    in_place
        Optional. True to modify and return the original TimeSeries. False
        to return a modified copy of the TimeSeries while leaving the
        original TimeSeries intact. Default is False.

    Returns
    -------
    TimeSeries
        The TimeSeries with the renamed data.

    Raises
    ------
    KeyError
        If this data key could not be found in the TimeSeries' data
        attribute.

    See Also
    --------
    ktk.TimeSeries.add_data
    ktk.TimeSeries.remove_data
    ktk.TimeSeries.rename_info

    Example
    -------
    >>> ts = ktk.TimeSeries(time=np.arange(10))
    >>> ts = ts.add_data("test", np.arange(10))
    >>> ts = ts.add_info("test", "Unit", "m")

    >>> ts
    TimeSeries with attributes:
          time: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
          data: {'test': array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}
        events: []
          info: {'Time': {'Unit': 's'}, 'test': {'Unit': 'm'}}

    >>> ts = ts.rename_data("test", "signal")

    >>> ts
    TimeSeries with attributes:
          time: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
          data: {'signal': array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}
        events: []
          info: {'Time': {'Unit': 's'}, 'test': {'Unit': 'm'}}

    """
    check_param("old_data_key", old_data_key, str)
    check_param("new_data_key", new_data_key, str)
    check_param("in_place", in_place, bool)
    self._check_valid_time()

    ts = self if in_place else self.copy()
    try:
        ts.data[new_data_key] = ts.data.pop(old_data_key)
    except KeyError:
        self._raise_data_key_error(old_data_key)

    return ts


def remove_data(
    self, data_key: str, *, in_place: bool = False
) -> "TimeSeries":
    """
    Remove a key in data.

    Parameters
    ----------
    data_key
        Name of the data key.
    in_place
        Optional. True to modify and return the original TimeSeries. False
        to return a modified copy of the TimeSeries while leaving the
        original TimeSeries intact. Default is False.

    Returns
    -------
    TimeSeries
        The TimeSeries with the removed data.

    Raises
    ------
    KeyError
        If this data key could not be found in the TimeSeries' data
        attribute.

    See Also
    --------
    ktk.TimeSeries.add_data
    ktk.TimeSeries.rename_data
    ktk.TimeSeries.remove_info

    Example
    -------
    >>> # Prepare a test TimeSeries with data "test"
    >>> ts = ktk.TimeSeries()
    >>> ts = ts.add_data("test", np.arange(10))
    >>> ts = ts.add_info("test", "Unit", "m")

    >>> ts
    TimeSeries with attributes:
          time: array([], dtype=float64)
          data: {'test': array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}
        events: []
          info: {'Time': {'Unit': 's'}, 'test': {'Unit': 'm'}}

    >>> # Now remove data "test"
    >>> ts = ts.remove_data("test")

    >>> ts
    TimeSeries with attributes:
          time: array([], dtype=float64)
          data: {}
        events: []
          info: {'Time': {'Unit': 's'}, 'test': {'Unit': 'm'}}

    """
    check_param("data_key", data_key, str)
    check_param("in_place", in_place, bool)
    self._check_valid_time()

    ts = self if in_place else self.copy()
    try:
        ts.data.pop(data_key)
    except KeyError:
        self._raise_data_key_error(data_key)

    return ts
