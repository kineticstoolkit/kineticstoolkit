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
"""
Provide the TimeSeries and TimeSeriesEvent classes.

The classes defined in this module are accessible directly from the top-
level Kinetics Toolkit's namespace (i.e. ktk.TimeSeries,
ktk.TimeSeriesEvent)

"""

from __future__ import annotations  # For forward refs to self

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2025 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


from ast import literal_eval
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

import kineticstoolkit._repr
from kineticstoolkit.typing_ import ArrayLike, check_param

from .checks import (
    _check_constant_sample_rate,
    _check_increasing_time,
    _check_not_empty_data,
    _check_not_empty_time,
    _check_valid_time,
    _check_well_shaped,
    _is_equivalent,
    _raise_data_key_error,
    _raise_info_inner_key_error,
    _raise_info_outer_key_error,
)
from .classes import (
    TimeSeriesDataDict,
    TimeSeriesEvent,
    TimeSeriesEventList,
    TimeSeriesInfoDict,
)
from .data import add_data, remove_data, rename_data
from .deprecated import add_data_info, remove_data_info, sort_events
from .events import (
    _get_duplicate_event_indexes,
    _get_event_index,
    _get_event_indexes,
    add_event,
    count_events,
    remove_duplicate_events,
    remove_event,
    rename_event,
    trim_events,
)
from .get_index import (
    get_index_after_event,
    get_index_after_time,
    get_index_at_event,
    get_index_at_time,
    get_index_before_event,
    get_index_before_time,
)
from .get_ts import (
    get_ts_after_event,
    get_ts_after_index,
    get_ts_after_time,
    get_ts_before_event,
    get_ts_before_index,
    get_ts_before_time,
    get_ts_between_events,
    get_ts_between_indexes,
    get_ts_between_times,
)
from .gui import plot, ui_edit_events, ui_sync
from .info import add_info, remove_info, rename_info
from .missing_samples import fill_missing_samples, isnan
from .subset import get_subset, merge
from .time import _get_time_unit, get_sample_rate, resample, shift


class TimeSeries:
    """
    A class that holds time, data series, events and metadata.

    Attributes
    ----------
    time : np.ndarray
        Time attribute as 1-dimension np.array.

    data : dict[str, np.ndarray]
        Contains the data, where each element contains a np.array
        which first dimension corresponds to time.

    events : list[TimeSeriesEvent]
        List of events.

    info : dict[str, Any]
        Contains metadata such as units or other information.

    Examples
    --------
    A TimeSeries can be constructed from another TimeSeries, a Pandas DataFrame
    or any array with at least one dimension.

    1. Creating an empty TimeSeries:

    >>> ktk.TimeSeries()
    TimeSeries with attributes:
          time: array([], dtype=float64)
          data: {}
        events: []
          info: {'Time': {'Unit': 's'}}

    2. Creating a TimeSeries and setting time and data:

    >>> ktk.TimeSeries(time=np.arange(0, 10), data={"test": np.arange(0, 10)})
    TimeSeries with attributes:
          time: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
          data: {'test': array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}
        events: []
          info: {'Time': {'Unit': 's'}}

    3. Creating a TimeSeries as a copy of another TimeSeries:

    >>> ts1 = ktk.TimeSeries(
    ...     time=np.arange(0, 10), data={"test": np.arange(0, 10)}
    ... )
    >>> ts2 = ktk.TimeSeries(ts1)
    >>> ts2
    TimeSeries with attributes:
          time: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
          data: {'test': array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}
        events: []
          info: {'Time': {'Unit': 's'}}

    See Also: TimeSeries.copy

    4. Creating a TimeSeries from a Pandas DataFrame:

    >>> df = pd.DataFrame()
    >>> df.index = [0.0, 0.1, 0.2, 0.3, 0.4]  # Time in seconds
    >>> df["x"] = [0.0, 1.0, 2.0, 3.0, 4.0]
    >>> df["y"] = [5.0, 6.0, 7.0, 8.0, 9.0]
    >>> df["z"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    >>> df
           x    y    z
    0.0  0.0  5.0  0.0
    0.1  1.0  6.0  0.0
    0.2  2.0  7.0  0.0
    0.3  3.0  8.0  0.0
    0.4  4.0  9.0  0.0

    >>> ts = ktk.TimeSeries(df)
    >>> ts
    TimeSeries with attributes:
          time: array([0. , 0.1, 0.2, 0.3, 0.4])
          data: <dict with 3 entries>
        events: []
          info: {'Time': {'Unit': 's'}}

    >>> ts.data["x"]
    array([0., 1., 2., 3., 4.])
    >>> ts.data["y"]
    array([5., 6., 7., 8., 9.])

    See Also: TimeSeries.from_dataframe

    5. Creating a multidimensional TimeSeries from a Pandas DataFrame (using
    brackets in column names):

    >>> df = pd.DataFrame()
    >>> df.index = [0.0, 0.1, 0.2, 0.3, 0.4]  # Time in seconds
    >>> df["point[:,0]"] = [0.0, 1.0, 2.0, 3.0, 4.0]
    >>> df["point[:,1]"] = [5.0, 6.0, 7.0, 8.0, 9.0]
    >>> df["point[:,2]"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    >>> df
         point[:,0]  point[:,1]  point[:,2]
    0.0         0.0         5.0         0.0
    0.1         1.0         6.0         0.0
    0.2         2.0         7.0         0.0
    0.3         3.0         8.0         0.0
    0.4         4.0         9.0         0.0

    >>> ts = ktk.TimeSeries(df)
    >>> ts.data
    {'point': array([[0., 5., 0.],
           [1., 6., 0.],
           [2., 7., 0.],
           [3., 8., 0.],
           [4., 9., 0.]])}

    See Also: TimeSeries.from_dataframe

    6. Creating a multidimensional TimeSeries of higher order from a Pandas
    DataFrame (using brackets and commas in column names):

    >>> df = pd.DataFrame()
    >>> df.index = [0.0, 0.1, 0.2, 0.3, 0.4]  # Time in seconds
    >>> df["R[:,0,0]"] = np.cos([0.0, 0.1, 0.2, 0.3, 0.4])
    >>> df["R[:,0,1]"] = -np.sin([0.0, 0.1, 0.2, 0.3, 0.4])
    >>> df["R[:,1,0]"] = np.sin([0.0, 0.1, 0.2, 0.3, 0.4])
    >>> df["R[:,1,1]"] = np.cos([0.0, 0.1, 0.2, 0.3, 0.4])
    >>> df["t[:,0]"] = [0.0, 0.1, 0.2, 0.3, 0.4]
    >>> df["t[:,1]"] = [5.0, 6.0, 7.0, 8.0, 9.0]
    >>> df
         R[:,0,0]  R[:,0,1]  R[:,1,0]  R[:,1,1]    t[:,0]    t[:,1]
    0.0  1.000000 -0.000000  0.000000  1.000000       0.0       5.0
    0.1  0.995004 -0.099833  0.099833  0.995004       0.1       6.0
    0.2  0.980067 -0.198669  0.198669  0.980067       0.2       7.0
    0.3  0.955336 -0.295520  0.295520  0.955336       0.3       8.0
    0.4  0.921061 -0.389418  0.389418  0.921061       0.4       9.0

    >>> ts = ktk.TimeSeries(df)
    >>> ts.data
    {'R': array([[[ 1.        , -0.        ],
            [ 0.        ,  1.        ]],
    <BLANKLINE>
           [[ 0.99500417, -0.09983342],
            [ 0.09983342,  0.99500417]],
    <BLANKLINE>
           [[ 0.98006658, -0.19866933],
            [ 0.19866933,  0.98006658]],
    <BLANKLINE>
           [[ 0.95533649, -0.29552021],
            [ 0.29552021,  0.95533649]],
    <BLANKLINE>
           [[ 0.92106099, -0.38941834],
            [ 0.38941834,  0.92106099]]]), 't': array([[0. , 5. ],
           [0.1, 6. ],
           [0.2, 7. ],
           [0.3, 8. ],
           [0.4, 9. ]])}

    See Also: TimeSeries.from_dataframe

    7. Creating a TimeSeries from any array (results in a TimeSeries with a
    single data key named "data" and with a matching time property with a
    period of 1 second - unless time attribute is also defined):

    >>> ktk.TimeSeries([0.1, 0.2, 0.3, 0.4, 0.5])
    TimeSeries with attributes:
          time: array([0., 1., 2., 3., 4.])
          data: {'data': array([0.1, 0.2, 0.3, 0.4, 0.5])}
        events: []
          info: {'Time': {'Unit': 's'}}

    >>> ktk.TimeSeries(
    ...     [0.1, 0.2, 0.3, 0.4, 0.5], time=[0.1, 0.2, 0.3, 0.4, 0.5]
    ... )
    TimeSeries with attributes:
          time: array([0.1, 0.2, 0.3, 0.4, 0.5])
          data: {'data': array([0.1, 0.2, 0.3, 0.4, 0.5])}
        events: []
          info: {'Time': {'Unit': 's'}}

    See Also: TimeSeries.from_array

    """

    # %% Initialization and properties

    def __init__(
        self,
        src: None | TimeSeries | pd.DataFrame | ArrayLike = None,
        *,
        time: ArrayLike | None = None,
        data: dict[str, ArrayLike] | None = None,
        events: list[TimeSeriesEvent] | None = None,
        info: dict[str, Any] | None = None,
        **kwargs,
    ):
        if time is None:
            time = []
        if data is None:
            data = {}
        if events is None:
            events = []
        if info is None:
            info = {"Time": {"Unit": "s"}}

        # Pre-0.17: time_info and data_info attributes
        if "time_info" in kwargs:
            info["Time"] = kwargs["time_info"].copy()
        if "data_info" in kwargs:
            for key in kwargs["data_info"]:
                info[key] = kwargs["data_info"][key].copy()

        # Default constructor
        if src is None:
            self.time = time
            self.data = data
            self.events = events.copy()
            self.info = info.copy()
            return

        # Else, construct based on a source:
        def _assign_self(src):
            self.time = src.time
            self.data = src.data
            self.events = src.events.copy()
            self.info = src.info.copy()

        # If src is compatible with a TimeSeries, then assign it.
        try:
            _assign_self(src)
            return
        except AttributeError:
            pass  # It was not a TimeSeries, or something compatible.

        # From DataFrame
        if isinstance(src, pd.DataFrame):
            _assign_self(
                TimeSeries.from_dataframe(
                    src,
                    events=events,
                    info=info,
                )
            )
            return

        # Else, try as an array
        _assign_self(
            TimeSeries.from_array(
                np.array(src),
                time=time,
                events=events,
                info=info,
            )
        )

    # Properties
    @property
    def time(self):
        """Time Property."""
        return self._time

    @time.setter
    def time(self, value):
        to_set = np.array(value, copy=True)
        if len(to_set.shape) != 1:
            raise AttributeError(
                "Time must be a unidimensional array. However, a value of "
                f"{value} was provided."
            )
        self._time = to_set

    @time.deleter
    def time(self):
        raise AttributeError("time property cannot be deleted.")

    @property
    def data(self):
        """Data Property."""
        return self._data

    @data.setter
    def data(self, value):
        self._data = TimeSeriesDataDict(value)

    @data.deleter
    def data(self):
        raise AttributeError("data property cannot be deleted.")

    @property
    def events(self):
        """Events Property."""
        return self._events

    @events.setter
    def events(self, value):
        self._events = TimeSeriesEventList(value)

    @events.deleter
    def events(self):
        raise AttributeError("events property cannot be deleted.")

    @property
    def info(self):
        """Info Property."""
        return self._info

    @info.setter
    def info(self, value):
        self._info = TimeSeriesInfoDict(value)

    @info.deleter
    def info(self):
        raise AttributeError("info property cannot be deleted.")

    # pre-0.17 compatibility
    @property
    def time_info(self):
        """Pre-0.17 time-info property."""
        return self.info["Time"]

    @time_info.setter
    def time_info(self, value):
        check_param("time_info", value, dict, key_type=str)
        self.info["Time"] = value

    @property
    def data_info(self):
        """Pre-0.17 data-info property."""
        return {key: self.info[key] for key in self.info if key != "Time"}

    @data_info.setter
    def data_info(self, value):
        check_param("value", value, dict, key_type=str)
        for key in value:
            check_param(f"data_info[{key}]", value, dict, key_type=str)
            self.info[key] = value[key]

    # %% Dunders

    @classmethod
    def __dir__(cls):
        """Return the directory for the TimeSeries."""
        return [
            "copy",
            # Info management
            "add_info",
            "rename_info",
            "remove_info",
            # Data management
            "get_subset",
            "merge",
            "add_data",
            "rename_data",
            "remove_data",
            # Time management
            "shift",
            "get_sample_rate",
            "resample",
            # Event management
            "add_event",
            "rename_event",
            "remove_event",
            "count_events",
            "remove_duplicate_events",
            "trim_events",
            # Get index from time
            "get_index_at_time",
            "get_index_before_time",
            "get_index_after_time",
            # Get index from event
            "get_index_at_event",
            "get_index_before_event",
            "get_index_after_event",
            # Get TimeSeries from index
            "get_ts_before_index",
            "get_ts_after_index",
            "get_ts_between_indexes",
            # Get TimeSeries from time
            "get_ts_before_time",
            "get_ts_after_time",
            "get_ts_between_times",
            # Get TimeSeries from event
            "get_ts_before_event",
            "get_ts_after_event",
            "get_ts_between_events",
            # Missing data
            "isnan",
            "fill_missing_samples",
            # Interactive and plotting
            "ui_edit_events",
            "ui_sync",
            "plot",
            # IO
            "to_dataframe",
            "from_dataframe",
            "from_array",
        ]

    def __str__(self):
        """
        Print a textual descriptive of the TimeSeries contents.

        Returns
        -------
        str
            String that describes the contents of each attribute ot the
            TimeSeries

        """
        return kineticstoolkit._repr._format_class_attributes(
            self,
            overrides={
                "_time": "time",
                "_data": "data",
                "_events": "events",
                "_info": "info",
            },
        )

    def __repr__(self):
        """Generate the class representation."""
        return str(self)

    def __eq__(self, ts):
        """
        Compare two TimeSeries for equality.

        Returns
        -------
        True if each attribute of ts is equal to the TimeSeries' attributes.

        """
        return self._is_equivalent(ts)

    # Private check methods
    _check_constant_sample_rate = _check_constant_sample_rate
    _check_increasing_time = _check_increasing_time
    _check_not_empty_data = _check_not_empty_data
    _check_not_empty_time = _check_not_empty_time
    _check_valid_time = _check_valid_time
    _check_well_shaped = _check_well_shaped
    _is_equivalent = _is_equivalent
    _raise_data_key_error = _raise_data_key_error
    _raise_info_inner_key_error = _raise_info_inner_key_error
    _raise_info_outer_key_error = _raise_info_outer_key_error

    # %% Copy
    def copy(
        self,
        *,
        copy_time: bool = True,
        copy_data: bool = True,
        copy_events: bool = True,
        copy_info: bool = True,
        **kwargs,
    ) -> TimeSeries:
        """
        Deep copy of a TimeSeries.

        Parameters
        ----------
        copy_time
            Optional. True to copy time to the new TimeSeries,
            False to keep the time attribute empty. Default is True.
        copy_data
            Optional. True to copy data to the new TimeSeries,
            False to keep the data attribute empty. Default is True.
        copy_events
            Optional. True to copy events to the new TimeSeries,
            False to keep the events attribute empty. Default is True.
        copy_info
            Optional. True to copy info to the new TimeSeries,
            False to keep the info attribute empty. Default is True.

        Returns
        -------
        TimeSeries
            A deep copy of the TimeSeries.

        """
        # Pre-0.17 compatibility
        if "copy_time_info" in kwargs or "copy_data_info" in kwargs:
            if "copy_time_info" in kwargs:
                copy_time_info = kwargs["copy_time_info"]
            else:
                copy_time_info = True  # Original default value
            if "copy_data_info" in kwargs:
                copy_data_info = kwargs["copy_data_info"]
            else:
                copy_data_info = True  # Original default value

        if (
            "copy_time_info" in kwargs and kwargs["copy_time_info"] is False
        ) or (
            "copy_data_info" in kwargs and kwargs["copy_data_info"] is False
        ):
            copy_info = False

        check_param("copy_time", copy_time, bool)
        check_param("copy_data", copy_data, bool)
        check_param("copy_events", copy_events, bool)
        check_param("copy_info", copy_info, bool)

        self._check_valid_time()

        if copy_time and copy_data and copy_events and copy_info:
            # General case
            return deepcopy(self)
        else:
            # Specific cases
            ts = TimeSeries()
            if copy_time:
                ts.time = deepcopy(self.time)
            if copy_data:
                ts.data = deepcopy(self.data)
            if copy_events:
                ts.events = deepcopy(self.events)
            if copy_info:
                ts.info = deepcopy(self.info)

            # Pre-0.17 compatibility
            if "copy_time_info" in kwargs or "copy_data_info" in kwargs:
                if copy_time_info:
                    ts.time_info = deepcopy(self.time_info)
                if copy_data_info:
                    ts.data_info = deepcopy(self.data_info)

            return ts

    # %% Imported methods

    # Info management
    add_info = add_info
    rename_info = rename_info
    remove_info = remove_info

    # Time management
    _get_time_unit = _get_time_unit
    shift = shift
    get_sample_rate = get_sample_rate
    resample = resample

    # Data management
    add_data = add_data
    rename_data = rename_data
    remove_data = remove_data

    # Event management
    _get_event_index = _get_event_index
    _get_event_indexes = _get_event_indexes
    _get_duplicate_event_indexes = _get_duplicate_event_indexes
    add_event = add_event
    rename_event = rename_event
    remove_event = remove_event
    count_events = count_events
    remove_duplicate_events = remove_duplicate_events
    trim_events = trim_events

    # Get index methods
    get_index_at_time = get_index_at_time
    get_index_before_time = get_index_before_time
    get_index_after_time = get_index_after_time
    get_index_at_event = get_index_at_event
    get_index_before_event = get_index_before_event
    get_index_after_event = get_index_after_event

    # Get ts methods
    get_ts_before_index = get_ts_before_index
    get_ts_after_index = get_ts_after_index
    get_ts_between_indexes = get_ts_between_indexes
    get_ts_before_time = get_ts_before_time
    get_ts_after_time = get_ts_after_time
    get_ts_between_times = get_ts_between_times
    get_ts_before_event = get_ts_before_event
    get_ts_after_event = get_ts_after_event
    get_ts_between_events = get_ts_between_events

    # Subsetting and merging
    get_subset = get_subset
    merge = merge

    # Missing sample management
    isnan = isnan
    fill_missing_samples = fill_missing_samples

    # Graphical user interfaces
    plot = plot
    ui_sync = ui_sync
    ui_edit_events = ui_edit_events

    # %% Input/Output

    def _to_dataframe_and_info(
        self,
    ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        """
        Implement TimeSeries.to_dataframe with additional info.

        The second element of the output tuple is a list where each element
        corresponds to a column of the DataFrame, and each element is a copy
        of the inner info dictionary for this data. For instance,
        an element of the list could be: {"Unit": "N"}.

        """
        # Init
        df_out = pd.DataFrame()
        info_out = []

        # Go through data
        the_keys = self.data.keys()
        for the_key in the_keys:
            # Assign data
            original_data = self.data[the_key]

            if original_data.shape[0] > 0:  # Not empty
                original_data_shape = original_data.shape
                data_length = original_data.shape[0]

                reshaped_data = np.reshape(original_data, (data_length, -1))
                reshaped_data_shape = reshaped_data.shape

                df_data = pd.DataFrame(reshaped_data)

                # Get the column names index from the shape of the original
                # data. The strategy here is to build matrices of indexes,
                # that have the same shape as the original data, then reshape
                # these matrices the same way we reshaped the original data.
                # Then we know where the original indexes are in the new
                # reshaped data.
                original_indexes = np.indices(original_data_shape[1:])
                reshaped_indexes = np.reshape(
                    original_indexes, (-1, reshaped_data_shape[1])
                )

                # Hint for my future self:
                # For a one-dimension series, reshaped_indexes will be:
                # [[0]].
                # For a two-dimension series, reshaped_indexes will be:
                # [[0 1 2 ...]].
                # For a three-dimension series, reshaped_indexes will be:
                # [[0 0 0 ... 1 1 1 ... 2 2 2 ...]
                #   0 1 2 ... 0 1 2 ... 0 1 2 ...]]
                # and so on.

                # Assign column names
                column_names = []
                for i_column in range(0, len(df_data.columns)):
                    this_column_name = the_key
                    n_indexes = np.shape(reshaped_indexes)[0]
                    if n_indexes > 0:
                        # This data is expressed in more than one dimension.
                        # We must add brackets to the column names to specify
                        # the indexes.
                        this_column_name += "[:,"

                        for i_indice in range(0, n_indexes):
                            this_column_name += str(
                                reshaped_indexes[i_indice, i_column]
                            )
                            if i_indice == n_indexes - 1:
                                this_column_name += "]"
                            else:
                                this_column_name += ","

                    column_names.append(this_column_name)

                df_data.columns = column_names

            else:  # empty data
                df_data = pd.DataFrame(columns=[the_key])

            # Merge this dataframe with the output dataframe
            df_out = pd.concat([df_out, df_data], axis=1)

            # Add the info that correspond to this key
            for _i in df_data.columns:
                try:
                    info = self.info[the_key]
                    info_out.append(deepcopy(info))
                except KeyError:
                    info_out.append({})

        df_out.index = self.time

        return (df_out, info_out)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Create a DataFrame by reshaping all data to one bidimensional table.

        Undimensional data is converted to a single column, and two-dimensional
        (or more) data are converted to multiple columns with the additional
        dimensions in brackets. The TimeSeries's events and info attributes are
        not included in the resulting DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with the index as the TimeSeries' time.

        See Also
        --------
        ktk.TimeSeries.from_dataframe
        ktk.TimeSeries.from_array
        ktk.TimeSeries.to_array

        Examples
        --------
        Example with unidimensional data:

        >>> ts = ktk.TimeSeries(time=np.arange(3) / 10)
        >>> ts = ts.add_data("test", np.array([0.0, 2.0, 3.0]))
        >>> ts.to_dataframe()
             test
        0.0   0.0
        0.1   2.0
        0.2   3.0

        Example with multidimensional data:

        >>> ts = ktk.TimeSeries(time=np.arange(4) / 10)
        >>> ts = ts.add_data("test", np.repeat([[0.0, 2.0, 3.0]], 4, axis=0))
        >>> ts.data["test"]
        array([[0., 2., 3.],
               [0., 2., 3.],
               [0., 2., 3.],
               [0., 2., 3.]])

        >>> ts.to_dataframe()
              test[:,0]  test[:,1]  test[:,2]
         0.0        0.0        2.0        3.0
         0.1        0.0        2.0        3.0
         0.2        0.0        2.0        3.0
         0.3        0.0        2.0        3.0

        """
        self._check_well_shaped()
        return self._to_dataframe_and_info()[0]

    @staticmethod
    def from_dataframe(
        dataframe: pd.DataFrame,
        /,
        *,
        events: list[TimeSeriesEvent] | None = None,
        info: dict[str, Any] | None = None,
        **kwargs,
    ) -> TimeSeries:
        """
        Create a new TimeSeries from a Pandas Dataframe.

        Data in column which names end with bracketed indexes such as
        [0], [1], [0,0], [0,1], etc. are converted to multidimensional
        arrays. For example, if a DataFrame has these column names::

            "Forces[0]", "Forces[1]", "Forces[2]", "Forces[3]"

        then a single data key is created ("Forces") and the shape of the
        data is Nx4.

        Parameters
        ----------
        dataframe
            A Pandas DataFrame where the index corresponds to time, and
            where each column corresponds to a data key.
        events
            Optional. Will be copied to the TimeSeries' events attribute.
        info
            Optional. Will be copied to the TimeSeries' info attribute.

        Returns
        -------
        TimeSeries
            The converted TimeSeries.

        See Also
        --------
        ktk.TimeSeries.to_dataframe
        ktk.TimeSeries.from_array
        ktk.TimeSeries.to_array

        Examples
        --------
        **Example with unidimensional data**

        Create a DataFrame with two series of 3 samples:

        >>> import pandas as pd
        >>> df = pd.DataFrame([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        >>> df.columns = ["test1", "test2"]
        >>> df
           test1  test2
        0    1.0    2.0
        1    3.0    4.0
        2    5.0    6.0

        Convert to a TimeSeries:

        >>> ts = ktk.TimeSeries.from_dataframe(df)
        >>> ts.data
        {'test1': array([1., 3., 5.]), 'test2': array([2., 4., 6.])}

        **Example with multidimensional data**

        Create a DataFrame with one series of 3 samples of dimension 2:

        >>> df = pd.DataFrame([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        >>> df.columns = ["test[0]", "test[1]"]
        >>> df
           test[0]  test[1]
        0      1.0      2.0
        1      3.0      4.0
        2      5.0      6.0

        Convert to a TimeSeries:

        >>> ts = ktk.TimeSeries.from_dataframe(df)
        >>> ts.data
        {'test': array([[1., 2.], [3., 4.], [5., 6.]])}

        **Example with even more dimensions**

        Create a DataFrame with one series of 5 samples of dimension 2x2 (rot)
        and one series of 5 samples of dimension 2 (trans):

        >>> df = pd.DataFrame()
        >>> df.index = [0.0, 0.1, 0.2, 0.3, 0.4]  # Time in seconds
        >>> df["R[:,0,0]"] = np.cos([0.0, 0.1, 0.2, 0.3, 0.4])
        >>> df["R[:,0,1]"] = -np.sin([0.0, 0.1, 0.2, 0.3, 0.4])
        >>> df["R[:,1,0]"] = np.sin([0.0, 0.1, 0.2, 0.3, 0.4])
        >>> df["R[:,1,1]"] = np.cos([0.0, 0.1, 0.2, 0.3, 0.4])
        >>> df["t[:,0]"] = [0.0, 0.1, 0.2, 0.3, 0.4]
        >>> df["t[:,1]"] = [5.0, 6.0, 7.0, 8.0, 9.0]
        >>> df
             R[:,0,0]  R[:,0,1]  R[:,1,0]  R[:,1,1]       t[:,0]    t[:,1]
        0.0  1.000000 -0.000000  0.000000  1.000000       0.0       5.0
        0.1  0.995004 -0.099833  0.099833  0.995004       0.1       6.0
        0.2  0.980067 -0.198669  0.198669  0.980067       0.2       7.0
        0.3  0.955336 -0.295520  0.295520  0.955336       0.3       8.0
        0.4  0.921061 -0.389418  0.389418  0.921061       0.4       9.0

        Convert to a TimeSeries:

        >>> ts = ktk.TimeSeries(df)
        >>> ts.data
        {'R': array([[[ 1.        , -0.        ],
                [ 0.        ,  1.        ]],
        <BLANKLINE>
               [[ 0.99500417, -0.09983342],
                [ 0.09983342,  0.99500417]],
        <BLANKLINE>
               [[ 0.98006658, -0.19866933],
                [ 0.19866933,  0.98006658]],
        <BLANKLINE>
               [[ 0.95533649, -0.29552021],
                [ 0.29552021,  0.95533649]],
        <BLANKLINE>
               [[ 0.92106099, -0.38941834],
                [ 0.38941834,  0.92106099]]]), 't': array([[0. , 5. ],
               [0.1, 6. ],
               [0.2, 7. ],
               [0.3, 8. ],
               [0.4, 9. ]])}

        """
        if events is None:
            events = []
        if info is None:
            info = {"Time": {"Unit": "s"}}

        check_param("dataframe", dataframe, pd.DataFrame)

        ts = TimeSeries(
            time=dataframe.index.to_numpy(),
            events=events,
            info=info,
        )

        # Pre-0.17: time_info and data_info attributes
        if "time_info" in kwargs:
            ts.time_info = kwargs["time_info"].copy()
        if "data_info" in kwargs:
            ts.data_info = kwargs["data_info"].copy()

        # Protect the original dataframe
        dataframe = dataframe.copy()

        # Remove spaces and ":," in indexes between brackets
        columns = dataframe.columns
        new_columns = []
        for _i_column, column in enumerate(columns):
            splitted = column.split("[")
            if len(splitted) > 1:  # There are brackets
                new_columns.append(
                    splitted[0]
                    + "["
                    + splitted[1].replace(" ", "").replace(":,", "")
                )
            else:
                new_columns.append(column)
        dataframe.columns = new_columns

        # Search for the column names and their dimensions
        # At the end, we end with something like:
        #    dimensions['Data1'] = []
        #    dimensions['Data2'] = [[0], [1], [2]]
        #    dimensions['Data3'] = [[0,0],[0,1],[1,0],[1,1]]
        dimensions = dict()  # type: dict[str, list]
        for column in dataframe.columns:
            splitted = column.split("[")
            if len(splitted) == 1:  # No brackets
                dimensions[column] = []
            else:  # With brackets
                key = splitted[0]
                index = literal_eval("[" + splitted[1])

                if key in dimensions:
                    dimensions[key].append(index)
                else:
                    dimensions[key] = [index]

        n_samples = len(dataframe)

        # Assign the columns to the output
        for key, dimension in dimensions.items():
            if len(dimension) == 0:
                ts.data[key] = dataframe[key].to_numpy()
            else:
                highest_dims = np.max(np.array(dimension), axis=0)

                columns = [
                    key + str(dim).replace(" ", "")
                    for dim in sorted(dimension)
                ]
                ts.data[key] = dataframe[columns].to_numpy()
                ts.data[key] = np.reshape(
                    ts.data[key], [n_samples] + (highest_dims + 1).tolist()
                )

        return ts

    @staticmethod
    def from_array(
        array: ArrayLike,
        /,
        *,
        data_key: str = "data",
        time: ArrayLike | None = None,
        events: list[TimeSeriesEvent] | None = None,
        info: dict[str, Any] | None = None,
        **kwargs,
    ) -> TimeSeries:
        """
        Create a new TimeSeries from an array.

        Parameters
        ----------
        array
            An array or list where the first dimension corresponds to time.
        data_key
            Optional. The name of the data (used as the key in the TimeSeries'
            data attribute). Default is "data".
        time
            Optional. An array that indicates the time for each sample. Its
            length must match the first dimension of the data array. If None
            (default), a matching time attribute of with a period of one second
            is created.
        events
            Optional. Will be copied to the TimeSeries' events attribute.
        info
            Optional. Will be copied to the TimeSeries' info attribute.

        Returns
        -------
        TimeSeries
            The new TimeSeries.

        See Also
        --------
        ktk.TimeSeries.to_array
        ktk.TimeSeries.from_dataframe
        ktk.TimeSeries.to_dataframe

        Examples
        --------
        **Using default time**

        >>> ktk.TimeSeries([0.1, 0.2, 0.3, 0.4, 0.5])
        TimeSeries with attributes:
              time: array([0., 1., 2., 3., 4.])
              data: {'data': array([0.1, 0.2, 0.3, 0.4, 0.5])}
            events: []
              info: {'Time': {'Unit': 's'}}

        **Specifiying time**

        >>> ktk.TimeSeries(
        ...     [0.1, 0.2, 0.3, 0.4, 0.5], time=[0.1, 0.2, 0.3, 0.4, 0.5]
        ... )
        TimeSeries with attributes:
              time: array([0.1, 0.2, 0.3, 0.4, 0.5])
              data: {'data': array([0.1, 0.2, 0.3, 0.4, 0.5])}
            events: []
              info: {'Time': {'Unit': 's'}}

        """
        # Default values
        if time is None:
            time = []
        if events is None:
            events = []
        if info is None:
            info = {"Time": {"Unit": "s"}}

        check_param("data_key", data_key, str)

        time = np.array(time)
        ts = TimeSeries(data={data_key: array}, events=events, info=info)

        # Pre-0.17: time_info and data_info attributes
        if "time_info" in kwargs:
            ts.time_info = kwargs["time_info"].copy()
        if "data_info" in kwargs:
            ts.data_info = kwargs["data_info"].copy()

        if time.shape[0] == 0:
            ts.time = np.arange(ts.data[data_key].shape[0]) * 1.0  # floats
        else:
            ts.time = time

        return ts

    # Deprecated methods
    sort_events = sort_events
    add_data_info = add_data_info
    remove_data_info = remove_data_info
