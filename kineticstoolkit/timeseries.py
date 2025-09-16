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


import kineticstoolkit._repr
from kineticstoolkit.decorators import deprecated
from kineticstoolkit.exceptions import (
    TimeSeriesRangeError,
    TimeSeriesEventNotFoundError,
    TimeSeriesMergeConflictError,
)
from kineticstoolkit.tools import check_interactive_backend

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd
import limitedinteraction as li
from dataclasses import dataclass
from kineticstoolkit.typing_ import ArrayLike, check_param
from typing import Any, cast
from numbers import Real

import warnings
from ast import literal_eval
from copy import deepcopy

import kineticstoolkit as ktk  # For doctests


WINDOW_PLACEMENT = {"top": 50, "right": 0}


class TimeSeriesEventList(list):
    """Event list that ensures every element is a TimeSeriesEvent."""

    def __init__(self, source: list = []):
        """Initialize the class instance using a source list."""
        check_param("source", source, list)
        for element in source:
            self.append(element)

    def __setitem__(self, index, value):
        """Cast the value to a TimeSeriesEvent."""
        check_param("index", index, int)
        try:
            event = TimeSeriesEvent(time=value.time, name=value.name)
        except AttributeError:
            raise AttributeError(
                f"The provided value {value} cannot be interpreted as a "
                "TimeSeriesEvent, because it does not have `time` and `name` "
                "attributes."
            )
        super(TimeSeriesEventList, self).__setitem__(index, event)
        # Sort the events
        self.sort()

    def append(self, value):
        """Ensure the appended value is a TimeSeriesEvent."""
        super(TimeSeriesEventList, self).append(None)
        self[-1] = value  # Calls __setitem__ which does the check

    def extend(self, values):
        """Ensure the extended values are TimeSeriesEvent."""
        for value in values:
            self.append(value)  # Calls append that calls __setitem__ that
            # does the check.


class TimeSeriesDataDict(dict):
    """Data dictionary that checks sizes and converts to NumPy arrays."""

    def __init__(self, source: dict = {}):
        """Initialize the class instance using a source dictionary."""
        check_param("source", source, dict, key_type=str)
        for key in source:
            self[key] = source[key]

    def __setitem__(self, key, value):
        """Cast the added data as a NumPy array."""
        check_param("key", key, str)
        to_set = np.array(value, copy=True)

        if len(to_set.shape) == 0:
            raise AttributeError(
                "Data must be an array. However, a value of "
                f"{value} was provided."
            )

        super(TimeSeriesDataDict, self).__setitem__(key, to_set)


@dataclass
class TimeSeriesEvent:
    """
    Define an event in a TimeSeries.

    This class is rarely used by itself, it is easier to use `TimeSeries`'
    methods to manage events.

    Attributes
    ----------
    time : float
        Event time.

    name : str
        Event name. Does not need to be unique.

    Example
    -------
    >>> event = ktk.TimeSeriesEvent(time=1.5, name="event_name")
    >>> event
    TimeSeriesEvent(time=1.5, name='event_name')

    """

    time: float = 0.0
    name: str = "event"

    def __lt__(self, other):
        """Define < operator."""
        return self.time < other.time

    def __le__(self, other):
        """Define <= operator."""
        return self.time <= other.time

    def __gt__(self, other):
        """Define > operator."""
        return self.time > other.time

    def __ge__(self, other):
        """Define >= operator."""
        return self.time >= other.time

    def _to_tuple(self) -> tuple[float, str]:
        """
        Convert a TimeSeriesEvent to a tuple.

        Example
        -------
        >>> event = ktk.TimeSeriesEvent(time=1.5, name="event_name")
        >>> event._to_tuple()
        (1.5, 'event_name')

        """
        return (self.time, self.name)

    def _to_list(self) -> list[float | str]:
        """
        Convert a TimeSeriesEvent to a list.

        Example
        -------
        >>> event = ktk.TimeSeriesEvent(time=1.5, name="event_name")
        >>> event._to_list()
        [1.5, 'event_name']

        """
        return [self.time, self.name]

    def _to_dict(self) -> dict[str, float | str]:
        """
        Convert a TimeSeriesEvent to a dict.

        Example
        -------
        >>> event = ktk.TimeSeriesEvent(time=1.5, name="event_name")
        >>> event._to_dict()
        {'Time': 1.5, 'Name': 'event_name'}

        """
        return {"Time": self.time, "Name": self.name}


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

    time_info : dict[str, Any]
        Contains metadata relative to time. The default is {"Unit": "s"}

    data_info : dict[str, dict[str, Any]]
        Contains optional metadata relative to data. For example, the
        data_info attribute could indicate the unit of data["Forces"]::

            data["Forces"] = {"Unit": "N"}

        To facilitate the management of data_info, please use
        `ktk.TimeSeries.add_data_info` and `ktk.TimeSeries.remove_data_info`.

    events : list[TimeSeriesEvent]
        List of events.

    Examples
    --------
    A TimeSeries can be constructed from another TimeSeries, a Pandas DataFrame
    or any array with at least one dimension.

    1. Creating an empty TimeSeries:

    >>> ktk.TimeSeries()
    TimeSeries with attributes:
             time: array([], dtype=float64)
             data: {}
        time_info: {'Unit': 's'}
        data_info: {}
           events: []

    2. Creating a TimeSeries and setting time and data:

    >>> ktk.TimeSeries(time=np.arange(0, 10), data={"test":np.arange(0, 10)})
    TimeSeries with attributes:
             time: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
             data: {'test': array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}
        time_info: {'Unit': 's'}
        data_info: {}
           events: []

    3. Creating a TimeSeries as a copy of another TimeSeries:

    >>> ts1 = ktk.TimeSeries(time=np.arange(0, 10), data={"test":np.arange(0, 10)})
    >>> ts2 = ktk.TimeSeries(ts1)
    >>> ts2
    TimeSeries with attributes:
             time: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
             data: {'test': array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}
        time_info: {'Unit': 's'}
        data_info: {}
           events: []

    See Also: TimeSeries.copy

    4. Creating a TimeSeries from a Pandas DataFrame:

    >>> df = pd.DataFrame()
    >>> df.index = [0., 0.1, 0.2, 0.3, 0.4]  # Time in seconds
    >>> df["x"] = [0., 1., 2., 3., 4.]
    >>> df["y"] = [5., 6., 7., 8., 9.]
    >>> df["z"] = [0., 0., 0., 0., 0.]
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
        time_info: {'Unit': 's'}
        data_info: {}
           events: []

    >>> ts.data
    {'x': array([0., 1., 2., 3., 4.]), 'y': array([5., 6., 7., 8., 9.]), 'z': array([0., 0., 0., 0., 0.])}

    See Also: TimeSeries.from_dataframe

    5. Creating a multidimensional TimeSeries from a Pandas DataFrame (using
    brackets in column names):

    >>> df = pd.DataFrame()
    >>> df.index = [0., 0.1, 0.2, 0.3, 0.4]  # Time in seconds
    >>> df["point[0]"] = [0., 1., 2., 3., 4.]
    >>> df["point[1]"] = [5., 6., 7., 8., 9.]
    >>> df["point[2]"] = [0., 0., 0., 0., 0.]
    >>> df
         point[0]  point[1]  point[2]
    0.0       0.0       5.0       0.0
    0.1       1.0       6.0       0.0
    0.2       2.0       7.0       0.0
    0.3       3.0       8.0       0.0
    0.4       4.0       9.0       0.0

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
    >>> df.index = [0., 0.1, 0.2, 0.3, 0.4]  # Time in seconds
    >>> df["rot[0,0]"] = np.cos([0., 0.1, 0.2, 0.3, 0.4])
    >>> df["rot[0,1]"] = -np.sin([0., 0.1, 0.2, 0.3, 0.4])
    >>> df["rot[1,0]"] = np.sin([0., 0.1, 0.2, 0.3, 0.4])
    >>> df["rot[1,1]"] = np.cos([0., 0.1, 0.2, 0.3, 0.4])
    >>> df["trans[0]"] = [0., 0.1, 0.2, 0.3, 0.4]
    >>> df["trans[1]"] = [5., 6., 7., 8., 9.]
    >>> df
         rot[0,0]  rot[0,1]  rot[1,0]  rot[1,1]  trans[0]  trans[1]
    0.0  1.000000 -0.000000  0.000000  1.000000       0.0       5.0
    0.1  0.995004 -0.099833  0.099833  0.995004       0.1       6.0
    0.2  0.980067 -0.198669  0.198669  0.980067       0.2       7.0
    0.3  0.955336 -0.295520  0.295520  0.955336       0.3       8.0
    0.4  0.921061 -0.389418  0.389418  0.921061       0.4       9.0

    >>> ts = ktk.TimeSeries(df)
    >>> ts.data
    {'rot': array([[[ 1.        , -0.        ],
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
            [ 0.38941834,  0.92106099]]]), 'trans': array([[0. , 5. ],
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
        time_info: {'Unit': 's'}
        data_info: {}
           events: []

    >>> ktk.TimeSeries([0.1, 0.2, 0.3, 0.4, 0.5], time=[0.1, 0.2, 0.3, 0.4, 0.5])
    TimeSeries with attributes:
             time: array([0.1, 0.2, 0.3, 0.4, 0.5])
             data: {'data': array([0.1, 0.2, 0.3, 0.4, 0.5])}
        time_info: {'Unit': 's'}
        data_info: {}
           events: []

    See Also: TimeSeries.from_array

    """

    # %% Initialization and properties

    def __init__(
        self,
        src: None | TimeSeries | pd.DataFrame | ArrayLike = None,
        *,
        time: ArrayLike = [],
        time_info: dict[str, Any] = {"Unit": "s"},
        data: dict[str, ArrayLike] = {},
        data_info: dict[str, dict[str, Any]] = {},
        events: list[TimeSeriesEvent] = [],
    ):
        # Default constructor
        if src is None:
            self.time = time
            self.data = data
            self.time_info = time_info.copy()
            self.data_info = data_info.copy()
            self.events = events.copy()
            return

        # Else, construct based on a source:
        def _assign_self(src):
            self.time = src.time
            self.data = src.data
            self.time_info = src.time_info.copy()
            self.data_info = src.data_info.copy()
            self.events = src.events.copy()

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
                    time_info=time_info,
                    data_info=data_info,
                    events=events,
                )
            )
            return

        # Else, try as an array
        _assign_self(
            TimeSeries.from_array(
                np.array(src),
                time=time,
                time_info=time_info,
                data_info=data_info,
                events=events,
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

    # %% Dunders

    @classmethod
    def __dir__(cls):
        """Return the directory for the TimeSeries."""
        return [
            "copy",
            # Data info management
            "add_data_info",
            "remove_data_info",
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
            overrides={"_time": "time", "_data": "data", "_events": "events"},
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

    # %% Private check functions

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
            ts._check_well_typed()
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
                        print(
                            f"{one_data} is missing in one of the TimeSeries"
                        )
                    return False
                except ValueError:
                    if debug:
                        print(
                            f"{one_data} does not have the same size in both "
                            "TimeSeries"
                        )
                    return False

        if self.time_info != ts.time_info:
            if debug:
                print("time_info is not equal")
            return False

        if self.data_info != ts.data_info:
            if debug:
                print("data_info is not equal")
            return False

        if self.events != ts.events:
            if debug:
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

        *except _check_increasing_time, which is run in preprocessing and not
        only in failures.

        Raises
        ------
        AttributeError
            If the TimeSeries are missing some attributes.

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
        Check that the TimeSeries' time attribute is not empty.

        Raises
        ------
        ValueError
            If the TimeSeries' time is empty

        """
        if self.time.shape[0] == 0:
            raise ValueError(
                "The TimeSeries is empty: the length of its time "
                "attribute is 0."
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
            "data_info attribute."
        )

    def _raise_data_info_key_error(self, data_key, info_key) -> None:
        raise KeyError(
            f"The key '{info_key}' was not found among the "
            f"{len(self.data_info[data_key])} key(s) of the TimeSeries' "
            f"data_info[{data_key}] attribute."
        )

    # %% Copy

    def copy(
        self,
        *,
        copy_time: bool = True,
        copy_data: bool = True,
        copy_time_info: bool = True,
        copy_data_info: bool = True,
        copy_events: bool = True,
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
        copy_time_info
            Optional. True to copy time_info to the new TimeSeries,
            False to keep the time_info attribute empty. Default is True.
        copy_data_info
            Optional. True to copy data_info to the new TimeSeries,
            False to keep the data_info attribute empty. Default is True.
        copy_events
            Optional. True to copy events to the new TimeSeries,
            False to keep the events attribute empty. Default is True.

        Returns
        -------
        TimeSeries
            A deep copy of the TimeSeries.

        """
        check_param("copy_time", copy_time, bool)
        check_param("copy_data", copy_data, bool)
        check_param("copy_time_info", copy_time_info, bool)
        check_param("copy_data_info", copy_data_info, bool)
        check_param("copy_events", copy_events, bool)
        self._check_well_typed()

        if copy_data and copy_time_info and copy_data_info and copy_events:
            # General case
            return deepcopy(self)
        else:
            # Specific cases
            ts = ktk.TimeSeries()
            if copy_time:
                ts.time = deepcopy(self.time)
            if copy_data:
                ts.data = deepcopy(self.data)
            if copy_time_info:
                ts.time_info = deepcopy(self.time_info)
            if copy_data_info:
                ts.data_info = deepcopy(self.data_info)
            if copy_events:
                ts.events = deepcopy(self.events)
            return ts

    # %% Data info management

    def add_data_info(
        self,
        data_key: str,
        info_key: str,
        value: Any,
        *,
        overwrite: bool = False,
        in_place: bool = False,
    ) -> TimeSeries:
        """
        Add metadata to TimeSeries' data.

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

        See Also
        --------
        ktk.TimeSeries.remove_data_info

        Example
        -------
        >>> ts = ktk.TimeSeries()
        >>> ts = ts.add_data_info("Forces", "Unit", "N")
        >>> ts = ts.add_data_info("Marker1", "Color", [43, 2, 255])

        >>> ts.data_info["Forces"]
        {'Unit': 'N'}

        >>> ts.data_info["Marker1"]
        {'Color': [43, 2, 255]}

        """
        check_param("data_key", data_key, str)
        check_param("info_key", info_key, str)
        check_param("overwrite", overwrite, bool)
        check_param("in_place", in_place, bool)
        self._check_well_typed()

        ts = self if in_place else self.copy()

        try:
            self.data_info[data_key][info_key]
            # It worked, therefore this data already exists.
            if overwrite is False:
                warnings.warn(
                    f"A data info with same data_key ({data_key}) and "
                    f"info_key ({info_key}) already exists in the TimeSeries. "
                    "Please use overwrite=True to suppress this warning. "
                    "This warning will become an error in Kinetics Toolkit "
                    "1.0."
                )
        except KeyError:
            pass  # This data does not exist yet.

        try:
            ts.data_info[data_key][info_key] = value
        except KeyError:
            ts.data_info[data_key] = {info_key: value}
        return ts

    def remove_data_info(
        self, data_key: str, info_key: str, *, in_place: bool = False
    ) -> TimeSeries:
        """
        Remove metadata from a TimeSeries' data.

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

        See Also
        --------
        ktk.TimeSeries.add_data_info

        Example
        -------
        >>> ts = ktk.TimeSeries()
        >>> ts = ts.add_data_info("Forces", "Unit", "N")
        >>> ts.data_info["Forces"]
        {'Unit': 'N'}

        >>> ts = ts.remove_data_info("Forces", "Unit")
        >>> ts.data_info["Forces"]
        {}

        """
        check_param("data_key", data_key, str)
        check_param("info_key", info_key, str)
        check_param("in_place", in_place, bool)
        self._check_well_typed()

        ts = self if in_place else self.copy()
        try:
            data_info = ts.data_info[data_key]
            try:
                data_info.pop(info_key)
            except KeyError:
                self._raise_data_info_key_error(data_key, info_key)
        except KeyError:
            self._raise_data_key_error(data_key)
        return ts

    # %% Data management

    def add_data(
        self,
        data_key: str,
        data_value: ArrayLike,
        *,
        overwrite: bool = False,
        in_place: bool = False,
    ) -> TimeSeries:
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
            In any of these conditions:
            If data with this key already exists and overwrite is False.
            If the size of the data (first dimension) does not match the size
            of existing data or the existing time.
            If data is a pandas DataFrame and its index does not match the
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
            time_info: {'Unit': 's'}
            data_info: {}
               events: []

        # Size matching example
        >>> ts = ktk.TimeSeries(time = [0.0, 0.1, 0.2, 0.3])
        >>> ts = ts.add_data("data1", [9.9])
        >>> ts
        TimeSeries with attributes:
                 time: array([0. , 0.1, 0.2, 0.3])
                 data: {'data1': array([9.9, 9.9, 9.9, 9.9])}
            time_info: {'Unit': 's'}
            data_info: {}
               events: []

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
    ) -> TimeSeries:
        """
        Rename a key in data and data_info.

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

        Example
        -------
        >>> ts = ktk.TimeSeries()
        >>> ts = ts.add_data("test", np.arange(10))
        >>> ts = ts.add_data_info("test", "Unit", "m")

        >>> ts.data
        {'test': array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}

        >>> ts.data_info
        {'test': {'Unit': 'm'}}

        >>> ts = ts.rename_data("test", "signal")

        >>> ts.data
        {'signal': array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}

        >>> ts.data_info
        {'signal': {'Unit': 'm'}}

        """
        check_param("old_data_key", old_data_key, str)
        check_param("new_data_key", new_data_key, str)
        check_param("in_place", in_place, bool)
        self._check_well_typed()

        ts = self if in_place else self.copy()
        try:
            ts.data[new_data_key] = ts.data.pop(old_data_key)
        except KeyError:
            self._raise_data_key_error(old_data_key)

        try:
            ts.data_info[new_data_key] = ts.data_info.pop(old_data_key)
        except KeyError:
            pass  # It's okay if there was no data info for this data_key

        return ts

    def remove_data(
        self, data_key: str, *, in_place: bool = False
    ) -> TimeSeries:
        """
        Remove a data key and its associated metadata.

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

        Example
        -------
        >>> # Prepare a test TimeSeries with data "test"
        >>> ts = ktk.TimeSeries()
        >>> ts = ts.add_data("test", np.arange(10))
        >>> ts = ts.add_data_info("test", "Unit", "m")

        >>> ts.data
        {'test': array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}

        >>> ts.data_info
        {'test': {'Unit': 'm'}}

        >>> # Now remove data "test"
        >>> ts = ts.remove_data("test")

        >>> ts.data
        {}

        >>> ts.data_info
        {}

        """
        check_param("data_key", data_key, str)
        check_param("in_place", in_place, bool)
        self._check_well_typed()

        ts = self if in_place else self.copy()
        try:
            ts.data.pop(data_key)
        except KeyError:
            self._raise_data_key_error(data_key)
        try:
            ts.data_info.pop(data_key)
        except KeyError:
            pass  # It's okay if there was no data info for this data_key

        return ts

    # %% Event management

    def _get_event_indexes(self, name: str) -> list[int]:
        """
        Get a list of index of all occurrences of an event.

        Parameters
        ----------
        name
            Name of the event to look for in the events list.

        Returns
        -------
        list[int]
            The occurrences of this event.

        """
        self._check_well_typed()

        # list all events with correct name
        event_times = []
        event_indexes = []
        for i_event, event in enumerate(self.events):
            if event.name == name:
                event_times.append(event.time)
                event_indexes.append(i_event)

        # Sort the indexes by time
        sorted_indexes = np.argsort(event_times)
        event_indexes = [event_indexes[i] for i in sorted_indexes]
        return event_indexes

    def _get_event_index(self, name: str, occurrence: int = 0) -> int:
        """
        Get the events index of a given occurrence of an event name.

        Parameters
        ----------
        name
            Name of the event to look for in the events list.

        occurrence
            Occurrence of the event

        Returns
        -------
        int
            The index of the event occurrence in the events list.

        Raises
        ------
        TimeSeriesEventNotFoundError
            If the specified occurrence could not be found.

        """
        self._check_well_typed()

        occurrence = int(occurrence)

        if occurrence < 0:
            raise TimeSeriesEventNotFoundError(
                "The parameter `occurrence` must be positive a integer. "
                f"However, a value of {occurrence} was received."
            )

        # Get the event occurrence
        try:
            return self._get_event_indexes(name)[occurrence]
        except IndexError:
            raise TimeSeriesEventNotFoundError(
                f"The occurrence {occurrence} of event '{name}' could not "
                "be found in the TimeSeries. A total of "
                f"{len(self._get_event_indexes(name))} occurrence(s) of "
                "this event name were found."
            )

    def _get_duplicate_event_indexes(self) -> list[int]:
        """
        Find events with same name and same time.

        Returns
        -------
        list[int]
            A list of list of event indexes. The outer list corresponds to
            different events. The inner list corresponds to all occurences of
            this event. The integer corresponds to the event index in the
            TimeSeries' event list.

        Example
        -------
        >>> ts = ktk.TimeSeries()

        # Three occurrences of event1
        >>> ts = ts.add_event(0.0, "event1")
        >>> ts = ts.add_event(1E-12, "event1")
        >>> ts = ts.add_event(0.0, "event1")

        # One occurrence of event2, but also at 0.0 second
        >>> ts = ts.add_event(0.0, "event2")

        # Two occurrences of event3
        >>> ts = ts.add_event(2.0, "event3")
        >>> ts = ts.add_event(2.0, "event3")

        """
        self._check_well_typed()

        # Sort all events in a dict with key being tuple(time, name)
        sorted_events = {}  # type: dict[tuple[float, str], list[int]]
        for i_event, event in enumerate(self.events):
            tup_event = event._to_tuple()

            # Check if this event already exist in the list.
            # If it does, add it to the list.
            found = False
            for key in sorted_events:
                if np.isclose(key[0], event.time) and (key[1] == event.name):
                    sorted_events[key].append(i_event)
                    found = True
                    break
            if not found:
                # Otherwise, create it in the list
                sorted_events[tup_event] = [i_event]

        # Convert this dict to the desired list of lists
        out = []
        for key in sorted_events:
            if len(sorted_events[key]) > 1:
                out.extend(sorted_events[key][1:])

        return sorted(out)

    def add_event(
        self,
        time: float,
        name: str = "event",
        *,
        in_place: bool = False,
        unique: bool = False,
    ) -> TimeSeries:
        """
        Add an event to the TimeSeries.

        Parameters
        ----------
        time
            The time of the event, in the same unit as `time_info["Unit"]`.
        name
            Optional. The name of the event. Default is "event".
        in_place
            Optional. True to modify and return the original TimeSeries. False
            to return a modified copy of the TimeSeries while leaving the
            original TimeSeries intact. Default is False.
        unique
            Optional. True to prevent duplicating an already existing event. In
            this case, if an event with the same time and name already exists,
            no event is added. Default is False.

        Returns
        -------
        TimeSeries
            The TimeSeries with the added event.

        See Also
        --------
        ktk.TimeSeries.rename_event
        ktk.TimeSeries.remove_event
        ktk.TimeSeries.trim_events
        ktk.TimeSeries.ui_edit_events

        Example
        -------
        >>> ts = ktk.TimeSeries()
        >>> ts = ts.add_event(5.5, "event1")
        >>> ts = ts.add_event(10.8, "event2")
        >>> ts = ts.add_event(20.3, "event2")

        >>> ts.events
        [TimeSeriesEvent(time=5.5, name='event1'),
         TimeSeriesEvent(time=10.8, name='event2'),
         TimeSeriesEvent(time=20.3, name='event2')]

        """
        check_param("time", time, float)
        check_param("name", name, str)
        check_param("in_place", in_place, bool)
        check_param("unique", unique, bool)
        self._check_well_typed()

        ts = self if in_place else self.copy()

        if unique:
            # Ensure that no event of that name and time already exists
            for event in ts.events:
                if np.isclose(time, event.time) and (name == event.name):
                    return ts

        ts.events.append(TimeSeriesEvent(time, name))
        return ts

    def rename_event(
        self,
        old_name: str,
        new_name: str,
        occurrence: int | None = None,
        *,
        in_place: bool = False,
    ) -> TimeSeries:
        """
        Rename an event occurrence or all events of a same name.

        Parameters
        ----------
        old_name
            Name of the event to look for in the events list.
        new_name
            New event name
        occurrence
            Optional. i_th occurence of the event to look for in the events
            list, starting at 0, where the occurrences are sorted in time.
            If None (default), all occurences of this event name are renamed.
        in_place
            Optional. True to modify and return the original TimeSeries. False
            to return a modified copy of the TimeSeries while leaving the
            original TimeSeries intact. Default is False.

        Returns
        -------
        TimeSeries
            The TimeSeries with the renamed event.

        See Also
        --------
        ktk.TimeSeries.add_event
        ktk.TimeSeries.remove_event
        ktk.TimeSeries.trim_events
        ktk.TimeSeries.ui_edit_events

        Example
        -------
        >>> ts = ktk.TimeSeries()
        >>> ts = ts.add_event(5.5, "event1")
        >>> ts = ts.add_event(10.8, "event2")
        >>> ts = ts.add_event(20.3, "event2")

        >>> ts.events
        [TimeSeriesEvent(time=5.5, name='event1'),
         TimeSeriesEvent(time=10.8, name='event2'),
         TimeSeriesEvent(time=20.3, name='event2')]

        >>> ts = ts.rename_event("event2", "event3")
        >>> ts.events
        [TimeSeriesEvent(time=5.5, name='event1'),
         TimeSeriesEvent(time=10.8, name='event3'),
         TimeSeriesEvent(time=20.3, name='event3')]

        >>> ts = ts.rename_event("event3", "event4", occurrence=0)
        >>> ts.events
        [TimeSeriesEvent(time=5.5, name='event1'),
         TimeSeriesEvent(time=10.8, name='event4'),
         TimeSeriesEvent(time=20.3, name='event3')]

        """
        check_param("old_name", old_name, str)
        check_param("new_name", new_name, str)
        check_param("occurrence", occurrence, (int, None))
        check_param("in_place", in_place, bool)
        self._check_well_typed()

        ts = self if in_place else self.copy()

        if old_name == new_name:
            return ts

        if occurrence is None:
            # Rename every occurrence of this event
            for index in self._get_event_indexes(old_name):
                ts.events[index].name = new_name
        else:
            index = self._get_event_index(old_name, occurrence)
            ts.events[index].name = new_name
        return ts

    def remove_event(
        self,
        name: str,
        occurrence: int | None = None,
        *,
        in_place: bool = False,
    ) -> TimeSeries:
        """
        Remove an event occurrence or all events of a same name.

        Parameters
        ----------
        name
            Name of the event to look for in the events list.
        occurrence
            Optional. i_th occurence of the event to look for in the events
            list, starting at 0, where the occurrences are sorted in time.
            If None (default), all occurences of this event name or removed.
        in_place
            Optional. True to modify and return the original TimeSeries. False
            to return a modified copy of the TimeSeries while leaving the
            original TimeSeries intact. Default is False.

        Returns
        -------
        TimeSeries
            The TimeSeries with the removed event.

        See Also
        --------
        ktk.TimeSeries.add_event
        ktk.TimeSeries.rename_event
        ktk.TimeSeries.trim_events
        ktk.TimeSeries.ui_edit_events

        Example
        -------
        >>> # Instanciate a TimeSeries with some events
        >>> ts = ktk.TimeSeries()
        >>> ts = ts.add_event(5.5, "event1")
        >>> ts = ts.add_event(10.8, "event2")
        >>> ts = ts.add_event(20.3, "event2")

        >>> ts.events
        [TimeSeriesEvent(time=5.5, name='event1'),
         TimeSeriesEvent(time=10.8, name='event2'),
         TimeSeriesEvent(time=20.3, name='event2')]

        >>> ts = ts.remove_event("event1")
        >>> ts.events
        [TimeSeriesEvent(time=10.8, name='event2'),
         TimeSeriesEvent(time=20.3, name='event2')]

        >>> ts = ts.remove_event("event2", 1)
        >>> ts.events
        [TimeSeriesEvent(time=10.8, name='event2')]

        """
        check_param("name", name, str)
        check_param("occurrence", occurrence, (int, None))
        check_param("in_place", in_place, bool)
        self._check_well_typed()

        ts = self if in_place else self.copy()

        if occurrence is None:  # Remove all occurrences
            event_index = ts._get_event_index(name, 0)
            try:
                # Continually remove the first event of this name, until
                # there are no more.
                count = 0
                while True:
                    ts.remove_event(name, occurrence=0, in_place=True)
                    count += 1
            except TimeSeriesEventNotFoundError:
                if count == 0:  # No event of that name was even found.
                    raise TimeSeriesEventNotFoundError(
                        f"No event named {name} could be found."
                    )

        else:  # Remove only the specified occurrence
            event_index = ts._get_event_index(name, occurrence)
            ts.events.pop(event_index)
        return ts

    def count_events(self, name: str) -> int:
        """
        Count the number of occurrence of a given event name.

        Parameters
        ----------
        name
            The name of the events to count.

        Returns
        -------
        int
            The number of occurrences.

        Example
        -------
        >>> # Instanciate a TimeSeries with some events
        >>> ts = ktk.TimeSeries()
        >>> ts = ts.add_event(5.5, "event1")
        >>> ts = ts.add_event(10.8, "event2")
        >>> ts = ts.add_event(20.3, "event2")

        >>> ts.count_events("event2")
        2

        """
        check_param("name", name, str)
        self._check_well_typed()

        indexes = self._get_event_indexes(name)
        return len(indexes)

    def remove_duplicate_events(self, *, in_place: bool = False) -> TimeSeries:
        """
        Remove events with same name and time so that each event gets unique.

        Parameters
        ----------
        in_place
            Optional. True to modify and return the original TimeSeries. False
            to return a modified copy of the TimeSeries while leaving the
            original TimeSeries intact. Default is False.

        Returns
        -------
        TimeSeries
            The TimeSeries with only unique events.

        Example
        -------
        >>> ts = ktk.TimeSeries()

        Three occurrences of event1:

        >>> ts = ts.add_event(0.0, "event1")
        >>> ts = ts.add_event(1E-12, "event1")
        >>> ts = ts.add_event(0.0, "event1")

        One occurrence of event2, but also at 0.0 second:

        >>> ts = ts.add_event(0.0, "event2")

        Two occurrences of event3:

        >>> ts = ts.add_event(2.0, "event3")
        >>> ts = ts.add_event(2.0, "event3")

        >>> ts.events
        [TimeSeriesEvent(time=0.0, name='event1'),
         TimeSeriesEvent(time=0.0, name='event1'),
         TimeSeriesEvent(time=0.0, name='event2'),
         TimeSeriesEvent(time=1e-12, name='event1'),
         TimeSeriesEvent(time=2.0, name='event3'),
         TimeSeriesEvent(time=2.0, name='event3')]

        >>> ts2 = ts.remove_duplicate_events()
        >>> ts2.events
        [TimeSeriesEvent(time=0.0, name='event1'),
         TimeSeriesEvent(time=0.0, name='event2'),
         TimeSeriesEvent(time=2.0, name='event3')]

        """
        check_param("in_place", in_place, bool)
        self._check_well_typed()

        ts = self if in_place else self.copy()
        duplicates = ts._get_duplicate_event_indexes()
        for event_index in duplicates[-1::-1]:
            ts.events.pop(event_index)
        return ts

    def trim_events(self, *, in_place: bool = False) -> TimeSeries:
        """
        Delete the events that are outside the TimeSeries' time attribute.

        Parameters
        ----------
        in_place
            Optional. True to modify and return the original TimeSeries. False
            to return a modified copy of the TimeSeries while leaving the
            original TimeSeries intact. Default is False.

        Returns
        -------
        TimeSeries
            The TimeSeries without the trimmed events.

        See Also
        --------
        ktk.TimeSeries.add_event
        ktk.TimeSeries.rename_event
        ktk.TimeSeries.remove_event
        ktk.TimeSeries.ui_edit_events

        Example
        -------
        >>> ts = ktk.TimeSeries(time = np.arange(10))
        >>> ts.time
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        >>> ts = ts.add_event(-2)
        >>> ts = ts.add_event(0)
        >>> ts = ts.add_event(5)
        >>> ts = ts.add_event(9)
        >>> ts = ts.add_event(10)
        >>> ts.events
        [TimeSeriesEvent(time=-2, name='event'),
         TimeSeriesEvent(time=0, name='event'),
         TimeSeriesEvent(time=5, name='event'),
         TimeSeriesEvent(time=9, name='event'),
         TimeSeriesEvent(time=10, name='event')]

        >>> ts = ts.trim_events()
        >>> ts.events
        [TimeSeriesEvent(time=0, name='event'),
         TimeSeriesEvent(time=5, name='event'),
         TimeSeriesEvent(time=9, name='event')]

        """
        check_param("in_place", in_place, bool)
        self._check_well_typed()

        ts = self if in_place else self.copy()

        events = deepcopy(ts.events)
        ts.events = []
        for event in events:
            if event.time <= np.max(ts.time) and event.time >= np.min(ts.time):
                ts.add_event(event.time, event.name, in_place=True)
        return ts

    # %% get_index methods

    def get_index_at_time(self, time: float) -> int:
        """
        Get the time index that is closest to the specified time.

        Parameters
        ----------
        time
            Time to look for in the TimeSeries' time attribute.

        Returns
        -------
        int
            The index in the time attribute.

        See Also
        --------
        ktk.TimeSeries.get_index_before_time
        ktk.TimeSeries.get_index_after_time
        ktk.TimeSeries.get_index_before_event
        ktk.TimeSeries.get_index_at_event
        ktk.TimeSeries.get_index_after_event


        Example
        -------
        >>> ts = ktk.TimeSeries(time=np.array([0, 0.5, 1, 1.5, 2]))

        >>> ts.get_index_at_time(0.9)
        2

        >>> ts.get_index_at_time(1)
        2

        >>> ts.get_index_at_time(1.1)
        2

        >>> ts.get_index_at_time(2.1)
        4

        """
        check_param("time", time, float)
        self._check_well_shaped()

        self._check_not_empty_time()
        return int(np.argmin(np.abs(self.time - float(time))))

    def get_index_before_time(
        self, time: float, *, inclusive: bool = False
    ) -> int:
        """
        Get the time index that is just before the specified time.

        Parameters
        ----------
        time
            Time to look for in the TimeSeries' time attribute.
        inclusive
            Optional. True to include the given time in the comparison.

        Returns
        -------
        int
            The index in the time attribute.

        Raises
        ------
        TimeSeriesRangeError
            If the resulting index would be outside the TimeSeries range.

        See Also
        --------
        ktk.TimeSeries.get_index_at_time
        ktk.TimeSeries.get_index_after_time
        ktk.TimeSeries.get_index_before_event
        ktk.TimeSeries.get_index_at_event
        ktk.TimeSeries.get_index_after_event

        Example
        -------
        >>> ts = ktk.TimeSeries(time=np.array([0, 0.5, 1, 1.5, 2]))

        >>> ts.get_index_before_time(0.9)
        1

        >>> ts.get_index_before_time(1)
        1

        >>> ts.get_index_before_time(1.1)
        2

        >>> ts.get_index_before_time(1.1, inclusive=True)
        2

        """
        check_param("time", time, float)
        check_param("inclusive", inclusive, bool)
        self._check_well_shaped()

        def _raise():
            raise TimeSeriesRangeError(
                f"There is no data before the requested time of {time} "
                f"{self.time_info['Unit']}."
            )

        self._check_increasing_time()

        if inclusive:
            mask = np.nonzero(self.time <= time)
        else:
            mask = np.nonzero(self.time < time)

        if mask[0].shape == (0,):
            _raise()

        return int(mask[0][-1])

    def get_index_after_time(
        self, time: float, *, inclusive: bool = False
    ) -> int:
        """
        Get the time index that is just after the specified time.

        Parameters
        ----------
        time
            Time to look for in the TimeSeries' time attribute.
        inclusive
            Optional. True to include the given time in the comparison.

        Returns
        -------
        int
            The index in the time attribute.

        Raises
        ------
        TimeSeriesRangeError
            If the resulting index would be outside the TimeSeries range.

        See Also
        --------
        ktk.TimeSeries.get_index_before_time
        ktk.TimeSeries.get_index_at_time
        ktk.TimeSeries.get_index_before_event
        ktk.TimeSeries.get_index_at_event
        ktk.TimeSeries.get_index_after_event

        Example
        -------
        >>> ts = ktk.TimeSeries(time=np.array([0, 0.5, 1, 1.5, 2]))

        >>> ts.get_index_after_time(0.9)
        2

        >>> ts.get_index_after_time(0.9, inclusive=True)
        2

        >>> ts.get_index_after_time(1)
        3

        >>> ts.get_index_after_time(1, inclusive=True)
        2

        """
        check_param("time", time, float)
        check_param("inclusive", inclusive, bool)
        self._check_well_shaped()

        def _raise():
            raise TimeSeriesRangeError(
                f"There is no data before the requested time of {time} "
                f"{self.time_info['Unit']}."
            )

        self._check_increasing_time()

        if inclusive:
            mask = np.nonzero(self.time >= time)
        else:
            mask = np.nonzero(self.time > time)

        if mask[0].shape == (0,):
            _raise()

        return int(mask[0][0])

    def get_index_at_event(self, name: str, occurrence: int = 0) -> int:
        """
        Get the time index that is closest to the specified event occurrence.

        Parameters
        ----------
        name
            Event name
        occurrence
            Occurrence of the event. The default is 0.

        Returns
        -------
        int
            The index in the time attribute.

        See Also
        --------
        ktk.TimeSeries.get_index_before_time
        ktk.TimeSeries.get_index_at_time
        ktk.TimeSeries.get_index_after_time
        ktk.TimeSeries.get_index_before_event
        ktk.TimeSeries.get_index_after_event

        Example
        -------
        >>> ts = ktk.TimeSeries(time=np.arange(10)/10)
        >>> ts = ts.add_event(0.2, "event")
        >>> ts = ts.add_event(0.36, "event")
        >>> ts.time
        array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        >>> ts.get_index_at_event("event")
        2

        >>> ts.get_index_at_event("event", occurrence=1)
        4

        """
        check_param("name", name, str)
        check_param("occurrence", occurrence, int)
        self._check_well_shaped()

        return self.get_index_at_time(
            self.events[self._get_event_index(name, occurrence)].time
        )

    def get_index_before_event(
        self, name: str, occurrence: int = 0, inclusive: bool = False
    ) -> int:
        """
        Get the time index that is just before the specified event occurrence.

        Parameters
        ----------
        name
            Event name
        occurrence
            Occurrence of the event. The default is 0.
        inclusive
            True to allow including one sample after the event if needed, to
            make sure that the event time is part of the returned TimeSeries's
            time. False to make sure that the returned TimeSeries does not
            include the event time. Default is False.

        Returns
        -------
        int
            The index in the time attribute.

        Raises
        ------
        TimeSeriesRangeError
            If the resulting index would be outside the TimeSeries range.

        See Also
        --------
        ktk.TimeSeries.get_index_before_time
        ktk.TimeSeries.get_index_at_time
        ktk.TimeSeries.get_index_after_time
        ktk.TimeSeries.get_index_at_event
        ktk.TimeSeries.get_index_after_event

        Example
        -------
        >>> ts = ktk.TimeSeries(time=np.arange(10)/10)
        >>> ts = ts.add_event(0.2, "event")
        >>> ts = ts.add_event(0.36, "event")
        >>> ts.time
        array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        >>> ts.get_index_before_event("event")
        1

        >>> ts.get_index_before_event("event", occurrence=1)
        3

        >>> ts.get_index_before_event("event", occurrence=0, inclusive=True)
        2

        """
        check_param("name", name, str)
        check_param("occurrence", occurrence, int)
        check_param("inclusive", inclusive, bool)
        self._check_well_shaped()

        if inclusive is False:
            return self.get_index_before_time(
                self.events[self._get_event_index(name, occurrence)].time,
                inclusive=False,
            )
        else:
            return self.get_index_after_time(
                self.events[self._get_event_index(name, occurrence)].time,
                inclusive=True,
            )

    def get_index_after_event(
        self, name: str, occurrence: int = 0, inclusive: bool = False
    ) -> int:
        """
        Get the time index that is just after the specified event occurrence.

        Parameters
        ----------
        name
            Event name
        occurrence
            Occurrence of the event. The default is 0.
        inclusive
            True to allow including one sample before the event if needed, to
            make sure that the event time is part of the output TimeSeries's
            time. False to make sure that the returned TimeSeries does not
            include the event time. Default is False.

        Returns
        -------
        int
            The index in the time attribute.

        Raises
        ------
        TimeSeriesRangeError
            If the resulting index would be outside the TimeSeries range.

        See Also
        --------
        ktk.TimeSeries.get_index_before_time
        ktk.TimeSeries.get_index_at_time
        ktk.TimeSeries.get_index_after_time
        ktk.TimeSeries.get_index_before_event
        ktk.TimeSeries.get_index_at_event

        Example
        -------
        >>> ts = ktk.TimeSeries(time=np.arange(10)/10)
        >>> ts = ts.add_event(0.2, "event")
        >>> ts = ts.add_event(0.36, "event")
        >>> ts.time
        array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        >>> ts.get_index_after_event("event")
        3

        >>> ts.get_index_after_event("event", occurrence=1)
        4

        >>> ts.get_index_after_event("event", inclusive=True)
        2

        """
        check_param("name", name, str)
        check_param("occurrence", occurrence, int)
        check_param("inclusive", inclusive, bool)
        self._check_well_shaped()

        if inclusive is False:
            return self.get_index_after_time(
                self.events[self._get_event_index(name, occurrence)].time,
                inclusive=False,
            )
        else:
            return self.get_index_before_time(
                self.events[self._get_event_index(name, occurrence)].time,
                inclusive=True,
            )

    # %% get_ts methods

    def get_ts_before_index(
        self, index: int, *, inclusive: bool = False
    ) -> TimeSeries:
        """
        Get a TimeSeries before the specified time index.

        Parameters
        ----------
        index
            Time index
        inclusive
            Optional. True to include the given time index.

        Returns
        -------
        TimeSeries
            A new TimeSeries that fulfils the specified conditions.

        Raises
        ------
        TimeSeriesRangeError
            If there is no data before the specified index.

        See Also
        --------
        ktk.TimeSeries.get_ts_before_time
        ktk.TimeSeries.get_ts_before_event
        ktk.TimeSeries.get_ts_after_index
        ktk.TimeSeries.get_ts_between_indexes

        Example
        -------
        >>> ts = ktk.TimeSeries(time=np.arange(10)/10)
        >>> ts.time
        array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        >>> ts.get_ts_before_index(2).time
        array([0. , 0.1])

        >>> ts.get_ts_before_index(2, inclusive=True).time
        array([0. , 0.1, 0.2])

        """
        check_param("index", index, int)
        check_param("inclusive", inclusive, bool)
        self._check_well_shaped()
        self._check_increasing_time()

        if (inclusive and (index < 0)) or (not inclusive and (index <= 0)):
            raise TimeSeriesRangeError(
                "Negative indexing is not supported in TimeSeries."
            )

        return self.get_ts_between_indexes(
            0, index, inclusive=(True, inclusive)
        )

    def get_ts_after_index(
        self, index: int, *, inclusive: bool = False
    ) -> TimeSeries:
        """
        Get a TimeSeries after the specified time index.

        Parameters
        ----------
        index
            Time index
        inclusive
            Optional. True to include the given time index.

        Returns
        -------
        TimeSeries
            A new TimeSeries that fulfils the specified conditions.

        Raises
        ------
        TimeSeriesRangeError
            If there is no data after the specified index.

        See Also
        --------
        ktk.TimeSeries.get_ts_after_time
        ktk.TimeSeries.get_ts_after_event
        ktk.TimeSeries.get_ts_before_index
        ktk.TimeSeries.get_ts_between_indexes

        Example
        -------
        >>> ts = ktk.TimeSeries(time=np.arange(10)/10)
        >>> ts.time
        array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        >>> ts.get_ts_after_index(2).time
        array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        >>> ts.get_ts_after_index(2, inclusive=True).time
        array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        """
        check_param("index", index, int)
        check_param("inclusive", inclusive, bool)
        self._check_well_shaped()
        self._check_increasing_time()

        if (inclusive and (index > self.time.shape[0] - 1)) or (
            not inclusive and (index >= self.time.shape[0] - 1)
        ):
            raise TimeSeriesRangeError(
                "There is no data in this TimeSeries after the specified "
                f"index of {index} since the time of this TimeSeries has a "
                f"shape of {self.time.shape}."
            )

        return self.get_ts_between_indexes(
            index, self.time.shape[0] - 1, inclusive=(inclusive, True)
        )

    def get_ts_between_indexes(
        self,
        index1: int,
        index2: int,
        *,
        inclusive: bool | tuple[bool, bool] = False,
    ) -> TimeSeries:
        """
        Get a TimeSeries between two specified time indexes.

        Parameters
        ----------
        index1, index2
            Time indexes
        inclusive
            Optional. Either a bool or a tuple of two bools. Used to
            specify which indexes are returned:

            - False or (False, False) (default): index1 < index < index2
            - True or (True, True): index1 <= index <= index2
            - (True, False): index1 <= index < index2
            - (False, True): index1 < index <= index2

        Returns
        -------
        TimeSeries
            A new TimeSeries that fulfils the specified conditions.

        Raises
        ------
        TimeSeriesRangeError
            If there is no data between the specified indexes.


        See Also
        --------
        ktk.TimeSeries.get_ts_between_times
        ktk.TimeSeries.get_ts_between_events
        ktk.TimeSeries.get_ts_before_index
        ktk.TimeSeries.get_ts_after_index

        Example
        -------
        >>> ts = ktk.TimeSeries(time=np.arange(10)/10)
        >>> ts.time
        array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        >>> ts.get_ts_between_indexes(2, 5).time
        array([0.3, 0.4])

        >>> ts.get_ts_between_indexes(2, 5, inclusive=True).time
        array([0.2, 0.3, 0.4, 0.5])

        >>> ts.get_ts_between_indexes(2, 5, inclusive=[True, False]).time
        array([0.2, 0.3, 0.4])

        """
        check_param("index1", index1, int)
        check_param("index2", index2, int)
        if isinstance(inclusive, bool):
            inclusive = (inclusive, inclusive)
        try:
            inclusive = cast(tuple[bool, bool], tuple(inclusive))
            check_param(
                "inclusive",
                inclusive,
                tuple,
                length=2,
                contents_type=bool,
            )
        except TypeError:
            raise TypeError(
                "inclusive must be either a bool or a tuple of two bools."
            )

        self._check_well_shaped()
        self._check_increasing_time()

        if index2 < index1:
            raise ValueError(
                "The parameter index2 must be higher than index1. "
                f"However, index2 is {index2} while index1 is {index1}."
            )

        if index1 < 0 or index1 >= len(self.time):
            raise TimeSeriesRangeError(
                f"The specified index1 of {index1} is out of "
                f"range. The TimeSeries has {len(self.time)} samples."
            )
        index1 -= int(inclusive[0])

        if index2 < 0 or index2 >= len(self.time):
            raise TimeSeriesRangeError(
                f"The specified index2 of {index2} is out of "
                f"range. The TimeSeries has {len(self.time)} samples."
            )
        index2 += int(inclusive[1])

        index_range = range(index1 + 1, index2)

        out_ts = self.copy(copy_data=False, copy_time=False)
        out_ts.time = self.time[index_range]
        for the_data in self.data.keys():
            out_ts.data[the_data] = self.data[the_data][index_range]
        return out_ts

    def get_ts_before_time(
        self, time: float, *, inclusive: bool = False
    ) -> TimeSeries:
        """
        Get a TimeSeries before the specified time.

        Parameters
        ----------
        time
            Time to look for in the TimeSeries' time attribute.
        inclusive
            Optional. True to include the given time in the comparison.

        Returns
        -------
        TimeSeries
            A new TimeSeries that fulfils the specified conditions.

        Raises
        ------
        TimeSeriesRangeError
            If there is no data before the specified time.

        See Also
        --------
        ktk.TimeSeries.get_ts_before_index
        ktk.TimeSeries.get_ts_before_event
        ktk.TimeSeries.get_ts_after_time
        ktk.TimeSeries.get_ts_between_times

        Example
        -------
        >>> ts = ktk.TimeSeries(time=np.arange(10)/10)
        >>> ts.time
        array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        >>> ts.get_ts_before_time(0.3).time
        array([0. , 0.1, 0.2])

        >>> ts.get_ts_before_time(0.3, inclusive=True).time
        array([0. , 0.1, 0.2, 0.3])

        """
        check_param("time", time, float)
        check_param("inclusive", inclusive, bool)
        self._check_well_shaped()
        self._check_increasing_time()

        if (inclusive and (time < self.time[0])) or (
            not inclusive and (time <= self.time[0])
        ):
            raise TimeSeriesRangeError(
                "There is no data in this TimeSeries before the specified "
                f"time of {time} since the begin time of this TimeSeries is "
                "{self.time[-1]}."
            )

        return self.get_ts_between_times(
            self.time[0], time, inclusive=(True, inclusive)
        )

    def get_ts_after_time(
        self, time: float, *, inclusive: bool = False
    ) -> TimeSeries:
        """
        Get a TimeSeries after the specified time.

        Parameters
        ----------
        time
            Time to look for in the TimeSeries' time attribute.
        inclusive
            Optional. True to include the given time in the comparison.

        Returns
        -------
        TimeSeries
            A new TimeSeries that fulfils the specified conditions.

        Raises
        ------
        TimeSeriesRangeError
            If there is no data after the specified index.

        See Also
        --------
        ktk.TimeSeries.get_ts_after_index
        ktk.TimeSeries.get_ts_after_event
        ktk.TimeSeries.get_ts_before_time
        ktk.TimeSeries.get_ts_between_times

        Example
        -------
        >>> ts = ktk.TimeSeries(time=np.arange(10)/10)
        >>> ts.time
        array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        >>> ts.get_ts_after_time(0.3).time
        array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        >>> ts.get_ts_after_time(0.3, inclusive=True).time
        array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        """
        check_param("time", time, float)
        check_param("inclusive", inclusive, bool)
        self._check_well_shaped()
        self._check_increasing_time()

        if (inclusive and (time > self.time[-1])) or (
            not inclusive and (time >= self.time[-1])
        ):
            raise TimeSeriesRangeError(
                "There is no data in this TimeSeries after the specified time "
                f"of {time} since the end time of this TimeSeries is "
                f"{self.time[-1]}."
            )

        return self.get_ts_between_times(
            time, self.time[-1], inclusive=(inclusive, True)
        )

    def get_ts_between_times(
        self,
        time1: float,
        time2: float,
        *,
        inclusive: bool | tuple[bool, bool] = False,
    ) -> TimeSeries:
        """
        Get a TimeSeries between two specified times.

        Parameters
        ----------
        time1, time2
            Times to look for in the TimeSeries' time attribute.
        inclusive
            Optional. Either a bool or a tuple of two bools. Used to
            specify which times are returned:

            - False or (False, False) (default): time1 < time < time2
            - True or (True, True): time1 <= time <= time2
            - (True, False): time1 <= time < time2
            - (False, True): time1 < time <= time2

        Returns
        -------
        TimeSeries
            A new TimeSeries that fulfils the specified conditions.

        Raises
        ------
        TimeSeriesRangeError
            If there is no data between the specified times.

        See Also
        --------
        ktk.TimeSeries.get_ts_between_indexes
        ktk.TimeSeries.get_ts_between_events
        ktk.TimeSeries.get_ts_before_time
        ktk.TimeSeries.get_ts_after_time

        Example
        -------
        >>> ts = ktk.TimeSeries(time=np.arange(10)/10)
        >>> ts.time
        array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        >>> ts.get_ts_between_times(0.2, 0.5).time
        array([0.3, 0.4])

        >>> ts.get_ts_between_times(0.2, 0.5, inclusive=True).time
        array([0.2, 0.3, 0.4, 0.5])

        >>> ts.get_ts_between_times(0.2, 0.5, inclusive=[True, False]).time
        array([0.2, 0.3, 0.4])

        """
        check_param("time1", time1, float)
        check_param("teim2", time2, float)
        if isinstance(inclusive, bool):
            inclusive = (inclusive, inclusive)
        try:
            inclusive = cast(tuple[bool, bool], tuple(inclusive))
            check_param(
                "inclusive",
                inclusive,
                tuple,
                length=2,
                contents_type=bool,
            )
        except TypeError:
            raise TypeError(
                "inclusive must be either a bool or a tuple of two bools."
            )

        if time2 < time1:
            raise ValueError(
                "The parameters time2 must be higher or equal to time1. "
                f"However, time2 is {time2} while time1 is {time1}."
            )

        index1 = self.get_index_after_time(time1, inclusive=inclusive[0])
        index2 = self.get_index_before_time(time2, inclusive=inclusive[1])
        return self.get_ts_between_indexes(index1, index2, inclusive=True)

    def get_ts_before_event(
        self, name: str, occurrence: int = 0, *, inclusive: bool = False
    ) -> TimeSeries:
        """
        Get a TimeSeries before the specified event.

        Parameters
        ----------
        name
            Name of the event to look for in the events list.
        occurrence
            Optional. i_th occurence of the event to look for in the events
            list, starting at 0.
        inclusive
            Optional. True to include the given time in the comparison.

        Returns
        -------
        TimeSeries
            A new TimeSeries that fulfils the specified conditions.

        Raises
        ------
        TimeSeriesRangeError
            If there is no data before the specified event.

        See Also
        --------
        ktk.TimeSeries.get_ts_before_index
        ktk.TimeSeries.get_ts_before_time
        ktk.TimeSeries.get_ts_after_event
        ktk.TimeSeries.get_ts_between_events

        Example
        -------
        >>> ts = ktk.TimeSeries(time=np.arange(10)/10)
        >>> ts = ts.add_event(0.2, "event")
        >>> ts = ts.add_event(0.35, "event")
        >>> ts.time
        array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        >>> ts.get_ts_before_event("event").time
        array([0. , 0.1])

        >>> ts.get_ts_before_event("event", inclusive=True).time
        array([0. , 0.1, 0.2])

        >>> ts.get_ts_before_event("event", 1).time
        array([0. , 0.1, 0.2, 0.3])

        >>> ts.get_ts_before_event("event", 1, inclusive=True).time
        array([0. , 0.1, 0.2, 0.3, 0.4])

        """
        check_param("name", name, str)
        check_param("occurrence", occurrence, int)
        check_param("inclusive", inclusive, bool)
        self._check_well_shaped()

        try:
            retval = self.get_ts_before_index(
                self.get_index_before_event(
                    name, occurrence, inclusive=inclusive
                ),
                inclusive=True,
            )
        except TimeSeriesRangeError:
            time = self.events[self._get_event_index(name, occurrence)].time
            raise TimeSeriesRangeError(
                f"There is no data before the occurrence {occurrence} of "
                f"event '{name}', which happens at {time} "
                f"{self.time_info['Unit']}."
            )
        else:
            return retval

    def get_ts_after_event(
        self, name: str, occurrence: int = 0, *, inclusive: bool = False
    ) -> TimeSeries:
        """
        Get a TimeSeries after the specified event.

        Parameters
        ----------
        name
            Name of the event to look for in the events list.
        occurrence
            Optional. i_th occurence of the event to look for in the events
            list, starting at 0.
        inclusive
            Optional. True to include the given event in the comparison.

        Returns
        -------
        TimeSeries
            A new TimeSeries that fulfils the specified conditions.

        Raises
        ------
        TimeSeriesRangeError
            If there is no data after the specified event.

        See Also
        --------
        ktk.TimeSeries.get_ts_after_index
        ktk.TimeSeries.get_ts_after_time
        ktk.TimeSeries.get_ts_before_event
        ktk.TimeSeries.get_ts_between_events

        Example
        -------
        >>> ts = ktk.TimeSeries(time=np.arange(10)/10)
        >>> ts = ts.add_event(0.2, "event")
        >>> ts = ts.add_event(0.35, "event")
        >>> ts.time
        array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        >>> ts.get_ts_after_event("event").time
        array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        >>> ts.get_ts_after_event("event", inclusive=True).time
        array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        >>> ts.get_ts_after_event("event", 1).time
        array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        >>> ts.get_ts_after_event("event", 1, inclusive=True).time
        array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        """
        check_param("name", name, str)
        check_param("occurrence", occurrence, int)
        check_param("inclusive", inclusive, bool)
        self._check_well_shaped()

        try:
            retval = self.get_ts_after_index(
                self.get_index_after_event(
                    name, occurrence, inclusive=inclusive
                ),
                inclusive=True,
            )
        except TimeSeriesRangeError:
            time = self.events[self._get_event_index(name, occurrence)].time
            raise TimeSeriesRangeError(
                f"There is no data after the occurrence {occurrence} of "
                f"event '{name}', which happens at {time} "
                f"{self.time_info['Unit']}."
            )
        else:
            return retval

    def get_ts_between_events(
        self,
        name1: str,
        name2: str,
        occurrence1: int = 0,
        occurrence2: int = 0,
        *,
        inclusive: bool | tuple[bool, bool] = False,
    ) -> TimeSeries:
        """
        Get a TimeSeries between two specified events.

        Parameters
        ----------
        name1, name2
            Name of the events to look for in the events list.
        occurrence1, occurrence2
            Optional. i_th occurence of the event to look for in the events
            list, starting at 0.
        inclusive
            Optional. Either a bool or a tuple of two bools. Used to
            specify which times are returned:

            - False or (False, False) (default): event1.time < time < event2.time
            - True or (True, True): event1.time <= time <= event2.time
            - (True, False): event1.time <= time < event2.time
            - (False, True): event1.time < time <= event2.time

        Returns
        -------
        TimeSeries
            A new TimeSeries that fulfils the specified conditions.

        Raises
        ------
        TimeSeriesRangeError
            If there is no data between the specified events.

        See Also
        --------
        ktk.TimeSeries.get_ts_between_indexes
        ktk.TimeSeries.get_ts_between_times
        ktk.TimeSeries.get_ts_before_event
        ktk.TimeSeries.get_ts_after_event

        Example
        -------
        >>> ts = ktk.TimeSeries(time=np.arange(10)/10)
        >>> ts = ts.add_event(0.2, "event")
        >>> ts = ts.add_event(0.55, "event")
        >>> ts.time
        array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        >>> ts.get_ts_between_events("event", "event", 0, 1).time
        array([0.3, 0.4, 0.5])

        >>> ts.get_ts_between_events("event", "event", 0, 1, \
                                     inclusive=True).time
        array([0.2, 0.3, 0.4, 0.5, 0.6])

        """
        check_param("name1", name1, str)
        check_param("name2", name2, str)
        check_param("occurrence1", occurrence2, int)
        check_param("occurrence1", occurrence2, int)
        if isinstance(inclusive, bool):
            inclusive = (inclusive, inclusive)
        try:
            inclusive = cast(tuple[bool, bool], tuple(inclusive))
            check_param(
                "inclusive",
                inclusive,
                tuple,
                length=2,
                contents_type=bool,
            )
        except TypeError:
            raise TypeError(
                "inclusive must be either a bool or a tuple of two bools."
            )

        self._check_well_shaped()

        time1 = self.events[self._get_event_index(name1, occurrence1)].time
        time2 = self.events[self._get_event_index(name2, occurrence2)].time

        if time2 < time1:
            raise ValueError(
                f"The end event (occurrence {occurrence2} of "
                f"'{name2}') happens at {time2} {self.time_info['Unit']}, "
                f"which is before the begin event (occurrence {occurrence1} "
                f"of '{name1}') that happens at {time1} "
                f"{self.time_info['Unit']}."
            )

        index1 = self.get_index_after_event(
            name1, occurrence1, inclusive=inclusive[0]
        )
        index2 = self.get_index_before_event(
            name2, occurrence2, inclusive=inclusive[1]
        )
        return self.get_ts_between_indexes(index1, index2, inclusive=True)

    # %% Time management

    def shift(self, time: float, *, in_place: bool = False) -> TimeSeries:
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
        >>> ts = ktk.TimeSeries(time=np.arange(10)/10)
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
        >>> ts = ktk.TimeSeries(time=np.arange(100)/10)  # 100 samples at 10 Hz
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
    ) -> TimeSeries:
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
        >>> ts = ktk.TimeSeries(time=np.arange(10.))
        >>> ts = ts.add_data("data", ts.time ** 2)
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
            for i in nan_indexes:
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

    # %% Subsetting and merging

    def get_subset(self, data_keys: str | list[str]) -> TimeSeries:
        """
        Return a subset of the TimeSeries.

        This method returns a TimeSeries that contains only selected data
        keys. The corresponding data_info keys are copied in the new
        TimeSeries. All events are also copied in the new TimeSeries.

        Parameters
        ----------
        data_keys
            The data keys to extract from the TimeSeries.

        Returns
        -------
        TimeSeries
            The TimeSeries, minus the unspecified data keys.

        Raises
        ------
        KeyError
            If one or more data keys could not be found in the TimeSeries
            data.

        See Also
        --------
        ktk.TimeSeries.merge

        Example
        -------
            >>> ts = ktk.TimeSeries(time = np.arange(10))
            >>> ts = ts.add_data("signal1", ts.time)
            >>> ts = ts.add_data("signal2", ts.time**2)
            >>> ts = ts.add_data("signal3", ts.time**3)
            >>> ts.data.keys()
            dict_keys(['signal1', 'signal2', 'signal3'])

            >>> ts2 = ts.get_subset(["signal1", "signal3"])
            >>> ts2.data.keys()
            dict_keys(['signal1', 'signal3'])

        """
        try:
            check_param("data_keys", data_keys, str)
        except TypeError:
            try:
                check_param("data_keys", data_keys, list, contents_type=str)
            except TypeError:
                raise TypeError(
                    "data_keys must be a string or a list of strings."
                )
        self._check_well_shaped()

        if isinstance(data_keys, str):
            data_keys = [data_keys]

        ts = TimeSeries()
        ts.time = self.time.copy()
        ts.time_info = deepcopy(self.time_info)
        ts.events = deepcopy(self.events)

        for key in data_keys:
            try:
                ts.data[key] = self.data[key].copy()
            except KeyError:
                raise KeyError(
                    f"The key '{key}' could not be found among the "
                    f"{len(self.data)} data entries of the TimeSeries"
                )

            try:
                ts.data_info[key] = deepcopy(self.data_info[key])
            except KeyError:
                pass

        return ts

    def merge(
        self,
        ts: TimeSeries,
        data_keys: str | list[str] = [],
        *,
        resample: bool = False,
        overwrite: bool = False,
        on_conflict: str = "warning",
        in_place: bool = False,
    ) -> TimeSeries:
        """
        Merge the TimeSeries with another TimeSeries.

        Parameters
        ----------
        ts
            The TimeSeries to merge into the current TimeSeries.
        data_keys
            Optional. The data keys to merge from ts. If left empty, all the
            data keys are merged.
        resample
            Optional. Set to True to resample the source TimeSeries to the
            target one using a linear interpolation. If the time attributes are
            not equivalent and resample is False, an exception is raised. To
            resample using other methods than linear interpolation, please
            resample the source TimeSeries manually before, using
            TimeSeries.resample. Default is False.
        overwrite
            Optional. Select what to do if a key from the source TimeSeries
            already exists in the destination TimeSeries. True to overwrite
            the already existing value, False to ignore the new value.
            Default is False.
        on_conflict
            Optional. Select what the warning level when a key from the source
            TimeSeries already exists in the destination TimeSeries. May take
            the following values:
            "mute": No warning;
            "warning": Warns that duplicate keys were found and how the
            conflict has been resolved following the `overwrite` parameter.
            "error": Raises a TimeSeriesMergeConflictError.
            Default is "warning".
        in_place
            Optional. True to modify and return the original TimeSeries. False
            to return a modified copy of the TimeSeries while leaving the
            original TimeSeries intact. Default is False.

        Returns
        -------
        TimeSeries
            The merged TimeSeries.

        Raises
        ------
        TimeSeriesMergeConflictError
            If a key from the source TimeSeries already exists in the
            destination TimeSeries and on_conflict is set to "error".

        See Also
        --------
        ktk.TimeSeries.get_subset
        ktk.TimeSeries.resample

        Notes
        -----
        - All events are also merged from both TimeSeries.

        """
        try:
            check_param("data_keys", data_keys, str)
        except TypeError:
            try:
                data_keys = list(data_keys)
                check_param("data_keys", data_keys, list, contents_type=str)
            except TypeError:
                raise TypeError(
                    "data_keys must be a string or a list of strings."
                )
        check_param("resample", resample, bool)
        check_param("overwrite", overwrite, bool)
        check_param("on_conflict", on_conflict, str)
        if on_conflict not in ["mute", "warning", "error"]:
            raise ValueError(
                "Parameter on_conflict must be either 'mute', 'warning' or "
                "'error'."
            )
        check_param("in_place", in_place, bool)
        self._check_well_shaped()
        ts._check_well_shaped()
        # --

        ts_out = self if in_place else self.copy()
        ts = ts.copy()
        if len(data_keys) == 0:
            data_keys = list(ts.data.keys())
        elif isinstance(data_keys, str):
            data_keys = [data_keys]

        # Check if resampling is needed
        if len(ts_out.time) == 0:
            ts_out.time = deepcopy(ts.time)

        if (ts_out.time.shape == ts.time.shape) and np.all(
            ts_out.time == ts.time
        ):
            must_resample = False
        else:
            must_resample = True

        if must_resample is True and resample is False:
            raise ValueError(
                "Time attributes do not match, resampling is required."
            )

        if must_resample is True:
            ts.resample(ts_out.time, in_place=True)

        for key in data_keys:
            # Check if this key is a duplicate, then continue to next key if
            # required.
            if key in ts_out.data:
                if overwrite is True:
                    if on_conflict.lower() == "warning":
                        warnings.warn(
                            f"The key '{key}' exists in both TimeSeries. "
                            f"According to the overwrite={overwrite} "
                            "parameter, its prior value has been overwritten "
                            "by the new value. Use on_conflict='mute' to mute "
                            "this warning."
                        )
                    elif on_conflict.lower() == "error":
                        raise TimeSeriesMergeConflictError(
                            f"The key '{key}' exists in both TimeSeries. "
                        )
                    ts_out.data[key] = ts.data[key]

                else:  # overwrite is False
                    if on_conflict.lower() == "warning":
                        warnings.warn(
                            f"The key '{key}' exists in both TimeSeries. "
                            f"According to the overwrite={overwrite} "
                            "parameter, the new value has been ignored. Use "
                            "on_conflict='mute' to mute this warning."
                        )
                    elif on_conflict.lower() == "error":
                        raise TimeSeriesMergeConflictError(
                            f"The key {key} exist in both TimeSeries. "
                        )

            else:
                # Add this data
                ts_out.data[key] = ts.data[key]

                if key in ts.data_info:
                    for info_key in ts.data_info[key].keys():
                        ts_out.add_data_info(
                            key,
                            info_key,
                            ts.data_info[key][info_key],
                            in_place=True,
                        )

        # Merge events
        for event in ts.events:
            ts_out.add_event(
                event.time, event.name, in_place=True, unique=True
            )
        return ts_out

    # %% Missing sample management

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
    ) -> TimeSeries:
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
                    elif (
                        current_index - still_visible_index
                        > max_missing_samples
                    ):
                        to_keep[
                            still_visible_index + 1 : current_index + 1
                        ] = 0

                ts.data[data][to_keep == 0] = np.nan

            ts_out.data[data] = ts.data[data]

        return ts_out

    # %% Graphical user interfaces

    def ui_edit_events(
        self,
        name: str | list[str] = [],
        data_keys: str | list[str] = [],
    ) -> TimeSeries:  # pragma: no cover
        """
        Edit events interactively.

        Parameters
        ----------
        name
            Optional. The name of the event(s) to add. May be a string
            or a list of strings. These events appear on their own buttons
            "add `name`". Event names can also be defined interactively.
        data_keys
            Optional. A signal name of list of signal name to be plotted,
            similar to the data_keys argument of ktk.TimeSeries.plot.

        Returns
        -------
        TimeSeries
            The TimeSeries with the modified events. If the operation was
            cancelled by the user, this is the original TimeSeries.

        Warning
        -------
        This function, which has been introduced in 0.6, is still experimental
        and may change signature or behaviour in the future.

        See Also
        --------
        ktk.TimeSeries.add_event
        ktk.TimeSeries.rename_event
        ktk.TimeSeries.remove_event
        ktk.TimeSeries.trim_events

        Note
        ----
        Matplotlib must be in interactive mode for this function to work.

        """
        check_interactive_backend()

        try:
            check_param("name", name, str)
        except TypeError:
            try:
                check_param("name", name, list, contents_type=str)
            except TypeError:
                raise TypeError("name must be a string or a list of strings.")
        try:
            check_param("data_keys", data_keys, str)
        except TypeError:
            try:
                check_param("data_keys", data_keys, list, contents_type=str)
            except TypeError:
                raise TypeError(
                    "data_keys must be a string or a list of strings."
                )
        self._check_well_shaped()
        self._check_not_empty_time()
        self._check_not_empty_data()

        def add_this_event(ts: TimeSeries, name: str) -> TimeSeries:
            kineticstoolkit.gui.message(
                "Place the event on the figure.", **WINDOW_PLACEMENT
            )
            this_time = plt.ginput(1)[0][0]
            ts = ts.add_event(this_time, name)
            kineticstoolkit.gui.message("")
            return ts

        def get_event_index(ts: TimeSeries) -> int:
            kineticstoolkit.gui.message(
                "Select an event on the figure.", **WINDOW_PLACEMENT
            )
            this_time = plt.ginput(1)[0][0]
            event_times = np.array([event.time for event in ts.events])
            kineticstoolkit.gui.message("")
            return int(np.argmin(np.abs(event_times - this_time)))

        # Set Matplotlib interactive mode
        isinteractive = plt.isinteractive()
        plt.ion()

        ts = self.copy()

        if isinstance(name, str):
            event_names = [name]
        else:
            event_names = deepcopy(name)

        fig = plt.figure()
        ts.plot(data_keys, _raise_on_no_data=True)

        while True:
            # Populate the choices to the user
            choices = [f"Add '{s}'" for s in event_names]

            choice_index = {}
            choice_index["add"] = len(choices)
            if len(event_names) == 0:
                choices.append("Add event")
            else:
                choices.append("Add event with another name")

            if len(ts.events) > 0:
                choice_index["remove"] = len(choices)
                choices.append("Remove event")

            if len(ts.events) > 0:
                choice_index["remove_all"] = len(choices)
                choices.append("Remove all events")

                choice_index["move"] = len(choices)
                choices.append("Move event")

            choice_index["close"] = len(choices)
            choices.append("Save and close")

            choice_index["cancel"] = len(choices)
            choices.append("Cancel")

            # Show the button dialog
            choice = kineticstoolkit.gui.button_dialog(
                "Move and zoom on the figure,\n"
                "then select an option below.",
                choices,
                **WINDOW_PLACEMENT,
            )

            # Execute
            if choice < choice_index["add"]:
                ts = add_this_event(ts, event_names[choice])

            elif choice == choice_index["add"]:
                event_names.append(
                    li.input_dialog(
                        "Please enter the event name:", **WINDOW_PLACEMENT
                    )
                )
                # Add this event name to the list of recently added events
                if len(event_names) > 5:
                    event_names = event_names[-5:]

                # Add the event
                ts = add_this_event(ts, event_names[-1])

            elif ("remove" in choice_index) and (
                choice == choice_index["remove"]
            ):
                event_index = get_event_index(ts)
                try:
                    ts.events.pop(event_index)
                except IndexError:
                    li.button_dialog(
                        "No event was removed.",
                        choices=["OK"],
                        icon="error",
                        **WINDOW_PLACEMENT,
                    )

            elif ("remove_all" in choice_index) and (
                choice == choice_index["remove_all"]
            ):
                if (
                    li.button_dialog(
                        "Do you really want to remove all events from this "
                        "TimeSeries?",
                        ["Yes, remove all events", "No"],
                        icon="alert",
                        **WINDOW_PLACEMENT,
                    )
                    == 0
                ):
                    ts.events = []

            elif ("move" in choice_index) and (choice == choice_index["move"]):
                event_index = get_event_index(ts)
                event_name = ts.events[event_index].name
                try:
                    ts.events.pop(event_index)
                    ts = add_this_event(ts, event_name)
                except IndexError:
                    li.button_dialog(
                        "Could not move this event.",
                        choices=["OK"],
                        icon="error",
                        **WINDOW_PLACEMENT,
                    )

            elif ("close" in choice_index) and (
                choice == choice_index["close"]
            ):
                plt.close(fig)
                if not isinteractive:
                    plt.ioff()
                return ts

            elif (choice == -1) or (
                ("cancel" in choice_index)
                and (choice == choice_index["cancel"])
            ):
                plt.close(fig)
                if not isinteractive:
                    plt.ioff()
                return self.copy()

            # Refresh
            ts.remove_duplicate_events(in_place=True)
            axes = plt.axis()
            plt.cla()
            ts.plot(data_keys, _raise_on_no_data=True)
            plt.axis(axes)

    def ui_sync(
        self,
        data_keys: str | list[str] = [],
        ts2: TimeSeries | None = None,
        data_keys2: str | list[str] = [],
    ) -> TimeSeries:  # pragma: no cover
        """
        Synchronize one or two TimeSeries by shifting their time.

        If this method is called on only one TimeSeries, an interactive
        interface asks the user to click on the time to set to zero.

        If another TimeSeries is given, an interactive interface allows
        synchronizing both TimeSeries together.

        Parameters
        ----------
        data_keys
            Optional. The data keys to plot. If empty, all data is plotted.
        ts2
            Optional. A second TimeSeries to be synced to the first one. This
            TimeSeries is modified in place.
        data_keys2
            Optional. The data keys from the second TimeSeries to plot. If
            empty, all data is plotted.

        Returns
        -------
        TimeSeries
            The TimeSeries after synchronization.

        Warning
        -------
        This function, which has been introduced in 0.1, is still experimental
        and may change signature or behaviour in the future.

        See Also
        --------
        ktk.TimeSeries.shift

        Notes
        -----
        Matplotlib must be in interactive mode for this method to work.

        """
        check_interactive_backend()

        try:
            check_param("data_keys", data_keys, str)
        except TypeError:
            try:
                check_param("data_keys", data_keys, list, contents_type=str)
            except TypeError:
                raise TypeError(
                    "data_keys must be a string or a list of strings."
                )
        try:
            check_param("data_keys2", data_keys2, str)
        except TypeError:
            try:
                check_param("data_keys2", data_keys2, list, contents_type=str)
            except TypeError:
                raise TypeError(
                    "data_keys2 must be a string or a list of strings."
                )

        self._check_well_shaped()
        self._check_not_empty_time()
        self._check_not_empty_data()

        if ts2 is not None:
            ts2._check_well_shaped()
            ts2._check_not_empty_time()
            ts2._check_not_empty_data()

        ts1 = self.copy()

        fig = plt.figure("ktk.TimeSeries.ui_sync")

        if ts2 is None:
            # Synchronize ts1 only
            ts1.plot(data_keys)
            choice = kineticstoolkit.gui.button_dialog(
                "Please zoom on the time zero and press Next.",
                ["Cancel", "Next"],
                **WINDOW_PLACEMENT,
            )
            if choice != 1:
                plt.close(fig)
                return ts1

            kineticstoolkit.gui.message(
                "Click on the sync event.", **WINDOW_PLACEMENT
            )
            click = plt.ginput(1)
            kineticstoolkit.gui.message("")
            plt.close(fig)
            ts1 = ts1.shift(-click[0][0])

        else:  # Sync two TimeSeries together
            finished = False
            # list of axes:
            axes = []  # type: list[Any]
            while finished is False:
                if len(axes) == 0:
                    axes.append(fig.add_subplot(2, 1, 1))
                    axes.append(fig.add_subplot(2, 1, 2, sharex=axes[0]))

                plt.sca(axes[0])
                axes[0].cla()
                ts1.plot(data_keys)
                plt.title("First TimeSeries (ts1)")
                plt.grid(True)
                plt.tight_layout()

                plt.sca(axes[1])
                axes[1].cla()
                ts2.plot(data_keys2)
                plt.title("Second TimeSeries (ts2)")
                plt.grid(True)
                plt.tight_layout()

                choice = kineticstoolkit.gui.button_dialog(
                    "Please select an option.",
                    choices=[
                        "Zero ts1 only, using ts1",
                        "Zero ts2 only, using ts2",
                        "Zero both TimeSeries, using ts1",
                        "Zero both TimeSeries, using ts2",
                        "Sync both TimeSeries on a common event",
                        "Finished",
                    ],
                    **WINDOW_PLACEMENT,
                )

                if choice == 0:  # Zero ts1 only
                    kineticstoolkit.gui.message(
                        "Click on the time zero in ts1.", **WINDOW_PLACEMENT
                    )
                    click_1 = plt.ginput(1)
                    kineticstoolkit.gui.message("")

                    ts1 = ts1.shift(-click_1[0][0])

                elif choice == 1:  # Zero ts2 only
                    kineticstoolkit.gui.message(
                        "Click on the time zero in ts2.", **WINDOW_PLACEMENT
                    )
                    click_1 = plt.ginput(1)
                    kineticstoolkit.gui.message("")

                    ts2 = ts2.shift(-click_1[0][0])

                elif choice == 2:  # Zero ts1 and ts2 using ts1
                    kineticstoolkit.gui.message(
                        "Click on the time zero in ts1.", **WINDOW_PLACEMENT
                    )
                    click_1 = plt.ginput(1)
                    kineticstoolkit.gui.message("")

                    ts1 = ts1.shift(-click_1[0][0])
                    ts2 = ts2.shift(-click_1[0][0])

                elif choice == 3:  # Zero ts1 and ts2 using ts2
                    kineticstoolkit.gui.message(
                        "Click on the time zero in ts2.", **WINDOW_PLACEMENT
                    )
                    click_2 = plt.ginput(1)
                    kineticstoolkit.gui.message("")

                    ts1 = ts1.shift(-click_2[0][0])
                    ts2 = ts2.shift(-click_2[0][0])

                elif choice == 4:  # Sync on a common event
                    kineticstoolkit.gui.message(
                        "Click on the sync event in ts1.", **WINDOW_PLACEMENT
                    )
                    click_1 = plt.ginput(1)
                    kineticstoolkit.gui.message(
                        "Now click on the same event in ts2.",
                        **WINDOW_PLACEMENT,
                    )
                    click_2 = plt.ginput(1)
                    kineticstoolkit.gui.message("")

                    ts1 = ts1.shift(-click_1[0][0])
                    ts2 = ts2.shift(-click_2[0][0])

                elif choice == 5 or choice < -1:  # OK or closed figure, quit.
                    plt.close(fig)
                    finished = True

        return ts1

    def plot(
        self,
        data_keys: str | list[str] = [],
        *args,
        event_names: bool = True,
        legend: bool = True,
        **kwargs,
    ) -> None:
        """
        Plot the TimeSeries in the current matplotlib figure.

        Parameters
        ----------
        data_keys
            The data keys to plot. If left empty, all data is plotted.
        event_names
            Optional. True to plot the event names on top of the event lines.
        legend
            Optional. True to plot a legend, False otherwise.

        Note
        ----
        Additional positional and keyboard arguments are passed to
        matplotlib's ``pyplot.plot`` function::

            ts.plot(["Forces"], "--")

        plots the forces using a dashed line style.

        Example
        -------
        For a TimeSeries ``ts`` with data keys being "Forces", "Moments" and
        "Angle"::

            ts.plot()

        plots all data (Forces, Moments and Angle), whereas::

            ts.plot(["Forces", "Moments"])

        plots only the forces and moments, without plotting the angle.

        """
        try:
            check_param("data_keys", data_keys, str)
        except TypeError:
            try:
                check_param("data_keys", data_keys, list, contents_type=str)
            except TypeError:
                raise TypeError(
                    "data_keys must be a string or a list of strings."
                )
        check_param("event_names", event_names, bool)
        check_param("legend", legend, bool)
        self._check_well_shaped()

        # Private argument _raise_on_no_data: Raise an EmptyDataSeriesError
        # instead of warning when no data is available to plot.
        if "_raise_on_no_data" in kwargs:
            raise_on_no_data = kwargs.pop("_raise_on_no_data")
        else:
            raise_on_no_data = False

        if data_keys is None or len(data_keys) == 0:
            # Plot all
            ts = self.copy()
        else:
            ts = self.get_subset(data_keys)

        if raise_on_no_data:
            self._check_not_empty_time()
            self._check_not_empty_data()

        df = ts.to_dataframe()
        labels = df.columns.to_list()

        axes = plt.gca()
        # Don't know why I need to disable mypy on these lines.
        axes.set_prop_cycle(
            mpl.cycler(linewidth=[1, 2, 3, 4])  # type: ignore
            * mpl.cycler(linestyle=["-", "--", "-.", ":"])  # type: ignore
            * plt.rcParams["axes.prop_cycle"]
        )

        # Plot the curves
        for i_label, label in enumerate(labels):
            axes.plot(
                df.index.to_numpy(),
                df[label].to_numpy(),
                *args,
                label=label,
                **kwargs,
            )

        # Add labels
        plt.xlabel("Time (" + ts.time_info["Unit"] + ")")

        # Make unique list of units
        unit_set = set()
        for data in ts.data_info:
            for info in ts.data_info[data]:
                if info == "Unit":
                    unit_set.add(ts.data_info[data][info])
        # Plot this list
        unit_str = ""
        for unit in unit_set:
            if len(unit_str) > 0:
                unit_str += ", "
            unit_str += unit

        plt.ylabel(unit_str)

        # Plot the events
        n_events = len(ts.events)
        event_times = []
        for event in ts.events:
            event_times.append(event.time)

        if len(ts.events) > 0:
            a = plt.axis()
            min_y = a[2]
            max_y = a[3]
            event_line_x = np.zeros(3 * n_events)
            event_line_y = np.zeros(3 * n_events)

            for i_event in range(0, n_events):
                event_line_x[3 * i_event] = event_times[i_event]
                event_line_x[3 * i_event + 1] = event_times[i_event]
                event_line_x[3 * i_event + 2] = np.nan

                event_line_y[3 * i_event] = min_y
                event_line_y[3 * i_event + 1] = max_y
                event_line_y[3 * i_event + 2] = np.nan

            plt.plot(event_line_x, event_line_y, ":k")

            if event_names:
                occurrences = {}  # type:dict[str, int]

                for event in ts.events:
                    if event.name == "_":
                        name = "_"
                    elif event.name in occurrences:
                        occurrences[event.name] += 1
                        name = f"{event.name} {occurrences[event.name]}"
                    else:
                        occurrences[event.name] = 0
                        name = f"{event.name} 0"

                    plt.text(
                        event.time,
                        max_y,
                        name,
                        rotation="vertical",
                        horizontalalignment="center",
                        fontsize="small",
                    )

        if legend and len(ts.data) > 0:
            if len(labels) < 20:
                legend_location = "best"
            else:
                legend_location = "upper right"

            axes.legend(
                loc=legend_location, ncol=1 + int(len(labels) / 40)
            )  # Max 40 items per line

    # %% Input/Output

    def _to_dataframe_and_info(
        self,
    ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        """
        Implement TimeSeries.to_dataframe with additional data_info.

        The second element of the output tuple is a list where each element
        corresponds to a column of the DataFrame, and each element is a copy
        of the inner data_info dictionary for this data. For instance,
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

                # Get the column names index from the shape of the original data
                # The strategy here is to build matrices of indexes, that have
                # the same shape as the original data, then reshape these matrices
                # the same way we reshaped the original data. Then we know where
                # the original indexes are in the new reshaped data.
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
                        this_column_name += "["

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

            # Add the data_info that correspond to this key
            for i in df_data.columns:
                try:
                    data_info = self.data_info[the_key]
                    info_out.append(deepcopy(data_info))
                except KeyError:
                    info_out.append({})

        df_out.index = self.time

        return (df_out, info_out)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Create a DataFrame by reshaping all data to one bidimensional table.

        Undimensional data is converted to a single column, and two-dimensional
        (or more) data are converted to multiple columns with the additional
        dimensions in brackets. The TimeSeries's events and metadata such as
        `time_info` and `data_info` are not included in the resulting
        DataFrame.

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
              test[0]  test[1]  test[2]
         0.0      0.0      2.0      3.0
         0.1      0.0      2.0      3.0
         0.2      0.0      2.0      3.0
         0.3      0.0      2.0      3.0

        """
        self._check_well_shaped()
        return self._to_dataframe_and_info()[0]

    @staticmethod
    def from_dataframe(
        dataframe: pd.DataFrame,
        /,
        *,
        time_info: dict[str, Any] = {"Unit": "s"},
        data_info: dict[str, dict[str, Any]] = {},
        events: list[TimeSeriesEvent] = [],
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
        time_info
            Optional. Will be copied to the TimeSeries' time_info attribute.
        data_info
            Optional. Will be copied to the TimeSeries' data_info attribute.
        events
            Optional. Will be copied to the TimeSeries' events attribute.

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
        >>> df = pd.DataFrame([[1., 2.], [3., 4.], [5., 6.]])
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

        >>> df = pd.DataFrame([[1., 2.], [3., 4.], [5., 6.]])
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
        >>> df.index = [0., 0.1, 0.2, 0.3, 0.4]  # Time in seconds
        >>> df["rot[0,0]"] = np.cos([0., 0.1, 0.2, 0.3, 0.4])
        >>> df["rot[0,1]"] = -np.sin([0., 0.1, 0.2, 0.3, 0.4])
        >>> df["rot[1,0]"] = np.sin([0., 0.1, 0.2, 0.3, 0.4])
        >>> df["rot[1,1]"] = np.cos([0., 0.1, 0.2, 0.3, 0.4])
        >>> df["trans[0]"] = [0., 0.1, 0.2, 0.3, 0.4]
        >>> df["trans[1]"] = [5., 6., 7., 8., 9.]
        >>> df
             rot[0,0]  rot[0,1]  rot[1,0]  rot[1,1]  trans[0]  trans[1]
        0.0  1.000000 -0.000000  0.000000  1.000000       0.0       5.0
        0.1  0.995004 -0.099833  0.099833  0.995004       0.1       6.0
        0.2  0.980067 -0.198669  0.198669  0.980067       0.2       7.0
        0.3  0.955336 -0.295520  0.295520  0.955336       0.3       8.0
        0.4  0.921061 -0.389418  0.389418  0.921061       0.4       9.0

        Convert to a TimeSeries:

        >>> ts = ktk.TimeSeries(df)
        >>> ts.data
        {'rot': array([[[ 1.        , -0.        ],
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
                [ 0.38941834,  0.92106099]]]), 'trans': array([[0. , 5. ],
               [0.1, 6. ],
               [0.2, 7. ],
               [0.3, 8. ],
               [0.4, 9. ]])}

        """
        check_param("dataframe", dataframe, pd.DataFrame)
        check_param("time_info", time_info, dict, key_type=str)
        check_param(
            "data_info", data_info, dict, key_type=str, contents_type=dict
        )
        check_param("events", events, list)

        ts = TimeSeries(
            time=dataframe.index.to_numpy(),
            time_info=time_info,
            data_info=data_info,
            events=events,
        )

        # Remove spaces in indexes between brackets
        columns = dataframe.columns
        new_columns = []
        for i_column, column in enumerate(columns):
            splitted = column.split("[")
            if len(splitted) > 1:  # There are brackets
                new_columns.append(
                    splitted[0] + "[" + splitted[1].replace(" ", "")
                )
            else:
                new_columns.append(column)
        dataframe.columns = columns

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
        for key in dimensions:
            if len(dimensions[key]) == 0:
                ts.data[key] = dataframe[key].to_numpy()
            else:
                highest_dims = np.max(np.array(dimensions[key]), axis=0)

                columns = [
                    key + str(dim).replace(" ", "")
                    for dim in sorted(dimensions[key])
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
        time: ArrayLike = [],
        time_info: dict[str, Any] = {"Unit": "s"},
        data_info: dict[str, dict[str, Any]] = {},
        events: list[TimeSeriesEvent] = [],
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
        time_info
            Optional. Will be copied to the TimeSeries' time_info attribute.
        data_info
            Optional. Will be copied to the TimeSeries' data_info attribute.
        events
            Optional. Will be copied to the TimeSeries' events attribute.

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
            time_info: {'Unit': 's'}
            data_info: {}
               events: []

        **Specifiying time**

        >>> ktk.TimeSeries([0.1, 0.2, 0.3, 0.4, 0.5], time=[0.1, 0.2, 0.3, 0.4, 0.5])
        TimeSeries with attributes:
                 time: array([0.1, 0.2, 0.3, 0.4, 0.5])
                 data: {'data': array([0.1, 0.2, 0.3, 0.4, 0.5])}
            time_info: {'Unit': 's'}
            data_info: {}
               events: []

        """
        check_param("data_key", data_key, str)
        check_param("time_info", time_info, dict, key_type=str)
        check_param(
            "data_info", data_info, dict, key_type=str, contents_type=dict
        )
        check_param("events", events, list)

        time = np.array(time)
        ts = TimeSeries(
            data={data_key: array},
            time_info=time_info,
            data_info=data_info,
            events=events,
        )

        if time.shape[0] == 0:
            ts.time = np.arange(ts.data[data_key].shape[0]) * 1.0  # floats
        else:
            ts.time = time

        return ts

    # %% Deprecrated methods
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
    ) -> TimeSeries:
        """
        Deprecated. Sorts the TimeSeries' events from the earliest to the
        latest.

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
        self._check_well_typed()

        ts = self if in_place else self.copy()
        if unique:
            ts.remove_duplicate_events(in_place=True)
        ts.events = sorted(ts.events)
        return ts


# %% Main

if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
