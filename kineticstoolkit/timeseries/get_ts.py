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
"""Provide get_ts_ methods for TimeSeries."""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2025 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


from typing import TYPE_CHECKING, cast

from kineticstoolkit.exceptions import (
    TimeSeriesRangeError,
)
from kineticstoolkit.typing_ import check_param

if TYPE_CHECKING:
    from kineticstoolkit import TimeSeries


def get_ts_before_index(
    self, index: int, *, inclusive: bool = False
) -> "TimeSeries":
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
    >>> ts = ktk.TimeSeries(time=np.arange(10) / 10)
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

    return self.get_ts_between_indexes(0, index, inclusive=(True, inclusive))


def get_ts_after_index(
    self, index: int, *, inclusive: bool = False
) -> "TimeSeries":
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
    >>> ts = ktk.TimeSeries(time=np.arange(10) / 10)
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
) -> "TimeSeries":
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
    >>> ts = ktk.TimeSeries(time=np.arange(10) / 10)
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
) -> "TimeSeries":
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
    >>> ts = ktk.TimeSeries(time=np.arange(10) / 10)
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
) -> "TimeSeries":
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
    >>> ts = ktk.TimeSeries(time=np.arange(10) / 10)
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
) -> "TimeSeries":
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
    >>> ts = ktk.TimeSeries(time=np.arange(10) / 10)
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
) -> "TimeSeries":
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
    >>> ts = ktk.TimeSeries(time=np.arange(10) / 10)
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
            self.get_index_before_event(name, occurrence, inclusive=inclusive),
            inclusive=True,
        )
    except TimeSeriesRangeError:
        time = self.events[self._get_event_index(name, occurrence)].time
        raise TimeSeriesRangeError(
            f"There is no data before the occurrence {occurrence} of "
            f"event '{name}', which happens at {time} "
            f"{self._get_time_unit()}."
        )
    else:
        return retval


def get_ts_after_event(
    self, name: str, occurrence: int = 0, *, inclusive: bool = False
) -> "TimeSeries":
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
    >>> ts = ktk.TimeSeries(time=np.arange(10) / 10)
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
            self.get_index_after_event(name, occurrence, inclusive=inclusive),
            inclusive=True,
        )
    except TimeSeriesRangeError:
        time = self.events[self._get_event_index(name, occurrence)].time
        raise TimeSeriesRangeError(
            f"There is no data after the occurrence {occurrence} of "
            f"event '{name}', which happens at {time} "
            f"{self._get_time_unit()}."
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
) -> "TimeSeries":
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
            f"'{name2}') happens at {time2} {self._get_time_unit()}, "
            f"which is before the begin event (occurrence {occurrence1} "
            f"of '{name1}') that happens at {time1} "
            f"{self._get_time_unit()}."
        )

    index1 = self.get_index_after_event(
        name1, occurrence1, inclusive=inclusive[0]
    )
    index2 = self.get_index_before_event(
        name2, occurrence2, inclusive=inclusive[1]
    )
    return self.get_ts_between_indexes(index1, index2, inclusive=True)
