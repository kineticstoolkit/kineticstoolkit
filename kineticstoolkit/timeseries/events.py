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
"""Provide event management methods for TimeSeries."""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2025 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np

from kineticstoolkit.exceptions import TimeSeriesEventNotFoundError
from kineticstoolkit.typing_ import check_param

from .classes import TimeSeriesEvent

if TYPE_CHECKING:
    from kineticstoolkit import TimeSeries


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
    self._check_valid_time()

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
    self._check_valid_time()

    occurrence = int(occurrence)

    if occurrence < 0:
        raise TimeSeriesEventNotFoundError(
            "The parameter `occurrence` must be positive a integer. "
            f"However, a value of {occurrence} was received."
        )

    # Get the event occurrence
    try:
        return self._get_event_indexes(name)[occurrence]
    except IndexError as e:
        raise TimeSeriesEventNotFoundError(
            f"The occurrence {occurrence} of event '{name}' could not "
            "be found in the TimeSeries. A total of "
            f"{len(self._get_event_indexes(name))} occurrence(s) of "
            "this event name were found."
        ) from e


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
    >>> ts = ts.add_event(1e-12, "event1")
    >>> ts = ts.add_event(0.0, "event1")

    # One occurrence of event2, but also at 0.0 second
    >>> ts = ts.add_event(0.0, "event2")

    # Two occurrences of event3
    >>> ts = ts.add_event(2.0, "event3")
    >>> ts = ts.add_event(2.0, "event3")

    """
    self._check_valid_time()

    # Sort all events in a dict with key being tuple(time, name)
    # and the value being the list of indexes in which this event appears.
    sorted_events = {}  # type: dict[tuple[float, str], list[int]]
    for i_event, event in enumerate(self.events):
        tup_event = event._to_tuple()

        # Check if this event already exist in the list.
        # If it does, add it to the list.
        found = False
        for key, occurrence_list in sorted_events.items():
            if np.isclose(key[0], event.time) and (key[1] == event.name):
                occurrence_list.append(i_event)
                found = True
                break
        if not found:
            # Otherwise, create it in the list
            sorted_events[tup_event] = [i_event]

    # Convert this dict to the desired list of lists
    out = []
    for _key, occurrence_list in sorted_events.items():
        if len(occurrence_list) > 1:
            out.extend(occurrence_list[1:])

    return sorted(out)


def add_event(
    self,
    time: float,
    name: str = "event",
    *,
    in_place: bool = False,
    unique: bool = False,
) -> "TimeSeries":
    """
    Add an event to the TimeSeries.

    Parameters
    ----------
    time
        The time of the event, in the same unit as `info["Time"]["Unit"]`.
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
    self._check_valid_time()

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
) -> "TimeSeries":
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
    self._check_valid_time()

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
) -> "TimeSeries":
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
    self._check_valid_time()

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
    self._check_valid_time()

    indexes = self._get_event_indexes(name)
    return len(indexes)


def remove_duplicate_events(self, *, in_place: bool = False) -> "TimeSeries":
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
    >>> ts = ts.add_event(1e-12, "event1")
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
    self._check_valid_time()

    ts = self if in_place else self.copy()
    duplicates = ts._get_duplicate_event_indexes()
    for event_index in duplicates[-1::-1]:
        ts.events.pop(event_index)
    return ts


def trim_events(self, *, in_place: bool = False) -> "TimeSeries":
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
    >>> ts = ktk.TimeSeries(time=np.arange(10))
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
    self._check_valid_time()

    ts = self if in_place else self.copy()

    events = deepcopy(ts.events)
    ts.events = []
    for event in events:
        if event.time <= np.max(ts.time) and event.time >= np.min(ts.time):
            ts.add_event(event.time, event.name, in_place=True)
    return ts
