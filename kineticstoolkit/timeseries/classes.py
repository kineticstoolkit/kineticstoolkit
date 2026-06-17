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
"""Provide the classes uses by TimeSeries."""

from dataclasses import dataclass

import numpy as np

from kineticstoolkit.typing_ import check_param


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
        super().__setitem__(index, event)
        # Sort the events
        self.sort()

    def append(self, value):
        """Ensure the appended value is a TimeSeriesEvent."""
        super().append(None)
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

        super().__setitem__(key, to_set)


class TimeSeriesInfoDict(dict):
    """Info dictionary that ensures it is well formatted."""

    def __init__(self, source: dict = {}):
        """Initialize the class instance using a source dictionary."""
        check_param("source", source, dict, key_type=str)

        for key in source:
            self[key] = source[key]

    def __setitem__(self, key, value):
        """Check the structure and assign."""
        check_param("key", key, str)
        to_set = TimeSeriesStringDict(value)

        super().__setitem__(key, to_set)


class TimeSeriesStringDict(dict):
    """Dictionary that ensures it only has string keys."""

    def __init__(self, source: dict = {}):
        """Initialize the class instance using a source dictionary."""
        check_param("source", source, dict, key_type=str)
        for key in source:
            self[key] = source[key]

    def __setitem__(self, key, value):
        """Ensure the kay is a string."""
        check_param("key", key, str)

        super().__setitem__(key, value)


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
