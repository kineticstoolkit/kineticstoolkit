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
Implement the classes used as properties by TimeSeries.

"""

from __future__ import annotations


__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2023 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


from dataclasses import dataclass
import numpy as np


class TimeSeriesDataDict(dict):
    """Data dictionary that checks sizes and converts to NumPy arrays."""

    def __init__(self, ts):
        """Initialize the class, with the parent TimeSeries as an argument."""
        self._ts = ts

    def __setitem__(self, key, value):
        """
        Overload setting an element.

        Ensure that the key is a string and that the value's size is consistent
        with the TimeSeries' time and already present data. We only check the
        first data because this function would have already failed before if
        all data were to have different first dimension sizes.

        """
        if not isinstance(key, str):
            raise ValueError("Data key must be a string.")
        value = np.array(value)

        if (self._ts.time.shape[0]) > 0 and (
            value.shape[0] != self._ts.time.shape[0]
        ):
            raise ValueError("Size mismatch")

        if (len(self) > 0) and (
            value.shape[0] != self[list(self.keys())[0]].shape[0]
        ):
            raise ValueError("Size mismatch")

        super(TimeSeriesDataDict, self).__setitem__(key, value)


@dataclass
class TimeSeriesEvent:
    """
    Define an event in a timeseries.

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
    >>> event = ktk.TimeSeriesEvent(time=1.5, name='event_name')
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
        >>> event = ktk.TimeSeriesEvent(time=1.5, name='event_name')
        >>> event._to_tuple()
        (1.5, 'event_name')

        """
        return (self.time, self.name)

    def _to_list(self) -> list[float | str]:
        """
        Convert a TimeSeriesEvent to a list.

        Example
        -------
        >>> event = ktk.TimeSeriesEvent(time=1.5, name='event_name')
        >>> event._to_list()
        [1.5, 'event_name']

        """
        return [self.time, self.name]

    def _to_dict(self) -> dict[str, float | str]:
        """
        Convert a TimeSeriesEvent to a dict.

        Example
        -------
        >>> event = ktk.TimeSeriesEvent(time=1.5, name='event_name')
        >>> event._to_dict()
        {'Time': 1.5, 'Name': 'event_name'}

        """
        return {"Time": self.time, "Name": self.name}


class TimeSeriesEventList(list):
    """Event list that ensures that every element is a TimeSeriesEvent."""
