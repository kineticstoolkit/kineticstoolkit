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
"""Provide get_index_ methods for TimeSeries."""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2025 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


from typing import TYPE_CHECKING

import numpy as np

from kineticstoolkit.exceptions import (
    TimeSeriesRangeError,
)
from kineticstoolkit.typing_ import check_param

if TYPE_CHECKING:
    pass


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
            f"{self._get_time_unit()}."
        )

    self._check_increasing_time()

    if inclusive:
        mask = np.nonzero(self.time <= time)
    else:
        mask = np.nonzero(self.time < time)

    if mask[0].shape == (0,):
        _raise()

    return int(mask[0][-1])


def get_index_after_time(self, time: float, *, inclusive: bool = False) -> int:
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
            f"{self._get_time_unit()}."
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
    >>> ts = ktk.TimeSeries(time=np.arange(10) / 10)
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
    >>> ts = ktk.TimeSeries(time=np.arange(10) / 10)
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
    >>> ts = ktk.TimeSeries(time=np.arange(10) / 10)
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
