#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020 Félix Chénier

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
Identify cycles and time-normalize data.

Warning
-------
This module is in early development and may change in the future.

"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import numpy as np
from ktk.timeseries import TimeSeries, TimeSeriesEvent
import warnings
from typing import Sequence, Optional, List, Dict, Tuple
import matplotlib.pyplot as plt  # For doc


def detect_cycles(ts: TimeSeries,
                  data_key: str, /,
                  event_name1: str,
                  event_name2: str,
                  raising_threshold: float,
                  falling_threshold: float, *,
                  min_length1: float = 0,
                  min_length2: float = 0,
                  target_height1: Optional[float] = None,
                  target_height2: Optional[float] = None,
                  ) -> TimeSeries:
    """
    Detect cycles in a TimeSeries based on a dual threshold approach.

    Warning
    -------
    This function is currently experimental and may change signature and
    behaviour in the future.


    This function detects biphasic cycles and identifies the transitions as
    new events in the output TimeSeries. These new events are named:

        - `event_name1`:
          corresponds to the start of phase 1
        - `event_name2`:
          corresponds to the start of phase 2
        - '_':
          corresponds to the end of the cycle. Apart from the last cycle,
          this always coincides with the start of the next phase 1.

    Parameters
    ----------
    ts
        TimeSeries to analyze.
    data_key
        Name of the data key to analyze in the TimeSeries. This signal must be
        high during phase 1, and low during phase 2. For example, one could
        use the absolute ground reaction force to detect stance (phase 1) and
        swing (phase 2).
    event_name1
        Name of the events in the output TimeSeries that corresponds to the
        start of phase 1.
    event_name2
        Name of the events in the output TimeSeries that corresponds to the
        start of phase 2.
    raising_threshold:
        Value to cross upward to register the start of phase 1.
    falling_threshold:
        Value to cross downward to register the start of phase 2.
    min_length1
        Optional. Minimal time of phase 1 in seconds.
    min_length2
        Optional. Minimal time of phase 2 in seconds.
    target_height1
        Optional. A value that the signal must cross in phase 1. Use None for
        no target height.
    target_height2
        Optional. A value that the signal must cross in phase 2. Use None for
        no target height.

    Returns
    -------
    TimeSeries
        A copy of ts with the events added.

    """
    # Convert optional Nones to floats
    if target_height1 is None:
        target_height1 = -np.Inf
    if target_height2 is None:
        target_height2 = np.Inf



    # Find the pushes
    time = ts.time
    data = ts.data[data_key]

    # To wait for a first release, which allows to ensure the cycle will
    # begin with event1:
    is_first_part_of_cycle = True

    events = []

    for i in range(time.shape[0]):

        if (is_first_part_of_cycle is True and
                data[i] < falling_threshold):

            is_first_part_of_cycle = False
            events.append(TimeSeriesEvent(time[i], event_name2))

        elif (is_first_part_of_cycle is False and
              data[i] > raising_threshold):

            is_first_part_of_cycle = True
            events.append(TimeSeriesEvent(time[i], event_name1))

    # The first event in list was only to initiate the list. We must remove it.
    events = events[1:]

    # Remove cycles where criteria are not reached.
    valid_events = []

    for i_event in range(0, len(events) - 1, 2):
        time1 = events[i_event].time
        time2 = events[i_event + 1].time
        try:
            time3 = events[i_event + 2].time
        except IndexError:
            time3 = np.Inf

        sub_ts1 = ts.get_ts_between_times(time1, time2)
        sub_ts2 = ts.get_ts_between_times(time1, time3)

        if (time2 - time1 >= min_length1 and
                time3 - time2 >= min_length2 and
                np.max(sub_ts1.data[data_key]) >= target_height1 and
                np.min(sub_ts2.data[data_key]) <= target_height2):
            # Save it.
            valid_events.append(events[i_event])
            valid_events.append(events[i_event + 1])
            if not np.isinf(time3):
                valid_events.append(TimeSeriesEvent(time3, '_'))

    # Form the output timeseries
    tsout = ts.copy()
    for event in valid_events:
        tsout.add_event(event.time, event.name)
    tsout.sort_events()

    return tsout


def time_normalize(
        ts: TimeSeries, /,
        event_name1: str,
        event_name2: str, *,
        n_points: int = 100,
        ) -> TimeSeries:
    """
    Time-normalize cycles in a TimeSeries.

    This method time-normalizes the TimeSeries at each cycle defined by
    event_name1 and event_name2 on n_points. The time-normalized cycles are
    put end to end. For example, for a TimeSeries that contains three
    cycles, a time normalization with 100 points will give a TimeSeries
    of length 300. The TimeSeries' events are also time-normalized.

    To time-normalize a cycle between two events of the same name, use '_' for
    event_name2. For example, to time-normalize a TimeSeries between each
    heel strike and the next heel strike:

        ktk.cycles.time_normalize(ts, 'heelstrike', '_')

    Parameters
    ----------
    ts
        The TimeSeries to analyze.
    event_name1
        The event name that correspond to the begin of a cycle.
    event_name2
        The event name that correspond to the end of a cycle.
    n_points
        Optional. The number of points of the output TimeSeries.

    Returns
    -------
    TimeSeries
        A new TimeSeries where each cycle has been time-normalized.
    """
    # Find the final number of cycles
    if len(ts.events) < 2:
        raise(ValueError('No cycle can be defined from these event names.'))

    if event_name1 == event_name2:
        warnings.warn("It is better practice to use '_' as the second event "
                      "name instead of repeating the same name twice.")
        event_name2 = '_'

    i_cycle = 0

    # Initialize the destination TimeSeries
    dest_ts = ts.copy()
    dest_ts.events = []
    dest_ts.time_info['Unit'] = '%'

    dest_data = {}  # type: Dict[str, List[np.ndarray]]
    dest_data_shape = {}  #type: Dict[str, Tuple[int]]
    dest_time = []  # type: List[np.ndarray]

    while True:
        # Get the TimeSeries for this cycle
        subts = ts.get_ts_between_events(event_name1, event_name2,
                                         i_cycle, i_cycle,
                                         inclusive=True)

        if subts.time.shape[0] == 0:  # Empty TimeSeries, retry
            subts = ts.get_ts_between_events(event_name1, event_name1,
                                             i_cycle, i_cycle + 1,
                                             inclusive=True)

        if subts.time.shape[0] == 0:  # We are done. Quit the loop.
            break

        subts.trim_events()
        subts.sort_events()

        # Separate start/end events from the other
        start_end_events = []
        other_events = []
        for event in subts.events:
            if event.name == event_name1 or event.name == event_name2:
                start_end_events.append(event)
            else:
                other_events.append(event)

        original_start = subts.events[0].time
        original_stop = subts.events[-1].time

        # Resample this TimeSeries on n_points
        try:
            subts.resample(
                np.linspace(subts.time[0], subts.time[-1], n_points))
        except ValueError:
            subts.resample(
                np.linspace(subts.time[0], subts.time[-1], n_points),
                fill_value='extrapolate')
            warnings.warn(f"Cycle {i_cycle} has been extrapolated.")

        # Resample the events and add them to the
        # destination TimeSeries
        dest_ts.add_event(i_cycle * n_points, event_name1)
        dest_ts.add_event((i_cycle + 1) * n_points - 1, '_')
        for i_event, event in enumerate(other_events):

            # Resample
            new_time = ((event.time - original_start) /
                        (original_stop - original_start) *
                        (n_points - 1)) + i_cycle * n_points
            dest_ts.add_event(new_time, event.name)

        # Add this cycle to dest_time and dest_data
        for key in subts.data:
            if key not in dest_data:
                dest_data[key] = []
                dest_data_shape[key] = ts.data[key].shape
            dest_data[key].append(subts.data[key])

        i_cycle += 1

    n_cycles = i_cycle

    # Put back dest_time and dest_data in dest_ts
    dest_ts.time = np.arange(n_cycles * n_points)
    for key in ts.data:
        # Stack the data into a [cycle, percent, values] shape
        temp = np.array(dest_data[key])
        # Reshape to put all cycles end to end
        new_shape = list(dest_data_shape[key])
        new_shape[0] = n_cycles * n_points
        dest_ts.data[key] = np.reshape(temp, new_shape)

    dest_ts.sort_events()
    return dest_ts


def stack_normalized_data(
        ts: TimeSeries, /,
        n_points: int = 100) -> Dict[str, np.ndarray]:
    """
    Stack time-normalized TimeSeries' data into a dict of arrays.

    Warning
    -------
    This function is currently experimental and may change signature and
    behaviour in the future.

    This methods returns the data of a time-normalized TimeSeries as a dict
    where each key corresponds to a TimeSeries' data, and contains a numpy
    array where the first dimension if the cycle, the second dimension is the
    percentage of cycle, and the other dimensions are the data itself.

    Parameters
    ----------
    ts
        The time-normalized TimeSeries.
    n_points
        Optional. The number of points the TimeSeries has been time-normalized
        on.

    Returns
    -------
    Dict[str, np.ndarray]

    Example
    -------
    >>> # Create a time-normalized TimeSeries

    """
    if np.mod(len(ts.time), n_points) != 0:
        raise(ValueError(
                'It seems that this TimeSeries is not time-normalized.'))

    data = dict()
    for key in ts.data.keys():
        current_shape = ts.data[key].shape
        new_shape = [-1, n_points]
        for i in range(1, len(current_shape)):
            new_shape.append(ts.data[key].shape[i])
        data[key] = ts.data[key].reshape(new_shape, order='C')
    return data


def stack_normalized_events(
        ts: TimeSeries, /,
        n_points: int = 100) -> Dict[str, np.ndarray]:
    """
    Stack time-normalized TimeSeries' events into a dict of arrays.

    Warning
    -------
    This function is currently experimental and may change signature and
    behaviour in the future.


    This methods returns the a dictionary where each key corresponds to an
    event name, and contains a 2d numpy array that contains the event's
    normalized time, with the first dimension being the cycle and the second
    dimension being the occurrence of this event during this cycle.

    Parameters
    ----------
    ts
        The time-normalized TimeSeries.
    n_points
        Optional. The number of points the TimeSeries has been time-normalized
        on.

    Returns
    -------
    Dict[str, np.ndarray]

    Example
    -------
    >>> # Create a TimeSeries with different time-normalized events
    >>> ts = ktk.TimeSeries(time=np.arange(400))  # 4 cycles of 100%
    >>> ts.add_event(9, 'event1')    # event1 at 9% of cycle 0
    >>> ts.add_event(110, 'event1')  # event1 at 10% of cycle 1
    >>> ts.add_event(312, 'event1')  # event1 at 12% of cycle 3
    >>> ts.add_event(382, 'event1')  # 2nd occurr. event1 at 82% of cycle 3

    >>> # Stack these events
    >>> events = ktk.cycles.stack_normalized_events(ts)
    >>> events['event1']
    array([[ 9., nan],
           [10., nan],
           [nan, nan],
           [12., 82.]])

    See also
    --------
    ktk.cycles.stack_normalized_data(),
    ktk.cycles.unstack_normalized_data()

    """
    ts = ts.copy()
    ts.sort_events()

    n_cycles = int(ts.time.shape[0] / n_points)
    out = {}  # type: Dict[str, np.ndarray]

    # Count the number of occurence of every event in every cycle
    n_occurrences = {}  # type: Dict[str, np.ndarray]
    for event in ts.events:
        if event.name not in n_occurrences:
            n_occurrences[event.name] = np.zeros(n_cycles)
        event_cycle = int(event.time / n_points)
        n_occurrences[event.name][event_cycle] += 1

    # Initialize the output
    for event_name in n_occurrences:
        out[event_name] = (np.nan *
                           np.ones(
                               (n_cycles,
                                int(n_occurrences[event_name][event_cycle]))))

    # Fill the output
    for event in ts.events:
        event_cycle = int(event.time / n_points)

        # Find the occurrence
        occurrence = 0
        while not np.isnan(out[event.name][event_cycle, occurrence]):
            occurrence += 1

        # Fill this occurrence
        out[event.name][event_cycle, occurrence] = np.mod(
            event.time, n_points)

    return out


def most_repeatable_cycles(data: np.ndarray) -> List[int]:
    """
    Get the indexes of the most repeatable cycles in TimeSeries or array.

    WARNING
    -------
    This function is currently experimental and may change signature and
    behaviour in the future.


    This function returns an ordered list of the most repeatable to the least
    repeatable cycles.

    Its algorithm is to recursively discards the cycle than maximizes the
    root-mean-square error between the cycle and the average of every
    remaining cycle, until there are only two cycles remaining. The function
    returns a list that is the reverse order of cycle removal: first the two
    last cycles, then the last-removed cycle, and so on. If two cycles are as
    equivalently repeatable, they are returned in order of appearance.

    Note
    ----
    Cycles that include at least one NaN are excluded.


    Parameters
    ----------
    data
        Stacked time-normalized data to analyze, in the shape
        (n_cycles, n_points).

    Returns
    -------
    List[int]
        List of indexes corresponding to the cycles in most to least
        repeatable order.

    Example
    -------
    >>> import ktk, numpy as np
    >>> # Create a data sample with four different cycles, the most different
    >>> # begin cycle 2 (cos instead of sin), then cycle 0.
    >>> x = np.arange(0, 10, 0.1)
    >>> data = np.array([[np.sin(x)],
        [np.sin(x) + 0.14], \
        [np.cos(x) + 0.14], \
        [np.sin(x) + 0.15]])

    .. plot::
        :format: doctest

        >>> # Plot these data
        for cycle in range(4):
            plt.plot(data, label=f"Cycle {cycle}")
        plt.legend()
        plt.show()

    >>> ktk.cycles.most_repeatable_cycles(data)
    [1, 3, 0, 2]

    """
    data = data.copy()
    n_cycles = data.shape[0]
    out_cycles = []  # type: List[int]

    # Exclude cycles with nans: put nans for all data of this cycle
    for i_cycle in range(n_cycles-1, -1, -1):
        if np.isnan(np.sum(data[i_cycle])):
            data[i_cycle] = np.nan
            out_cycles.append(i_cycle)

    # --- Iteratively remove the cycle that is the most different from the
    #     mean of the remaining cycles.
    while len(out_cycles) < n_cycles - 2:

        current_mean_cycle = np.nanmean(data, axis=0)

        rms = np.zeros(n_cycles)

        for i_curve in range(n_cycles-1, -1, -1):
            rms[i_curve] = np.sqrt(np.mean(np.sum(
                (data[i_curve] - current_mean_cycle) ** 2)))


        i_cycle = np.nanargmax(rms)
        out_cycles.append(i_cycle)
        data[i_cycle] = np.nan

    # Find the two remaining cycles
    set_all = set(range(n_cycles))
    set_out = set(out_cycles)
    remain = sorted(list(set_all - set_out))
    out_cycles.append(remain[1])
    out_cycles.append(remain[0])

    return out_cycles[-1::-1]


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
