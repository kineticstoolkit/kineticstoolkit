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
import ktk  # for doctests
from ktk.timeseries import TimeSeries, TimeSeriesEvent
from typing import Sequence, Optional, List, Dict


def find_cycles(ts: TimeSeries,
                data_key: str, /,
                event_names: Sequence[str],
                thresholds: Sequence[float],
                minimum_length: float = 0,
                minimum_height: float = 0) -> TimeSeries:
    """
    Find cycles in a TimeSeries based on a dual threshold approach.

    Warning
    -------
    This function is currently experimental and may change signature and
    behaviour in the future.

    Parameters
    ----------
    ts
        TimeSeries to analyze.
    data_key
        Name of the data key to analyze in the TimeSeries.
    event_names : tuple of 2 strings
        Name of the events that correspond to the start and end of the
        first phase.
    thresholds : tuple of 2 floats
        Thresholds that define the start and end of first phase. The first
        threshold is crossed while rising, and the second threshold is crossed
        while falling.
    minimum_length
        Optional. Minimal time of first phase. Cycles with first phase lasting
        less than minimum_length are rejected.
    minimum_height
        Optional. Minimum value the signal must reach in first phase. Cycles
        with first phase not reaching minimum_height are rejected.

    Returns
    -------
    TimeSeries
        A copy of ts with the added events.

    """
    # Find the pushes
    time = ts.time
    data = ts.data[data_key]

    # To wait for a first release, which allows to ensure the cycle will
    # begin with event1:
    is_first_part_of_cycle = True

    events = []

    for i in range(time.shape[0]):

        if (is_first_part_of_cycle is True) and (data[i] < thresholds[1]):

            is_first_part_of_cycle = False
            events.append(TimeSeriesEvent(time[i], event_names[1]))

        elif (is_first_part_of_cycle is False) and (data[i] > thresholds[0]):

            is_first_part_of_cycle = True
            events.append(TimeSeriesEvent(time[i], event_names[0]))

    # The first event in list was only to initiate the list. We must remove it.
    events = events[1:]

    # Remove cycles where criteria are not reached.
    valid_events = []

    for i_event in range(0, len(events) - 1, 2):
        time1 = events[i_event].time
        time2 = events[i_event + 1].time
        sub_ts = ts.get_ts_between_times(time1, time2)

        if (np.max(sub_ts.data[data_key]) >= minimum_height and
                time2 - time1 >= minimum_length):

            # Save it.
            valid_events.append(events[i_event])
            valid_events.append(events[i_event + 1])

    # Form the output timeseries
    tsout = ts.copy()
    for event in valid_events:
        tsout.add_event(event.time, event.name)
    tsout.sort_events()

    return tsout


def time_normalize(
        ts: TimeSeries,
        event_name1: str,
        event_name2: str,
        n_points: int = 100, *,
        out_event_name1: Optional[str] = None,
        out_event_name2: Optional[str] = '_',
        ) -> TimeSeries:
    """
    Time-normalize cycles in a TimeSeries.

    This method time-normalizes the TimeSeries at each cycle defined by
    event_name1 and event_name2 on n_points. The time-normalized cycles are
    put end to end. For example, for a TimeSeries that contains three
    cycles, a time normalization with 100 points will give a TimeSeries
    of length 300. The TimeSeries' events are also time-normalized.

    By default, event_name1 and event_name2 are also present in the resulting
    TimeSeries, but event_name2 is renamed to '_'. This is to ensure that the
    event names are not duplicated (which would be the case, for example if
    we normalize from event 'heel_strike' to next event 'heel_strike').

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
    out_event_name1
        Optional. The renamed event1 in the returned TimeSeries.
        Use None to use event_name1. Default is None.
    out_event_name2
        Optional. The renamed event2 in the returned TimeSeries.
        Use None to use event_name2. Default is '_'.

    Returns
    -------
    TimeSeries
        A new TimeSeries where each cycle has been time-normalized.
    """
    # Find the final number of cycles
    if len(ts.events) < 2:
        raise(ValueError('No cycle can be defined from these event names.'))

    n_cycles = np.min([
            np.sum(np.array(ts.events)[:, 1] == event_name1),
            np.sum(np.array(ts.events)[:, 1] == event_name2)])
    if event_name1 == event_name2:
        n_cycles -= 1
        event_offset = 1
    else:
        event_offset = 0

    if n_cycles <= 0:
        raise(ValueError('No cycle can be defined from these event names.'))

    # Initialize the destination TimeSeries
    dest_ts = ts.copy()
    dest_ts.events = []

    dest_ts.time = np.arange(n_points * n_cycles)
    dest_ts.time_info['Unit'] = '%'

    for key in ts.data.keys():
        new_shape = list(ts.data[key].shape)
        new_shape[0] = n_points * n_cycles
        dest_ts.data[key] = np.empty(new_shape)

    for i_cycle in range(n_cycles):
        # Get the TimeSeries for this cycle
        subts = ts.get_ts_between_events(event_name1, event_name2,
                                         i_cycle, i_cycle + event_offset,
                                         inclusive=True)
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

        # Rename the start and end events if required
        if out_event_name1 is not None:
            start_end_events[0].name = out_event_name1
        if out_event_name2 is not None:
            start_end_events[1].name = out_event_name2

        original_start = start_end_events[0].time
        original_stop = start_end_events[1].time

        # Resample this TimeSeries on n_points
        subts.resample(
            np.linspace(subts.time[0], subts.time[-1], n_points))

        # Resample the events and add them to the
        # destination TimeSeries
        for i_event, event in enumerate(start_end_events + other_events):

            # Resample
            new_time = ((event.time - original_start) /
                        (original_stop - original_start) *
                        (n_points - 1)) + i_cycle * n_points
            dest_ts.add_event(new_time, event.name)

        # Add this cycle to the destination TimeSeries
        for key in subts.data.keys():
            dest_ts.data[key][n_points * i_cycle:n_points * (i_cycle+1)] = \
                    subts.data[key]

    dest_ts.sort_events()
    return dest_ts


def get_reshaped_data(
        ts: TimeSeries,
        n_points: int = 100) -> Dict[str, np.ndarray]:
    """
    Get reshaped data from a time-normalized TimeSeries.

    Warning
    -------
    This function is currently experimental and may change signature and
    behaviour in the future.

    This methods returns the data of a time-normalized TimeSeries, reshaped
    into this form:

    n_cycles x n_points x data_shape

    Parameters
    ----------
    n_points : int
        The number of points the TimeSeries has been time-normalized on.
        Default is 100.

    Returns
    -------
    data : dict
        A dictionary that contains every TimeSeries data keys, expressed into
        the form n_points x n_cycles x data_shape.
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


def most_repeatable_cycles(data: np.ndarray, n_cycles: int) -> List[int]:
    """
    Get the indexes of the most repeatable cycles in TimeSeries or array.

    Cycles that include at least one NaN are excluded.

    WARNING:
    This function is currently experimental and may change signature and
    behaviour in the future.

    Parameters
    ----------
    data
        Data to analyze, in the form of an array of shape (MxN). M corresponds
        to cycles, and N corresponds to normalized time.

    n_cycles
        Number of cycles to keep

    Returns
    -------
    List[int]
        List of indexes corresponding to the most repeatable cycles.

    Example
    -------
    We create 8 cycles with:

    - cycle 2 that is different from the others,
    - cycles 5 and 6 that contain NaNs.

        >>> data = np.array(
                [[0. ,    0.1,    0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                 [0. ,    0.1,    0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                 [3. ,    3.1,    3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9],
                 [0.1,    0.2,    0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ],
                 [0.1,    0.2,    0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ],
                 [np.nan, np.nan, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ],
                 [np.nan, 0.2,    0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ],
                 [0.1,    0.2,    0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]])

        >>> ktk.cycles.most_repeatable_cycles(data, 5)
        [0, 1, 3, 4, 7]

    """

    data = data.copy()
    cycles = list(range(data.shape[0]))

    # --- Exclude cycles with nans
    new_cycles = []
    for cycle in cycles:
        if ~np.isnan(np.sum(data[cycle])):
            new_cycles.append(cycle)
    cycles = new_cycles

    # --- Iteratively remove the cycle that is the most different from the
    #     mean of the remaining cycles.
    while len(cycles) > n_cycles:

        current_mean_cycle = np.mean(data[cycles], axis=0)

        rms = np.zeros(len(cycles))

        for i_curve in range(len(cycles)):
            rms[i_curve] = np.sqrt(np.mean(np.sum(
                (data[cycles[i_curve]] - current_mean_cycle) ** 2)))

        cycles.pop(np.argmax(rms))

    return cycles


if __name__ == "__main__":
    import doctest
    doctest.testmod()
