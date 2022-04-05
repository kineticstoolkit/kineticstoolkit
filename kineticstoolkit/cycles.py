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

"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import numpy as np
from kineticstoolkit.timeseries import TimeSeries, TimeSeriesEvent
from kineticstoolkit.decorators import directory
import warnings
from typing import List, Dict, Tuple, Sequence, Optional


def detect_cycles(ts: TimeSeries,
                  data_key: str, *,
                  event_names: Sequence[str] = ['phase1', 'phase2'],
                  thresholds: Sequence[float] = [0., 1.],
                  directions: Sequence[str] = ['rising', 'falling'],
                  min_durations: Sequence[float] = [0., 0.],
                  max_durations: Sequence[float] = [np.Inf, np.Inf],
                  min_peak_heights: Sequence[float] = [-np.Inf, -np.Inf],
                  max_peak_heights: Sequence[float] = [np.Inf, np.Inf],
                  **kwargs,
                  ) -> TimeSeries:
    """
    Detect cycles in a TimeSeries based on a dual threshold approach.

    This function detects biphasic cycles and identifies the transitions as
    new events in the output TimeSeries. These new events are named:

    - `event_name1`:
      corresponds to the start of phase 1
    - `event_name2`:
      corresponds to the start of phase 2
    - '_':
      corresponds to the end of the cycle.

    Warning
    -------
    This function, which has been introduced in 0.4, is still experimental and
    may change signature or behaviour in the future.

    Parameters
    ----------
    ts
        TimeSeries to analyze.
    data_key
        Name of the data key to analyze in the TimeSeries.
    event_names
        Optional. Name of the events to add in the output TimeSeries. Default
        is ['phase1', 'phase2'].
    thresholds
        Optional. Value to cross to register phase changes. Default is
        [0., 1.].
    directions
        Optional. Direction to cross thresholds to register phase changes.
        Either ['rising', 'falling'] or ['falling', 'rising']. Default is
        ['rising', 'falling'].
    min_durations
        Optional. Minimal phase durations in seconds. Default is [0., 0.].
    max_durations
        Optional. Maximal phase durations in seconds. Default is
        [np.Inf, np.Inf]
    min_peak_heights
        Optional. Minimal peak values to be reached in both phases. Default is
        [-np.Inf, -np.Inf].
    max_peak_heights
        Optional. Maximal peak values to be reached in both phases. Default is
        [np.Inf, np.Inf].

    Returns
    -------
    TimeSeries
        A copy of ts with the events added.

    """
    # lowercase directions[0] once
    directions[0] = directions[0].lower()  # type: ignore
    if directions[0] != 'rising' and directions[0] != 'falling':
        raise ValueError("directions[0] must be 'rising' or 'falling'")

    # Find the pushes
    time = ts.time
    data = ts.data[data_key]

    events = []

    is_phase1 = True

    for i in range(time.shape[0]):

        if directions[0] == 'rising':
            crossing1 = data[i] >= thresholds[0]
            crossing2 = data[i] <= thresholds[1]
        else:
            crossing1 = data[i] <= thresholds[0]
            crossing2 = data[i] >= thresholds[1]

        if is_phase1 and crossing1:

            is_phase1 = False
            events.append(TimeSeriesEvent(time[i], event_names[0]))

        elif (not is_phase1) and crossing2:

            is_phase1 = True
            events.append(TimeSeriesEvent(time[i], event_names[1]))

    # Ensure that we start with event_name1 and that it's not on time0
    while (events[0].name != event_names[0]) or (events[0].time == time[0]):
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

        sub_ts1 = ts.get_ts_between_times(time1, time2, inclusive=True)
        sub_ts2 = ts.get_ts_between_times(time1, time3, inclusive=True)

        if directions[0] == 'rising':
            the_peak1 = np.max(sub_ts1.data[data_key])
            the_peak2 = np.min(sub_ts2.data[data_key])
        else:
            the_peak1 = np.min(sub_ts1.data[data_key])
            the_peak2 = np.max(sub_ts2.data[data_key])

        if (time2 - time1 >= min_durations[0] and
                time2 - time1 <= max_durations[0] and
                time3 - time2 >= min_durations[1] and
                time3 - time2 <= max_durations[1] and
                the_peak1 >= min_peak_heights[0] and
                the_peak1 <= max_peak_heights[0] and
                the_peak2 >= min_peak_heights[1] and
                the_peak2 <= max_peak_heights[1]):
            # Save it.
            valid_events.append(events[i_event])
            valid_events.append(events[i_event + 1])
            if not np.isinf(time3):
                valid_events.append(TimeSeriesEvent(time3, '_'))

    # Form the output timeseries
    tsout = ts.copy()
    for event in valid_events:
        tsout = tsout.add_event(event.time, event.name)
    tsout.sort_events()

    return tsout


def time_normalize(
        ts: TimeSeries, /,
        event_name1: str,
        event_name2: str, *,
        n_points: int = 100,
        span: Optional[Sequence[int]] = None,
) -> TimeSeries:
    """
    Time-normalize cycles in a TimeSeries.

    This method time-normalizes the TimeSeries at each cycle defined by
    event_name1 and event_name2 on n_points. The time-normalized cycles are
    put end to end. For example, for a TimeSeries that contains three
    cycles, a time normalization with 100 points will give a TimeSeries
    of length 300. The TimeSeries' events are also time-normalized, including
    event_name1 but with event_name2 renamed as '_'.

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
    span
        Optional. Specifies which normalized points to include in the output
        TimeSeries. See note below.

    Returns
    -------
    TimeSeries
        A new TimeSeries where each cycle has been time-normalized.

    Notes
    -----
    The span argument is experimental and has been introduced in version 0.4.
    Use it to define which normalized points to include in the output
    TimeSeries. For example, to normalize in percents and to include only data
    from 10 to 90% of each cycle, assign 100 to n_points and [10, 90] to span.
    The resulting TimeSeries will then be expressed in percents and wrap each
    80 points. It is also possible to include pre-cycle or post-cycle data.
    For example, to normalize in percents and to include 20% pre-cycle and 15%
    post-cycle, assign 100 to n_points and [-20, 15] to span. The resulting
    TimeSeries will then wrap each 135 points with the cycles starting at 20,
    155, etc. and ending at 119, 254, etc. For each cycle, events outside the
    0-100% spans are ignored.

    """
    # Optional span
    if span is None:
        span = [0, n_points]

    # Find the final number of cycles
    if len(ts.events) < 2:
        raise(ValueError('No cycle can be defined from these event names.'))

    i_cycle = 0

    # Initialize the destination TimeSeries
    dest_ts = ts.copy()
    dest_ts.events = []
    if n_points == 100:
        dest_ts.time_info['Unit'] = '%'
    else:
        dest_ts.time_info['Unit'] = f"1/{n_points}"

    dest_data = {}  # type: Dict[str, List[np.ndarray]]
    dest_data_shape = {}  # type: Dict[str, Tuple[int, ...]]

    # Go through all cycles
    while True:

        # Get the begin time for this cycle
        begin_time = ts.get_event_time(event_name1, i_cycle)

        # Get the end time for this cycle
        end_cycle = 0
        end_time = ts.get_event_time(event_name2, end_cycle)
        while end_time <= begin_time:
            end_cycle += 1
            end_time = ts.get_event_time(event_name2, end_cycle)

        # We are done. Quit the loop.
        if np.isnan(begin_time) or np.isnan(end_time):
            break

        # Get the extended begin and end times considering relative_span
        extended_begin_time = (begin_time +
                               span[0] / n_points *
                               (end_time - begin_time))
        extended_end_time = (begin_time +
                             span[1] / n_points *
                             (end_time - begin_time))

        # Extract this cycle
        subts = ts.get_ts_between_times(
            extended_begin_time, extended_end_time, inclusive=True)
        subts_backup = subts.copy()

        if subts.time.shape[0] == 0:
            raise ValueError("")

        # Resample this cycle on span + 1
        # (and get the first points after)
        try:
            subts = subts.resample(
                np.linspace(extended_begin_time,
                            extended_end_time,
                            span[1] - span[0] + 1))
        except ValueError:
            subts = subts_backup  # In case the try messed with subts
            subts = subts.resample(
                np.linspace(extended_begin_time,
                            extended_end_time,
                            span[1] - span[0] + 1),
                fill_value='extrapolate')
            warnings.warn(f"Cycle {i_cycle} has been extrapolated.")

        # Keep only the first points (the last one belongs to the next cycle)
        subts = subts.get_ts_between_indexes(
            0, span[1] - span[0] - 1, inclusive=True)

        # Keep only the events in the unextended span
        events = []
        for event in subts.events:
            if event.time >= begin_time and event.time < end_time:
                events.append(event)
        subts.events = events
        subts = subts.sort_events()

        # Separate start/end events from the other
        start_end_events = []
        other_events = []
        for event in subts.events:
            if event.name == event_name1 or event.name == event_name2:
                start_end_events.append(event)
            else:
                other_events.append(event)

        # Add event_name1 at the beginning and end (duplicates will be
        # cancelled at the end)
        dest_ts = dest_ts.add_event(
            -span[0] + i_cycle * (span[1] - span[0]),
            event_name1
        )
        dest_ts = dest_ts.add_event(
            -span[0] + n_points + i_cycle * (span[1] - span[0]),
            '_'
        )

        # Add the other events
        def time_to_normalized_time(time):
            """Resample the events times."""
            return ((time - extended_begin_time) /
                    (extended_end_time - extended_begin_time) *
                    (span[1] - span[0]) + i_cycle * (span[1] - span[0]))

        for i_event, event in enumerate(other_events):

            # Resample
            new_time = time_to_normalized_time(event.time)
            dest_ts = dest_ts.add_event(new_time, event.name)

        # Add this cycle to dest_time and dest_data
        for key in subts.data:
            if key not in dest_data:
                dest_data[key] = []
                dest_data_shape[key] = ts.data[key].shape
            dest_data[key].append(subts.data[key])

        i_cycle += 1

    n_cycles = i_cycle

    # Put back dest_time and dest_data in dest_ts
    dest_ts.time = 1.0 * np.arange(n_cycles * (span[1] - span[0]))
    for key in ts.data:
        # Stack the data into a [cycle, percent, values] shape
        temp = np.array(dest_data[key])
        # Reshape to put all cycles end to end
        new_shape = list(dest_data_shape[key])
        new_shape[0] = n_cycles * (span[1] - span[0])
        dest_ts.data[key] = np.reshape(temp, new_shape)

    dest_ts = dest_ts.sort_events()
    return dest_ts


def stack(ts: TimeSeries, /, n_points: int = 100) -> Dict[str, np.ndarray]:
    """
    Stack time-normalized TimeSeries' data into a dict of arrays.

    This methods returns the data of a time-normalized TimeSeries as a dict
    where each key corresponds to a TimeSeries' data, and contains a numpy
    array where the first dimension is the cycle, the second dimension is the
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

    See Also
    --------
    kineticstoolkit.cycles.unstack

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


def unstack(data: Dict[str, np.ndarray], /) -> TimeSeries:
    """
    Unstack time-normalized data from a dict of arrays to a TimeSeries.

    This methods creates a time-normalized TimeSeries by putting each cycle
    from the provided data dictionary end to end.

    Parameters
    ----------
    data
        A dict where each key contains a numpy array where the first dimension
        is the cycle, the second dimension is the percentage of cycle, and
        the other dimensions are the data itself.

    Returns
    -------
    TimeSeries

    See Also
    --------
    ktk.cycles.stack

    """
    ts = TimeSeries()
    for key in data.keys():
        current_shape = data[key].shape
        n_cycles = current_shape[0]
        n_points = current_shape[1]
        ts.data[key] = data[key].reshape([n_cycles * n_points], order='C')
    ts.time = np.arange(n_cycles * n_points)
    ts.time_info['Unit'] = ''
    return ts


# The stack_events function is working but commented for now, since I could not
# figure an obvious, undiscutable way to represent its output (use lists,
# TimeSeriesEvents, numpy arrays?). It's also unclear for me how to integrate
# with the standard stack function and its unstack counterpart.
#
# def stack_events(
#         ts: TimeSeries, /,
#         n_points: int = 100) -> Dict[str, np.ndarray]:
#     """
#     Stack time-normalized TimeSeries' events into a dict of arrays.

#     This methods returns the a dictionary where each key corresponds to an
#     event name, and contains a 2d numpy array that contains the event's
#     normalized time, with the first dimension being the cycle and the second
#     dimension being the occurrence of this event during this cycle.

#     Warning
#     -------
#     This function is currently experimental and may change signature and
#     behaviour in the future.

#     Parameters
#     ----------
#     ts
#         The time-normalized TimeSeries.
#     n_points
#         Optional. The number of points the TimeSeries has been
#         time-normalized on.

#     Returns
#     -------
#     Dict[str, np.ndarray]

#     Example
#     -------
#     >>> import kineticstoolkit.lab as ktk
#     >>> # Create a TimeSeries with different time-normalized events
#     >>> ts = ktk.TimeSeries(time=np.arange(400))  # 4 cycles of 100%
#     >>> ts = ts.add_event(9, 'event1')    # event1 at 9% of cycle 0
#     >>> ts = ts.add_event(110, 'event1')  # event1 at 10% of cycle 1
#     >>> ts = ts.add_event(312, 'event1')  # event1 at 12% of cycle 3
#     >>> ts = ts.add_event(382, 'event1')  # 2nd occ. event1 at 82% of cycle 3

#     >>> # Stack these events
#     >>> events = ktk.cycles.stack_events(ts)
#     >>> events['event1']
#     [[9.0], [10.0], [], [12.0, 82.0]]

#     """
#     ts = ts.copy()
#     ts.sort_events()

#     n_cycles = int(ts.time.shape[0] / n_points)
#     out = {}  # type: Dict[str, np.ndarray]

#     # Init
#     for event in ts.events:
#         if event.name not in out:
#             out[event.name] = [[] for i in range(n_cycles)]

#     for event in ts.events:
#         event_cycle = int(event.time / n_points)

#         out[event.name][event_cycle].append(np.mod(event.time, n_points))

#     return out


def most_repeatable_cycles(data: np.ndarray, /) -> List[int]:
    """
    Get the indexes of the most repeatable cycles in TimeSeries or array.

    This function returns an ordered list of the most repeatable to the least
    repeatable cycles.

    It works by recursively discarding the cycle than maximizes the
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
    >>> import kineticstoolkit.lab as ktk
    >>> import numpy as np
    >>> # Create a data sample with four different cycles, the most different
    >>> # begin cycle 2 (cos instead of sin), then cycle 0.
    >>> x = np.arange(0, 10, 0.1)
    >>> data = np.array([[np.sin(x)], \
        [np.sin(x) + 0.14], \
        [np.cos(x) + 0.14], \
        [np.sin(x) + 0.15]])

    >>> ktk.cycles.most_repeatable_cycles(data)
    [1, 3, 0, 2]

    """
    data = data.copy()
    n_cycles = data.shape[0]
    out_cycles = []  # type: List[int]
    done_cycles = []  # type: List[int]  # Like out_cycles but includes NaNs

    # Exclude cycles with nans: put nans for all data of this cycle
    for i_cycle in range(n_cycles-1, -1, -1):
        if np.isnan(np.sum(data[i_cycle])):
            data[i_cycle] = np.nan
            done_cycles.append(i_cycle)

    # Iteratively remove the cycle that is the most different from the
    # mean of the remaining cycles.
    while len(done_cycles) < n_cycles - 2:

        current_mean_cycle = np.nanmean(data, axis=0)

        rms = np.zeros(n_cycles)

        for i_curve in range(n_cycles-1, -1, -1):
            rms[i_curve] = np.sqrt(np.mean(np.sum(
                (data[i_curve] - current_mean_cycle) ** 2)))

        i_cycle = int(np.nanargmax(rms))
        out_cycles.append(i_cycle)
        done_cycles.append(i_cycle)
        data[i_cycle] = np.nan

    # Find the two remaining cycles
    set_all = set(range(n_cycles))
    set_done = set(done_cycles)
    remain = sorted(list(set_all - set_done))
    if len(remain) > 1:
        out_cycles.append(remain[1])
    if len(remain) > 0:
        out_cycles.append(remain[0])

    return out_cycles[-1::-1]


module_locals = locals()


def __dir__():
    return directory(module_locals)


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
