import numpy as np
import ktk


def find_cycles(ts, data_key, event_name1, event_name2, threshold1, threshold2,
                minimum_time1, minimum_time2):
    """
    Find cycles in a TimeSeries based on a dual threshold approach.

    Parameters
    ----------
    ts : TimeSeries
        TimeSeries to analyze.
    data_key : str
        Name of the data key to analyze in the TimeSeries.
    event_name1 : str
        Name of the events that correspond to threshold1.
    event_name2 : str
        Name of the events that correspond to threshold2.
    threshold1 : float
        Threshold to cross while rising to trigger event1.
    trigger2 : float
        Threshold to cross while falling to trigger event2.
    minimum_time1 : float
        Minimal time since event1 to consider event2 as a true event.
    minimum_time2 : float
        Minimal time since event2 to consider event1 as a true event.

    Returns
    -------
    TimeSeries : A copy of ts with the added events.

    """
    # Find the pushes
    time = ts.time
    data = ts.data[data_key]

    # To wait for a first release, which allows to ensure the cycle will
    # begin with event1:
    is_first_part_of_cycle = True
    is_first_cycle = True

    events = []

    for i in range(time.shape[0]):

        if (is_first_part_of_cycle is False) and (data[i] > threshold1):

            if ((is_first_cycle is True) or
                    (time[i] - events[-1].time >= minimum_time2)):

                is_first_part_of_cycle = True
                is_first_cycle = False
                events.append(ktk.TimeSeriesEvent(time[i], event_name1))

        elif (is_first_part_of_cycle is True) and (data[i] < threshold2):

            if ((is_first_cycle is True) or
                    (time[i] - events[-1].time >= minimum_time2)):

                is_first_part_of_cycle = False
                events.append(ktk.TimeSeriesEvent(time[i], event_name2))

    # The first event in list was only to initiate the list. We must remove it.
    events = events[1:]

    # Form the output timeseries
    tsout = ts.copy()
    for event in events:
        tsout.add_event(event.time, event.name)
    tsout.sort_events()

    return tsout



def time_normalize(ts, event_name1, event_name2, n_points=100):
    """
    Time-normalize cycles in a TimeSeries

    This method time-normalizes the TimeSeries at each cycle defined by
    event_name1 and event_name2 on n_points. The time-normalized cycles are
    put end to end. For example, for a TimeSeries that contains three
    cycles, a time normalization with 100 points will give a TimeSeries
    of length 300. The TimeSeries' events are also time-normalized.

    Parameters
    ----------
    event_name1, event_name2 : str
        The events that correspond to the begin and end of a cycle.
    n_points : int (optional)
        The number of points of the output TimeSeries (default is 100).

    Returns
    -------
    TimeSeries.
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
                                         i_cycle, i_cycle + event_offset)

        original_start = subts.time[0]
        original_stop = subts.time[-1]

        # Resample this TimeSeries on n_points
        subts.resample(np.linspace(subts.time[0], subts.time[-1],
                                   n_points), fill_value='extrapolate')

        # Resample the events and add the relevant ones to the
        # destination TimeSeries
        for i_event, event in enumerate(subts.events):

            tol = (subts.time[1] - subts.time[0]) / 2

            if ((event.time >= original_start - tol) and
                    (event.time < original_stop)):
                # Resample
                new_time = ((event.time - original_start) /
                              (original_stop - original_start) *
                              (n_points - 1)) + i_cycle * n_points
                dest_ts.add_event(new_time, event.name)

        # Add this cycle to the destination TimeSeries
        for key in subts.data.keys():
            dest_ts.data[key][n_points * i_cycle:n_points * (i_cycle+1)] = \
                    subts.data[key]

    # Assign the dest_ts data to ts and return.
    return dest_ts


def get_reshaped_time_normalized_data(ts, n_points=100):
    """
    Get reshaped data from a time-normalized TimeSeries

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
