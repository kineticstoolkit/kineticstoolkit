import numpy as np
from ktk import TimeSeries


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
            event.time = ((event.time - original_start) /
                          (original_stop - original_start) *
                          (n_points-1))
            if event.time >= 0 and event.time < (n_points-1):
                event.time += 100 * i_cycle
                dest_ts.events.append(event)

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
