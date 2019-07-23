"""
Module that manages the TimeSeries class.

Author: Felix Chenier
Date: July 2019
"""

import matplotlib.pyplot as plt
import numpy as np
import collections
import dataclasses

from copy import deepcopy as _deepcopy

from . import gui as _gui
from . import _repr


class TimeSeriesEvent(list):
    """
    Define an event in a timeseries.

    Attributes
    ----------
    time : float
        The time at which the event happened.
    name : str
        The name of the event.
    """

    def __init__(self, time=0., name='event'):
        list.__init__(self)
        self.append(float(time))
        self.append(str(name))

    @property
    def time(self):
        return self[0]

    @time.setter
    def time(self, time):
        self[0] = float(time)

    @property
    def name(self):
        return self[1]

    @name.setter
    def name(self, name):
        self[0] = str(name)


#    def __str__(self):
#        """Return the string representation of the TimeSeriesEvent."""
#        return 'time: ' + str(self.time) + ', name: ' + str(self.name)
#
#    def __repr__(self):
#        """Return the string representation of the TimeSeriesEvent."""
#        return '<' + str(self.time) + ' ' + str(self.name) + '>'

class TimeSeries():
    """
    A class that implements TimeSeries.

    This class implements a Timeseries in a way that resembles the timeseries
    and tscollection found in Matlab.

    Attributes
    ----------
        time : 1-dimension np.array. Default value is [].
            Contains the time vector

        data : dict. Default value is {}.
            Contains the data, where each element contains a np.array which
            first dimension corresponds to time.

        time_info : dict. Default value is {'unit': 's'}
            Contains metadata relative to time.

        data_info : dict. Default value is {}.
            Contains facultative metadata relative to data. For example, the
            data_info attribute could indicate the unit of data['Forces']:

            >>> data['Forces'] = {'unit': 'N'}.

            To facilitate the management of data_info, please refer to the
            class method:

            ``ktk.TimeSeries.add_data_info``

    Example of creation
    -------------------
        >>> ts = TimeSeries({time: np.array(range(0,100))})

    """

    def __init__(self, time=np.array([]), time_info={'unit': 's'},
                 data=dict(), data_info=dict(), events=list()):
        self.time = time.copy()
        self.time_info = time_info.copy()
        self.data = data.copy()
        self.data_info = data_info.copy()
        self.events = events.copy()

    def __str__(self):
        """
        Print a textual descriptive of the TimeSeries contents.

        Returns
        -------
        str
            String that describes the contents of each attribute ot the
            TimeSeries

        """
        return _repr._format_class_attributes(self)

    def add_data_info(self, signal_name, info_name, value):
        """
        Add information on a signal of the TimeSeries.

        Examples of use
        ---------------
            >>> the_timeseries.add_info('forces', 'unit', 'N')
            >>> the_timeseries.add_info('marker1', 'color', [43, 2, 255])

        This creates a corresponding entries in the 'data_info' dict.

        """
        try:
            self.data_info[signal_name]                     # Test if it exists
            self.data_info[signal_name][info_name] = value  # Assign the value

        except:  # No info yet for this signal name
            self.data_info[signal_name] = {info_name: value}  # Assign value

        return self

    def add_event(self, time, name='event'):
        """
        Add an event to the TimeSeries.

        Parameters
        ----------
        time : float
            The time of the event, in the same unit as the time_info{'unit'}
            attribute of the TimeSeries (default: 's').
        name : str
            The name of the event.

        Returns
        -------
        self.

        Example of use
        --------------
        >>> ts = ktk.TimeSeries()
        >>> ts.add_event(5.5, 'event1')
        >>> ts.add_event(10.8, 'event2')
        >>> ts.add_event(2.3, 'event2')
        >>> print(ts.events)
        [[5.5, 'event1'], [10.8, 'event2'], [2.3, 'event2']]

        """
        self.events.append(TimeSeriesEvent(time, name))
        return self

    # def ui_add_events(self, name='event'):
    #     """
    #     Add one or many events interactively to the TimeSeries.

    #     Parameters
    #     ----------
    #     name : str
    #         The name of the event.

    #     Returns
    #     -------
    #     self.
    #     """
    #     self.plot()
    #     _gui.buttondialog(title='uiaddevents',
    #                       message=('Please zoom on the figure, then click '
    #                                + 'Continue, or End to terminate.'),
    #                       choices=['Continue', 'End'])
    #     plt.suptitle(('Left-click to add events,\n'
    #                   + 'Right-click to remove last added events,\n'
    #                   + 'Enter to terminate.'))
    #     points = plt.ginput(1000)

    #     for the_point in points:
    #         self.addevent(time=the_point[0], name=name)

    #     # TODO Continue


    # def save(self, file_name): #TODO, still not what I want.
    #     np.save(file_name, self, allow_pickle=True)


    # def load(file_name): #TODO, still not what I want.
    #     temp = np.load(file_name, allow_pickle=True)
    #     temp = temp.tolist()
    #     return(temp)

    def copy(self):
        """
        Deep copy of a TimeSeries.

        Returns
        -------
        A deep copy of the original TimeSeries.

        """
        return _deepcopy(self)

    def plot(self):
        """
        Plot the TimeSeries using matplotlib.

        Returns
        -------
        None.

        """
        the_keys = self.data.keys()
        n_plots = len(the_keys)
        n_events = len(self.events)
        event_times = np.array(self.events)[:, 0]

        # Now plot
        i_plot = 1
        for the_keys in the_keys:

            if i_plot == 1:
                ax = plt.subplot(n_plots, 1, i_plot)
            else:
                plt.subplot(n_plots, 1, i_plot, sharex=ax)

            plt.cla()

            # Plot data
            plt.plot(self.time, self.data[the_keys])

            if (the_keys in self.data_info and
                    'unit' in self.data_info[the_keys]):
                plt.ylabel(the_keys + ' (' +
                           self.data_info[the_keys]['unit'] + ')')
            else:
                plt.ylabel(the_keys)

            # Plot the events
            a = plt.axis()
            min_y = a[2]
            max_y = a[3]
            event_line_x = np.zeros(3*n_events)
            event_line_y = np.zeros(3*n_events)

            for i_event in range(0, n_events):
                event_line_x[3*i_event] = event_times[i_event]
                event_line_x[3*i_event+1] = event_times[i_event]
                event_line_x[3*i_event+2] = np.nan

                event_line_y[3*i_event] = min_y
                event_line_y[3*i_event+1] = max_y
                event_line_y[3*i_event+2] = np.nan

            plt.plot(event_line_x, event_line_y, 'c')

            # Next plot
            i_plot += 1

        # Add labels and format
        plt.xlabel('Time (' + self.time_info['unit'] + ')')
        plt.tight_layout()
        plt.show()

    def get_index_at_time(self, time):
        """
        Get the time index that is the closest to the specified time.

        Parameters
        ----------
        time : float
            Time to lookfor in the TimeSeries' time vector.

        Returns
        -------
        The index in the time vector : int.

        Examples of use
        ---------------
        >>> ts = ktk.TimeSeries(time=np.array([0, 0.5, 1, 1.5, 2]))
        >>> ts.get_index_at_time(0.9)
        2
        >>> ts.get_index_at_time(1)
        2
        >>> ts.get_index_at_time(1.1)
        2

        """
        return np.argmin(np.abs(self.time - time))

    def get_index_before_time(self, time):
        """
        Get the time index that is just before or at the specified time.

        Parameters
        ----------
        time : float
            Time to lookfor in the TimeSeries' time vector.

        Returns
        -------
        int : the index in the time vector. If no value is before the
        specified time, a value of np.nan is returned.

        Examples of use
        ---------------
        >>> ts = ktk.TimeSeries(time=np.array([0, 0.5, 1, 1.5, 2]))
        >>> ts.get_index_before_time(0.9)
        1
        >>> ts.get_index_before_time(1)
        2
        >>> ts.get_index_before_time(1.1)
        2
        >>> ts.get_index_before_time(-1)
        nan

        """
        diff = time - self.time
        diff[diff < 0] = np.nan
        if np.all(np.isnan(diff)):  # All nans
            return np.nan
        else:
            return np.nanargmin(diff)

    def get_index_after_time(self, time):
        """
        Get the time index that is just after or at the specified time.

        Parameters
        ----------
        time : float
            Time to look for in the TimeSeries' time vector.

        Returns
        -------
        int : the index in the time vector. If no value is after the
        specified time, a value of np.nan is returned.

        Examples of use
        ---------------
        >>> ts = ktk.TimeSeries(time=np.array([0, 0.5, 1, 1.5, 2]))
        >>> ts.get_index_after_time(0.9)
        2
        >>> ts.get_index_after_time(1)
        2
        >>> ts.get_index_after_time(1.1)
        3
        >>> ts.get_index_after_time(13)
        nan

        """
        diff = self.time - time
        diff[diff < 0] = np.nan
        if np.all(np.isnan(diff)):  # All nans
            return np.nan
        else:
            return np.nanargmin(diff)

    def get_event_time(self, event_name, event_occurence=1):
        """
        Get the time of the specified event.

        Parameters
        ----------
        event_name : str
            Name of the event to look for in the events list.
        event_occurence : int, optional. Default is 1.
            i_th occurence of the event to look for in the events list.

        Returns
        -------
        The time of the specified event, as a float. If no corresponding event
        is found, then np.nan is returned.

        Examples of use
        ---------------
        >>> # Instanciate a timeseries with some events
        >>> ts = ktk.TimeSeries()
        >>> ts.add_event(5.5, 'event1')
        >>> ts.add_event(10.8, 'event2')
        >>> ts.add_event(2.3, 'event2')

        >>> # Now let call ``get_event_time``
        >>> ts.get_event_time('event1')
        5.5
        >>> ts.get_event_time('event2', 1)
        2.3
        >>> ts.get_event_time('event2', 2)
        10.8

        """
        if np.round(event_occurence) != event_occurence:
            raise Warning('Rounding event-occurence to the nearest integer')
            event_occurence = np.round(event_occurence)

        if event_occurence < 1:
            raise ValueError('event_occurence must be stricly positive')

        the_event_times = np.array([x.time for x in self.events])
        the_event_indices = [(x.name == event_name) for x in self.events]

        # Keep only the events with the specified name
        the_event_times = np.array(the_event_times[the_event_indices])

        n_events = len(the_event_times)
        if n_events == 0 or event_occurence > n_events:
            return np.nan
        else:
            the_event_times = np.sort(the_event_times)
            return the_event_times[event_occurence - 1]

    def get_ts_at_time(self, time):
        """
        Get a one-data subset of the TimeSeries at the nearest time.

        Parameters
        ----------
        time : float
            Time to look for in the TimeSeries' time vector.

        Returns
        -------
        TimeSeries

        """
        out_ts = self.copy()
        index = self.get_index_at_time(time)
        out_ts.time = out_ts.time[index]
        for the_data in out_ts.data.keys():
            out_ts.data[the_data] = out_ts.data[the_data][index]
        return out_ts

    def get_ts_at_event(self, event_name, event_occurence=1):
        """
        Get a one-data subset of the TimeSeries at the event's nearest time.

        Parameters
        ----------
        event_name : str
            Name of the event to look for in the events list.
        event_occurence : int, optional. Default is 1.
            i_th occurence of the event to look for in the events list.

        Returns
        -------
        TimeSeries : A one-data subset of the TimeSeries at the event's
        nearest time.

        """
        time = self.get_event_time(event_name, event_occurence)
        return self.get_ts_at_time(time)

    def get_ts_before_time(self, time):
        """
        Get a subset of the TimeSeries before and at the specified time.

        Parameters
        ----------
        time : float
            Time to look for in the TimeSeries' time vector.

        Returns
        -------
        TimeSeries

        Examples of use
        ---------------
        >>> ts = ktk.TimeSeries(time=np.linspace(0, 9, 10))
        >>> new_ts = ts.get_ts_before_time(3)
        >>> print(new_ts.time)
        [0., 1., 2., 3.]
        >>> new_ts = ts.get_ts_before_time(3.5)
        >>> print(new_ts.time)
        [0., 1., 2., 3.]
        >>> new_ts = ts.get_ts_before_time(-2)
        >>> print(new_ts.time)
        []
        >>> new_ts = ts.get_ts_before_time(13)
        >>> print(new_ts.time)
        [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]

        """
        out_ts = self.copy()
        index = self.get_index_before_time(time)
        if np.isnan(index):
            index_range = []
        else:
            index_range = range(0, index+1)

        out_ts.time = out_ts.time[index_range]
        for the_data in out_ts.data.keys():
            out_ts.data[the_data] = out_ts.data[the_data][index_range]
        return out_ts

    def get_ts_before_event(self, event_name, event_occurence=1):
        """
        Get a subset of the TimeSeries before and at the specified event.

        Parameters
        ----------
        event_name : str
            Name of the event to look for in the events list.
        event_occurence : int, optional. Default is 1.
            i_th occurence of the event to look for in the events list.

        Returns
        -------
        TimeSeries.

        """
        time = self.get_event_time(event_name, event_occurence)
        return self.get_ts_before_time(time)

    def get_ts_after_time(self, time):
        """
        Get a subset of the TimeSeries after and at the specified time.

        Parameters
        ----------
        time : float
            Time to look for in the TimeSeries' time vector.

        Returns
        -------
        TimeSeries

        Examples of use
        ---------------
        >>> ts = ktk.TimeSeries(time=np.linspace(0, 9, 10))
        >>> new_ts = ts.get_ts_after_time(3)
        >>> print(new_ts.time)
        [3., 4., 5., 6., 7., 8., 9.]
        >>> new_ts = ts.get_ts_after_time(3.5)
        >>> print(new_ts.time)
        [4., 5., 6., 7., 8., 9.]
        >>> new_ts = ts.get_ts_after_time(-2)
        >>> print(new_ts.time)
        [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]
        >>> new_ts = ts.get_ts_after_time(13)
        >>> print(new_ts.time)
        []

        """
        out_ts = self.copy()
        index = self.get_index_after_time(time)
        if np.isnan(index):
            index_range = []
        else:
            index_range = range(index, len(self.time))

        out_ts.time = out_ts.time[index_range]
        for the_data in out_ts.data.keys():
            out_ts.data[the_data] = out_ts.data[the_data][index_range]
        return out_ts

    def get_ts_after_event(self, event_name, event_occurence=1):
        """
        Get a subset of the TimeSeries after and at the specified event.

        Parameters
        ----------
        event_name : str
            Name of the event to look for in the events list.
        event_occurence : int, optional. Default is 1.
            i_th occurence of the event to look for in the events list.

        Returns
        -------
        TimeSeries.

        """
        time = self.get_event_time(event_name, event_occurence)
        return self.get_ts_after_time(time)

    def get_ts_between_times(self, time1, time2):
        """
        Get a subset of the TimeSeries between two specified times.

        Parameters
        ----------
        time1, time2 : float
            Times to look for in the TimeSeries' time vector.

        Returns
        -------
        TimeSeries

        Examples of use
        ---------------
        >>> ts = ktk.TimeSeries(time=np.linspace(0, 9, 10))
        >>> new_ts = ts.get_ts_between_times(3, 6)
        >>> print(new_ts.time)
        [3., 4., 5., 6.]
        >>> new_ts = ts.get_ts_between_time(3.5, 5.5)
        >>> print(new_ts.time)
        [4., 5.]
        >>> new_ts = ts.get_ts_between_times(-2, 13)
        >>> print(new_ts.time)
        [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]
        >>> new_ts = ts.get_ts_between_times(-2, -1)
        >>> print(new_ts.time)
        []

        """
        sorted_times = np.sort([time1, time2])
        new_ts = self.get_ts_after_time(sorted_times[0])
        new_ts = new_ts.get_ts_before_time(sorted_times[1])
        return new_ts

    def get_ts_between_events(self, event_name1, event_name2,
                              event_occurence1=1, event_occurence2=1):
        """
        Get a subset of the TimeSeries between two specified events.

        Parameters
        ----------
        event_name1, event_name2 : str
            Name of the events to look for in the events list.
        event_occurence1, event_occurence2 : int, optional. Default is 1.
            i_th occurence of the events to look for in the events list.

        Returns
        -------
        TimeSeries

        """
        time1 = self.get_event_time(event_name1, event_occurence1)
        time2 = self.get_event_time(event_name2, event_occurence2)
        return self.get_ts_between_times(time1, time2)
