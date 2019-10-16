"""
Module that manages the TimeSeries class.

Author: Felix Chenier
Date: July 2019
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from copy import deepcopy

from . import gui
from . import _repr


class TimeSeriesEvent(list):
    """
    Define an event in a timeseries.

    This class derives from the list class. A TimeSeriesEvent is always a
    two-items list with the first item being the time and the second item
    being the name of the event.

    Dependent properties time and name are added for convenience.

    Properties
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

        time_info : dict. Default value is {'Unit': 's'}
            Contains metadata relative to time.

        data_info : dict. Default value is {}.
            Contains facultative metadata relative to data. For example, the
            data_info attribute could indicate the unit of data['Forces']:

            >>> data['Forces'] = {'Unit': 'N'}.

            To facilitate the management of data_info, please refer to the
            class method:

            ``ktk.TimeSeries.add_data_info``

    Example of creation
    -------------------
        >>> ts = ktk.TimeSeries(time=np.arange(0,100))

    """

    def __init__(self, time=np.array([]), time_info={'Unit': 's'},
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

    def __repr__(self):
        return _repr._format_class_attributes(self)

    def __eq__(self, ts):
        """
        Compare two timeseries for equality.

        Returns
        -------
        True if each attribute of ts is equal to the TimeSeries' attributes.

        """
        if not np.array_equal(self.time, ts.time):
            print('Time is not equal')
            return False

        for one_data in self.data:
            if not np.isclose(self.data[one_data], ts.data[one_data],
                              rtol=1e-15).all():
                print('%s is not equal' % one_data)
                return False

        for one_data in ts.data:
            if not np.isclose(self.data[one_data], ts.data[one_data],
                              rtol=1e-15).all():
                print('%s is not equal' % one_data)
                return False

        if self.time_info != ts.time_info:
            print('time_info is not equal')
            return False

        if self.data_info != ts.data_info:
            print('data_info is not equal')
            return False

        if self.events != ts.events:
            print('events is not equal')
            return False

        return True

    def add_data_info(self, signal_name, info_name, value):
        """
        Add information on a signal of the TimeSeries.

        Parameters
        ----------
        signel_name : str
            The data key the info corresponds to.
        info_name : str
            The name of the info.
        value : any type
            The info.

        Returns
        -------
        None

        Examples
        --------
            >>> the_timeseries.add_info('Forces', 'Unit', 'N')
            >>> the_timeseries.add_info('Marker1', 'Color', [43, 2, 255])

        This creates a corresponding entries in the 'data_info' dict.

        """
        if signal_name in self.data_info:   # Assign the value
            self.data_info[signal_name][info_name] = value
        else: #  Create and assign value
            self.data_info[signal_name] = {info_name: value}

    def add_event(self, time, name='event'):
        """
        Add an event to the TimeSeries.

        Parameters
        ----------
        time : float
            The time of the event, in the same unit as the time_info{'Unit'}
            attribute of the TimeSeries (default: 's').
        name : str (optional)
            The name of the event.

        Returns
        -------
        None.

        Example
        -------
        >>> ts = ktk.TimeSeries()
        >>> ts.add_event(5.5, 'event1')
        >>> ts.add_event(10.8, 'event2')
        >>> ts.add_event(2.3, 'event2')
        >>> print(ts.events)
        [[5.5, 'event1'], [10.8, 'event2'], [2.3, 'event2']]

        """
        self.events.append(TimeSeriesEvent(time, name))

    def ui_add_event(self, name='event', plot=[], multiple_events=False):
        """
        Add one or many events interactively to the TimeSeries.

        Parameters
        ----------
        name : str (optional)
            The name of the event.
        plot : str, list of str or tuple of str (optional)
            A signal name of list of signal name to be plotted, similar to
            the argument of ktk.TimeSeries.plot().
        multiple_events : bool (optional)
            - True to add multiple events with the same name.
            - False to add only one event (default).

        Returns
        -------
        None.
        """
        ts = self.copy()

        fig = plt.figure()
        plt.cla()
        ts.plot(plot)

        finished = False

        while finished is False:
            finished = True  # Only one pass by default

            button = gui.button_dialog(
                    f'Adding the event "{name}"\n'
                    'Please zoom on the new event to add, then click Next.',
                    ['Cancel', 'Next'])

            if button <= 0:  # Cancel
                plt.close(fig)
                print('No event was added.')
                return self.copy()

            if multiple_events:
                gui.message('Please left-click to add events, '
                            'right-click to delete, '
                            'ENTER to finish.')
                plt.pause(0.001)  # Update the plot
                coordinates = plt.ginput(99999)
                gui.message()

            else:
                gui.message('Please left-click on the event to add.')
                coordinates = plt.ginput(1)
                gui.message()

            # Add these events
            for i in range(len(coordinates)):
                ts.add_event(coordinates[i][0], name)

            if multiple_events:
                plt.cla()
                ts.plot(plot)
                button = gui.button_dialog(
                        f'Adding the event "{name}"\n'
                        'Do you want to add more of these events?',
                        ['Cancel', 'Add more', 'Finished'])
                if button <= 0:  # Cancel
                    plt.close(fig)
                    print('No event was added.')
                    return self.copy()
                elif button == 1:
                    finished = False
                elif button == 2:
                    finished = True

        gui.message()
        plt.close(fig)
        self.events = ts.events  # Add the events to self.

    def copy(self):
        """
        Deep copy of a TimeSeries.

        Returns
        -------
        A deep copy of the original TimeSeries.

        """
        return deepcopy(self)

    def plot(self, data_keys=None, plot_event_names=False):
        """
        Plot the TimeSeries using matplotlib.

        Parameters
        ----------
        data_keys : string, list or tuple (optional)
            String or list of strings corresponding to the signals to plot.
            For example, if a TimeSeries's data attribute as keys 'Forces',
            'Moments' and 'Angle', then:
            >>> the_timeseries.plot(['Forces', 'Moments'])
            plots only the forces and moments, without plotting the angle.
            By default, all elements of the TimeSeries are plotted.

        plot_event_names : bool (optional)
            True to plot the event names on top of the event lines.
            Default = False.

        Returns
        -------
        None.

        """
        if data_keys is None or len(data_keys) == 0:
            # Plot all
            the_keys = self.data.keys()
        else:
            # Plot only what is asked for.
            if isinstance(data_keys, list) or isinstance(data_keys, tuple):
                the_keys = data_keys
            elif isinstance(data_keys, str):
                the_keys = [data_keys]
            else:
                raise(TypeError(
                        'data_keys must be a string or list of strings'))

        n_plots = len(the_keys)

        n_events = len(self.events)
        if n_events > 0:
            event_times = np.array(self.events)[:, 0]
        else:
            event_times = np.array([])

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
                    'Unit' in self.data_info[the_keys]):
                plt.ylabel(the_keys + ' (' +
                           self.data_info[the_keys]['Unit'] + ')')
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

            if plot_event_names:
                for event in self.events:
                    plt.text(event.time, max_y, event.name,
                             rotation='vertical',
                             horizontalalignment='center')

            # Next plot
            i_plot += 1

        # Add labels and format
        plt.xlabel('Time (' + self.time_info['Unit'] + ')')
        # plt.tight_layout(pad=0.05)
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

    def ui_get_ts_between_clicks(self, data_keys=None):
        """
        Get a subset of the TimeSeries between two mouse clicks.

        Parameters
        ----------
        data_keys : string, list or tuple (optional)
            String or list of strings corresponding to the signals to plot.
            See TimeSeries.plot() for more information.

        Returns
        -------
        TimeSeries

        """
        fig = plt.figure()
        self.plot(data_keys)
        plt.pause(0.001)  # Redraw
        gui.message('Click on the left of the desired zone.')
        left_point = plt.ginput(1)
        gui.message('')
        gui.message('Click on the right of the desired zone.')
        right_point = plt.ginput(1)
        gui.message('')
        plt.close(fig)
        return self.get_ts_between_times(left_point[0][0], right_point[0][0])

    def isnan(self, data_key):
        """
        Return a boolean array of missing samples.

        Parameters
        ----------
        data_key : str
            Key value of the data signal to analyze.

        Returns
        -------
        A boolean array of the same size as the time vector, where True values
        represent missing samples (samples that contain at least one NaN
        value).
        """
        values = self.data[data_key].copy()
        # Reduce the dimension of values while keeping the time dimension.
        while len(np.shape(values)) > 1:
            values = np.sum(values, 1)
        return np.isnan(values)

    def ui_sync(self, data_keys=None):
        """
        Synchronize a TimeSeries by setting its zero-time interactively.

        Parameters
        ----------
        data_keys : str or list of str (optional)
            The data keys to plot. Default is None, which means that all data
            is plotted.

        Returns
        -------
        None.
        """
        fig = plt.figure()
        self.plot(data_keys)
        choice = gui.button_dialog(
                'Please zoom on the sync event and press Next.',
                ['Cancel', 'Next'])
        if choice != 1:
            return

        gui.message('Click on the sync event.')
        click = plt.ginput(1)
        gui.message(None)
        plt.close(fig)

        time = click[0][0]

        for event in self.events:
            event.time -= time

        self.time -= time

    def get_subset(self, data_keys):
        """
        Return a subset of the TimeSeries.

        This method returns a TimeSeries that contains only specific data
        keys. For example, if a TimeSeries ts has the fields Forces, Moments
        and Angle, then:
            >>> ts.get_subset(['Forces', 'Moments'])
        returns an identical timeseries, but without the data key 'Angle'.

        The corresponding data_info keys are copied in the new TimeSeries.
        All events are also copied in the new TimeSeries.

        Parameters
        ----------
        data_keys : str or list of str
            The data keys to extract from the timeseries.

        Returns
        -------
        A copy of the TimeSeries, minus the unspecified data keys.
        """
        if isinstance(data_keys, str):
            data_keys = [data_keys]

        ts = TimeSeries()
        ts.time = self.time.copy()
        ts.time_info = deepcopy(self.time_info)
        ts.events = deepcopy(self.events)

        for key in data_keys:
            if key in self.data:
                ts.data[key] = self.data[key].copy()
            if key in self.data_info:
                ts.data_info[key] = deepcopy(self.data_info[key])

        return ts

    def merge(self, ts, data_keys=None, interp_kind=None, fill_value=None,
              overwrite=True):
        """
        Merge another TimeSeries into the current TimeSeries.

        This method merges a TimeSeries into the current TimeSeries, copying
        the data, data_info and events.

        Parameters
        ----------
        ts : TimeSeries
            The TimeSeries to merge into the current TimeSeries.
        data_keys : str or list of str (optional)
            The data keys to merge from ts. Default is None, which means that
            all the data keys are merged.
        interp_kind : str (optional)
            The interpolation method, if necessary. Default is None.
            If both TimeSeries' time vectors differ and interp_kind is None,
            then an exception is raised.
            If both TimeSeries' time vectors differ and interp_kind is
            specified, then ts is resampled to the current TimeSeries' time
            vector before merging.
            interp_kind may take any value that is supported by
            scipy.interpolate.interp1d:
                - 'linear'
                - 'nearest'
                - 'zero'
                - 'slinear'
                - 'quadratic'
                - 'cubic'
                - 'previous'
                - 'next'
        fill_value : array-like or 'extrapolate' (optional)
            The fill value to use if ts' time vector contains point outside
            the current TimeSeries' time vector. Use 'extrapolate' to
            extrapolate.
        overwrite : bool (optional)
            If duplicates are found and overwrite is True, then the source (ts)
            overwrites the destination. Otherwise (overwrite is False), the
            duplicated data is ignored. Default is True.

        Returns
        -------
        None
        """
        if data_keys is None or len(data_keys) == 0:
            data_keys = ts.data.keys()
        else:
            if isinstance(data_keys, list) or isinstance(data_keys, tuple):
                pass
            elif isinstance(data_keys, str):
                data_keys = [data_keys]
            else:
                raise(TypeError(
                        'data_keys must be a string or list of strings'))

        # Check if resampling is needed
        if ((self.time.shape == ts.time.shape) and
                np.all(self.time == ts.time)):
            must_resample = False
        else:
            must_resample = True

        if must_resample and interp_kind is None:
            raise(ValueError(
                    'Time vectors do not match, resampling is required.'))

        for key in data_keys:

            # Check if this key is a duplicate, then continue to next key if
            # required.
            if (key in self.data) and (key in ts.data) and overwrite is False:
                continue

            # Resample if needed
            if must_resample:
                index = ~ts.isnan(key)
                index
                if ~np.all(index):
                    print('Warning: Some NaNs found and interpolated.')

                f = sp.interpolate.interp1d(ts.time[index],
                                            ts.data[key][index],
                                            axis=0, fill_value=fill_value,
                                            kind=interp_kind)
                data = f(self.time)
            else:
                data = ts.data[key]

            # Add this data
            self.data[key] = data

            if key in ts.data_info:
                for info_name in ts.data_info[key].keys():
                    self.add_data_info(key, info_name,
                                       ts.data_info[key][info_name])

        # Merge events
        for event in ts.events:
            self.events.append(event)
