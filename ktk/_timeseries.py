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

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"

"""
_timeseries.py
==============

Module that provides the TimeSeries and TimeSeriesEvent classes. The classes
defined in this module are loaded into the toplevel ktk namespace.

"""

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import warnings

from copy import deepcopy
import ktk._repr
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval


# Helper functions
def _dict_of_arrays_to_dataframe(dict_of_arrays):
    """
    Convert a numpy ndarray of any dimension to a pandas DataFrame.

    Parameters
    ----------
    dict_of_array : dict
        A dict that contains numpy arrays. Each array must have the same
        first dimension's size.

    Returns
    -------
    DataFrame

    The rows in the output DataFrame correspond to the first dimension of the
    numpy arrays.
    - Vectors are converted to single-column DataFrames.
    - 2-dimensional arrays are converted to multi-columns DataFrames.
    - 3-dimensional (or more) arrays are also converted to DataFrames, but
      indices in brackets are added to the column names.

    """
    # Init
    df_out = pd.DataFrame()

    # Go through data
    the_keys = dict_of_arrays.keys()
    for the_key in the_keys:

        # Assign data
        original_data = dict_of_arrays[the_key]
        original_data_shape = np.shape(original_data)
        data_length = np.shape(original_data)[0]

        reshaped_data = np.reshape(original_data, (data_length, -1))
        reshaped_data_shape = np.shape(reshaped_data)

        df_data = pd.DataFrame(reshaped_data)

        # Get the column names index from the shape of the original data
        # The strategy here is to build arrays of indices, that have
        # the same shape as the original data, then reshape these matrices
        # the same way we reshaped the original data. Then we know where
        # the original indices are in the new reshaped data.
        original_indices = np.indices(original_data_shape[1:])
        reshaped_indices = np.reshape(original_indices,
                                      (-1, reshaped_data_shape[1]))

        # Hint for my future self:
        # For a one-dimension series, reshaped_indices will be:
        # [[0]].
        # For a two-dimension series, reshaped_indices will be:
        # [[0 1 2 ...]].
        # For a three-dimension series, reshaped_indices will be:
        # [[0 0 0 ... 1 1 1 ... 2 2 2 ...]
        #   0 1 2 ... 0 1 2 ... 0 1 2 ...]]
        # and so on.

        # Assign column names
        column_names = []
        for i_column in range(0, len(df_data.columns)):
            this_column_name = the_key
            n_indices = np.shape(reshaped_indices)[0]
            if n_indices > 0:
                # This data is expressed in more than one dimension.
                # We must add brackets to the column names to specify
                # the indices.
                this_column_name += '['

                for i_indice in range(0, n_indices):
                    this_column_name += str(
                        reshaped_indices[i_indice, i_column])
                    if i_indice == n_indices - 1:
                        this_column_name += ']'
                    else:
                        this_column_name += ','

            column_names.append(this_column_name)

        df_data.columns = column_names

        # Merge this dataframe with the output dataframe
        df_out = pd.concat([df_out, df_data], axis=1)

    return df_out


def _dataframe_to_dict_of_arrays(dataframe):
    """
    Convert a pandas DataFrame to a dict of numpy ndarrays.

    Parameters
    ----------
    pd_dataframe : pd.DataFrame
        The dataframe to be converted.

    Returns
    -------
    dict of ndarrays.

    If all the dataframe columns have the same name but with different indices
    in brackets, then the dataframe corresponds to a single array, which is
    returned.

    If the dataframe contains different column names (for example,
    Forces[0], Forces[1], Forces[2], Moments[0], Moments[1], Moments[2]), then
    a dict of arrays is returned. In this case, this dict would have the keys
    'Forces' and 'Moments', which would each contain an array.

    This function mirrors the dict_of_arrays_to_dataframe function. Its use is
    mainly to convert high-dimension (>2) dataframes to high-dimension (>2)
    arrays.
    """
    # Init output
    out = dict()

    # Search for the column names and highest dimensions
    all_column_names = dataframe.columns
    all_data_names = []
    all_data_highest_indices = []
    length = len(dataframe)

    for one_column_name in all_column_names:
        opening_bracket_position = one_column_name.find('[')
        if opening_bracket_position == -1:
            # No dimension for this data
            all_data_names.append(one_column_name)
            all_data_highest_indices.append([length - 1])
        else:
            # Extract name and dimension
            data_name = one_column_name[0:opening_bracket_position]
            data_dimension = literal_eval(
                '[' + str(length - 1) + ',' +
                one_column_name[opening_bracket_position + 1:])

            all_data_names.append(data_name)
            all_data_highest_indices.append(data_dimension)

    # Create a set of unique_data_names
    unique_data_names = []
    for data_name in all_data_names:
        if data_name not in unique_data_names:
            unique_data_names.append(data_name)

    for unique_data_name in unique_data_names:

        # Create a Pandas DataFrame with only the columns that match
        # this unique data name. In the same time, check the final
        # dimension of the data to know to which dimension we will
        # reshape the DataFrame's data.
        sub_dataframe = pd.DataFrame()
        unique_data_highest_index = []
        for i in range(0, len(all_data_names)):
            if all_data_names[i] == unique_data_name:
                sub_dataframe[all_column_names[i]] = (
                    dataframe[all_column_names[i]])
                unique_data_highest_index.append(
                    all_data_highest_indices[i])

        # Sort the sub-dataframe's columns
        sub_dataframe.reindex(sorted(sub_dataframe.columns), axis=1)

        # Calculate the data dimension we must reshape to
        unique_data_dimension = np.max(
            np.array(unique_data_highest_index) + 1, axis=0)

        # Convert the dataframe to a np.array, then reshape.
        new_data = sub_dataframe.to_numpy()
        new_data = np.reshape(new_data, unique_data_dimension)
        out[unique_data_name] = new_data

    return out


class TimeSeriesEvent(list):
    """
    Define an event in a timeseries.

    This class derives from the list class. A TimeSeriesEvent is always a
    two-items list with the first item being the time and the second item
    being the name of the event.

    The dependent properties `time` and `name` can be used both in read and
    write for convenience.

    This class is rarely used by itself, it is easier to use the `TimeSeries``
    methods to deal with events.

    Properties
    ----------
    time : float
        The time at which the event happened.
    name : str
        The name of the event.

    Example
    -------
        >>> event = ktk.TimeSeriesEvent()
        >>> event.time = 1.5
        >>> event.name = 'event_name'
        >>> event
        [1.5, 'event_name']

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
        self[1] = str(name)


class TimeSeries():
    """
    A class that implements TimeSeries.

    This class implements a Timeseries in a way that resembles the timeseries
    and tscollection found in Matlab.

    Attributes
    ----------
    time : 1-dimension np.array (optional)
        Contains the time vector. The default is [].

    data : dict (optional)
        Contains the data, where each element contains a np.array which
        first dimension corresponds to time. The default is {}.

    time_info : dict (optional)
        Contains metadata relative to time. The default is {'Unit': 's'}

    data_info : dict (optional)
        Contains facultative metadata relative to data. For example, the
        data_info attribute could indicate the unit of data['Forces']:

        data['Forces'] = {'Unit': 'N'}.

        To facilitate the management of data_info, please use
        `ktk.TimeSeries.add_data_info`.

        The default is {}.

    Example
    -------
        >>> ts = ktk.TimeSeries(time=np.arange(0,100))

    """

    def __init__(self, time=np.array([]), time_info={'Unit': 's'},
                 data=dict(), data_info=dict(), events=list(),
                 from_dataframe=None):

        self.time = time.copy()
        self.data = data.copy()
        self.time_info = time_info.copy()
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
        return ktk._repr._format_class_attributes(self)

    def __repr__(self):
        return ktk._repr._format_class_attributes(self)

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

    def to_dataframe(self):
        """
        Create a DataFrame by reshaping all data to one bidimensional table.

        Parameters
        ----------
        None.

        Returns
        -------
        df : DataFrame
            DataFrame with the index as the TimeSeries' time. Vector data are
            converted to single columns, and 2-dimensional (or more) data are
            converted to multiple columns with the additional dimensions in
            brackets. The TimeSeries's events and metadata such as time_info
            and data_info are not included in the resulting DataFrame.

        """
        df = _dict_of_arrays_to_dataframe(self.data)
        df.index = self.time
        return df

    def from_dataframe(self, dataframe):
        """
        Load time and data from a DataFrame.

        The current TimeSeries' time and data properties are overwritten.

        Parameters
        ----------
        dataframe : DataFrame
            A Pandas DataFrame where the index corresponds to time, and
            where each column corresponds to a data key. As special cases,
            data in column which names end with bracketed indices such as
            [0], [1], [0,0], [0,1], etc. are converted to multidimensional
            arrays. For example, if a DataFrame has these column names:

                Forces[0], Forces[1], Forces[2], Forces[3]

            then a single data key is created (Forces) and the data itself
            will be of shape Nx4, N being the number of samples (the length
            of the DataFrame).

        Returns
        -------
        self.

        """
        self.data = _dataframe_to_dict_of_arrays(dataframe)
        self.time = dataframe.index.to_numpy()
        return self

    def add_data_info(self, signal_name, info_name, value):
        """
        Add metadata to TimeSeries' data.

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
        None.

        Example
        -------
            >>> ts = ktk.TimeSeries()
            >>> ts.add_data_info('Forces', 'Unit', 'N')
            >>> ts.add_data_info('Marker1', 'Color', [43, 2, 255])

            >>> ts.data_info['Forces']
            {'Unit': 'N'}

            >>> ts.data_info['Marker1']
            {'Color': [43, 2, 255]}

        """
        if signal_name in self.data_info:   # Assign the value
            self.data_info[signal_name][info_name] = value
        else:  # Create and assign value
            self.data_info[signal_name] = {info_name: value}

    def rename_data(self, old_data_field, new_data_field):
        """
        Rename a key in data and data_info.

        Parameters
        ----------
        old_data_field : str
            Name of the data key.
        new_data_field : str
            New name of the data key.

        Returns
        -------
        None.

        Example
        -------
            >>> ts = ktk.TimeSeries()
            >>> ts.data['test'] = np.arange(10)
            >>> ts.add_data_info('test', 'Unit', 'm')

            >>> ts.data
            {'test': array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}

            >>> ts.data_info
            {'test': {'Unit': 'm'}}

            >>> ts.rename_data('test', 'signal')

            >>> ts.data
            {'signal': array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}

            >>> ts.data_info
            {'signal': {'Unit': 'm'}}

        """
        if old_data_field in self.data:
            self.data[new_data_field] = self.data.pop(old_data_field)
        if old_data_field in self.data_info:
            self.data_info[new_data_field] = self.data_info.pop(old_data_field)

    def add_event(self, time, name='event'):
        """
        Add an event to the TimeSeries.

        Parameters
        ----------
        time : float
            The time of the event, in the same unit as `time_info['Unit']`
            (default: 's').
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

            >>> ts.events
            [[2.3, 'event2'], [5.5, 'event1'], [10.8, 'event2']]

        """
        self.events.append(TimeSeriesEvent(time, name))
        self._sort_events()

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
        status : boolean
            - True if the event was added;
            - False if the operation was cancelled by the user.
        """
        ts = self.copy()

        fig = plt.figure()
        ts.plot(plot)

        finished = False

        while finished is False:
            finished = True  # Only one pass by default

            button = ktk.mplhelper.button_dialog(
                f'Adding the event "{name}".\n'
                'Please zoom on the location to \n'
                'add the event, then click Next.',
                ['Cancel', 'Next'])

            if button <= 0:  # Cancel
                plt.close(fig)
                return False

            if multiple_events:
                ktk.mplhelper.message(
                    'Left-click to add events; \n'
                    'Right-click to delete; \n'
                    'ENTER to finish.')
                plt.pause(0.001)  # Update the plot
                coordinates = plt.ginput(99999)
                ktk.mplhelper.message('')

            else:
                ktk.mplhelper.message(
                    'Please left-click on the event to add.')
                coordinates = plt.ginput(1)
                ktk.mplhelper.message('')

            # Add these events
            for i in range(len(coordinates)):
                ts.add_event(coordinates[i][0], name)

            if multiple_events:
                plt.cla()
                ts.plot(plot)
                button = ktk.mplhelper.button_dialog(
                    f'Adding the event "{name}".\n'
                    'Do you want to add more of these events?',
                    ['Cancel', 'Add more', "Finished"])
                if button <= 0:  # Cancel
                    plt.close(fig)
                    return False
                elif button == 1:
                    finished = False
                elif button == 2:
                    finished = True

        ktk.mplhelper.message('')
        plt.close(fig)
        self.events = ts.events  # Add the events to self.
        self.sort_events()
        return True

    def _sort_events(self):
        """
        Sorts the TimeSeries' events and ensure that all events are unique.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        self.events = sorted(self.events)
        for i in range(len(self.events) - 1, 0, -1):
            if ((self.events[i].time == self.events[i - 1].time) and
                    (self.events[i].name == self.events[i - 1].name)):
                self.events.pop(i)

    def copy(self):
        """
        Deep copy of a TimeSeries.

        Returns
        -------
        A deep copy of the original TimeSeries.

        """
        return deepcopy(self)

    def plot(self, data_keys=None, plot_event_names=False,
             max_legend_items=5, **kwargs):
        """
        Plot the TimeSeries using matplotlib.

        Parameters
        ----------
        data_keys : string, list or tuple (optional)
            String or list of strings corresponding to the signals to plot.
            By default, all elements of the TimeSeries are plotted.
        plot_event_names : bool (optional)
            True to plot the event names on top of the event lines.
            The default is False.
        max_legend_items : int (optional)
            Maximal number of legend items, including the 'events' entry. If
            there are more items in the legend, then the legend is not shown
            for space and performance considerations.

        Additional keyboard arguments are passed to the pyplot's plot function.

        Returns
        -------
        None.

        Example
        -------
        If a TimeSeries `ts`' data attribute as keys 'Forces', 'Moments' and
        'Angle', then:

            ts.plot(['Forces', 'Moments'])

        plots only the forces and moments, without plotting the angle.


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
        ax = plt.gca()
        for the_key in the_keys:

            # Set label
            label = the_key
            if (the_key in self.data_info and
                    'Unit' in self.data_info[the_key]):
                label += ' (' + self.data_info[the_key]['Unit'] + ')'

            # Plot data
            ax.plot(self.time, self.data[the_key], label=label, **kwargs)

        # Plot the events
        if len(self.events) > 0:
            a = ax.axis()
            min_y = a[2]
            max_y = a[3]
            event_line_x = np.zeros(3 * n_events)
            event_line_y = np.zeros(3 * n_events)

            for i_event in range(0, n_events):
                event_line_x[3 * i_event] = event_times[i_event]
                event_line_x[3 * i_event + 1] = event_times[i_event]
                event_line_x[3 * i_event + 2] = np.nan

                event_line_y[3 * i_event] = min_y
                event_line_y[3 * i_event + 1] = max_y
                event_line_y[3 * i_event + 2] = np.nan

            ax.plot(event_line_x, event_line_y, label='events')

            if plot_event_names:
                for event in self.events:
                    ax.text(event.time, max_y, event.name,
                            rotation='vertical',
                            horizontalalignment='center')

        # Add labels
        ax.set_xlabel('Time (' + self.time_info['Unit'] + ')')

        # Add legend if required
        if len(the_keys) > 1 or len(self.events) > 0:
            if len(the_keys) <= max_legend_items:
                ax.legend()
        else:  # Only one data, plot it on the y axis.
            ax.set_ylabel(label)

    def get_index_at_time(self, time):
        """
        Get the time index that is the closest to the specified time.

        Parameters
        ----------
        time : float
            Time to look for in the TimeSeries' time vector.

        Returns
        -------
        index : int
            The index in the time vector.

        Example
        -------
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
            Time to look for in the TimeSeries' time vector.

        Returns
        -------
        index : int
            The index in the time vector. If no value is before the specified
            time, a value of np.nan is returned.

        Example
        -------
            >>> ts = ktk.TimeSeries(time=np.array([0, 0.5, 1, 1.5, 2]))

            >>> ts.get_index_before_time(0.9) == 1
            True

            >>> ts.get_index_before_time(1) == 2
            True

            >>> ts.get_index_before_time(1.1) == 2
            True

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
        index : int
            The index in the time vector. If no value is after the
            specified time, a value of np.nan is returned.

        Example
        -------
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

    def get_event_time(self, event_name, event_occurrence=0):
        """
        Get the time of the specified event.

        Parameters
        ----------
        event_name : str
            Name of the event to look for in the events list.
        event_occurrence : int (optional)
            i_th occurence of the event to look for in the events list,
            starting at 0. The default is 0.

        Returns
        -------
        event_time : float
            The time of the specified event, as a float. If no corresponding
            event is found, then np.nan is returned.

        Example
        -------
            >>> # Instanciate a timeseries with some events
            >>> ts = ktk.TimeSeries()
            >>> ts.add_event(5.5, 'event1')
            >>> ts.add_event(10.8, 'event2')
            >>> ts.add_event(2.3, 'event2')

            >>> ts.get_event_time('event1')
            5.5

            >>> ts.get_event_time('event2', 0)
            2.3

            >>> ts.get_event_time('event2', 1)
            10.8

        """
        event_occurrence = int(event_occurrence)

        if event_occurrence < 0:
            raise ValueError('event_occurrence must be positive')

        the_event_times = np.array([x.time for x in self.events])
        the_event_indices = [(x.name == event_name) for x in self.events]

        # Keep only the events with the specified name
        the_event_times = np.array(the_event_times[the_event_indices])

        n_events = len(the_event_times)
        if n_events == 0 or event_occurrence >= n_events:
            return np.nan
        else:
            the_event_times = np.sort(the_event_times)
            return the_event_times[event_occurrence]

    def get_ts_at_time(self, time):
        """
        Get a one-data subset of the TimeSeries at the nearest time.

        Parameters
        ----------
        time : float
            Time to look for in the TimeSeries' time vector.

        Returns
        -------
        ts : TimeSeries
            A TimeSeries of length 1, at the time neasest to the specified
            time.

        """
        out_ts = self.copy()
        index = self.get_index_at_time(time)
        out_ts.time = out_ts.time[index]
        for the_data in out_ts.data.keys():
            out_ts.data[the_data] = out_ts.data[the_data][index]
        return out_ts

    def get_ts_at_event(self, event_name, event_occurrence=0):
        """
        Get a one-data subset of the TimeSeries at the event's nearest time.

        Parameters
        ----------
        event_name : str
            Name of the event to look for in the events list.
        event_occurrence : int (optional)
            i_th occurence of the event to look for in the events list,
            starting at 0. The default is 0.

        Returns
        -------
        ts : TimeSeries
            A TimeSeries of length 1, at the event's nearest time.

        """
        time = self.get_event_time(event_name, event_occurrence)
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
        ts : TimeSeries
            A new TimeSeries following the specification.

        """
        out_ts = self.copy()
        index = self.get_index_before_time(time)
        if np.isnan(index):
            index_range = []
        else:
            index_range = range(0, index + 1)

        out_ts.time = out_ts.time[index_range]
        for the_data in out_ts.data.keys():
            out_ts.data[the_data] = out_ts.data[the_data][index_range]
        return out_ts

    def get_ts_before_event(self, event_name, event_occurrence=0):
        """
        Get a subset of the TimeSeries before and at the specified event.

        Parameters
        ----------
        event_name : str
            Name of the event to look for in the events list.
        event_occurrence : int (optional)
            i_th occurence of the event to look for in the events list,
            starting at 0. The default is 0.

        Returns
        -------
        ts : TimeSeries
            A new TimeSeries following the specification.

        """
        time = self.get_event_time(event_name, event_occurrence)
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
        ts : TimeSeries
            A new TimeSeries following the specification.

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

    def get_ts_after_event(self, event_name, event_occurrence=0):
        """
        Get a subset of the TimeSeries after and at the specified event.

        Parameters
        ----------
        event_name : str
            Name of the event to look for in the events list.
        event_occurrence : int (optional)
            i_th occurence of the event to look for in the events list,
            starting at 0. The default is 0.

        Returns
        -------
        ts : TimeSeries
            A new TimeSeries following the specification.

        """
        time = self.get_event_time(event_name, event_occurrence)
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
        ts : TimeSeries
            A new TimeSeries following the specification.

        """
        sorted_times = np.sort([time1, time2])
        new_ts = self.get_ts_after_time(sorted_times[0])
        new_ts = new_ts.get_ts_before_time(sorted_times[1])
        return new_ts

    def get_ts_between_events(self, event_name1, event_name2,
                              event_occurrence1=0, event_occurrence2=0):
        """
        Get a subset of the TimeSeries between two specified events.

        Parameters
        ----------
        event_name1, event_name2 : str
            Name of the events to look for in the events list.
        event_occurrence1, event_occurrence2 : int (optional)
            i_th occurence of the event to look for in the events list,
            starting at 0. The default is 0.

        Returns
        -------
        ts : TimeSeries
            A new TimeSeries following the specification.

        """
        time1 = self.get_event_time(event_name1, event_occurrence1)
        time2 = self.get_event_time(event_name2, event_occurrence2)
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
        ts : TimeSeries
            A new TimeSeries following the user interaction.

        """
        fig = plt.figure()
        self.plot(data_keys)
        ktk.mplhelper.message('Click on both sides of the portion to keep.')
        plt.pause(0.001)  # Redraw
        points = plt.ginput(2)
        ktk.mplhelper.message('')
        times = [points[0][0], points[1][0]]
        plt.close(fig)
        return self.get_ts_between_times(min(times), max(times))

    def isnan(self, data_key):
        """
        Return a boolean array of missing samples.

        Parameters
        ----------
        data_key : str
            Key value of the data signal to analyze.

        Returns
        -------
        nans : array
            A boolean array of the same size as the time vector, where True
            values represent missing samples (samples that contain at least
            one nan value).
        """
        values = self.data[data_key].copy()
        # Reduce the dimension of values while keeping the time dimension.
        while len(np.shape(values)) > 1:
            values = np.sum(values, 1)
        return np.isnan(values)

    def fill_missing_samples(self, max_missing_samples, method='linear'):
        """
        Fill missing samples with the given method.

        The sample rate must be constant.

        Parameters
        ----------
        max_missing_samples : int
            Maximal number of consecutive missing samples to fill. Set to
            zero to fill all missing samples.
        method : str (optional)
            The interpolation method. This input may take any value
            supported by scipy.interpolate.interp1d, such as:
                - 'linear'
                - 'nearest'
                - 'zero'
                - 'slinear'
                - 'quadratic'
                - 'cubic'
                - 'previous'
                - 'next'

        Returns
        -------
        None.

        """
        max_missing_samples = int(max_missing_samples)

        for data in self.data:

            # Fill missing samples
            is_visible = ~self.isnan(data)
            ts = self.get_subset(data)
            ts.data[data] = ts.data[data][is_visible]
            ts.time = ts.time[is_visible]
            ts.resample(self.time, method, fill_value='extrapolate')

            # Put back missing samples in holes longer than max_missing_samples
            if max_missing_samples > 0:
                hole_start_index = 0
                to_keep = np.ones(self.time.shape)
                for current_index in range(ts.time.shape[0]):
                    if is_visible[current_index]:
                        hole_start_index = current_index
                    elif (current_index - hole_start_index >
                          max_missing_samples):
                        to_keep[hole_start_index + 1:current_index + 1] = 0

                ts.data[data][to_keep == 0] = np.nan

            self.data[data] = ts.data[data]

    def shift(self, time):
        """
        Shift time and events.time.

        Parameters
        ----------
        time_shift : float
            Time to be added to time and events.time.

        Returns
        -------
        None.

        """
        for event in self.events:
            event.time = event.time + time
        self.time = self.time + time

    def sync_event(self, event_name, event_occurrence=0):
        """
        Shift time and events.time so that event_name is the new time zero.

        Parameters
        ----------
        event_name : str
            Name of the event to sync on.
        event_occurrence : int (optional)
            Occurrence of the event to sync on. The default is 0, which
            corresponds to the first occurrence of the event.

        Returns
        -------
        None.

        """
        self.shift(-self.get_event_time(event_name, event_occurrence))

    def ui_sync(self, data_keys=None, ts2=None, data_keys2=None):
        """
        Synchronize one or two TimeSeries by shifting their time.

        If a second TimeSeries is given, both TimeSeries are synchronized and
        the sync process is done in three steps:

        1. Click on the second TimeSeries's zero-time.
        2. Click on the second TimeSeries on a recognizable event that
           is common with the first TimeSeries.
        3. Click on this same event on the first TimeSeries.

        Parameters
        ----------
        data_keys : str or list of str (optional)
            The data keys to plot. The default is None, which means that all
            data is plotted.
        ts2 : TimeSeries (optional)
            A second TimeSeries that contains both a recognizable zero-time
            event and a common event with the first TimeSeries.
        data_keys2 : str or list of str (optional)
            The data keys from the second TimeSeries to plot. The default is
            None, which means that all data is plotted.

        Returns
        -------
        None.

        """
        fig = plt.figure()

        if ts2 is None:
            # Synchronize ts1 only
            self.plot(data_keys)
            choice = ktk.mplhelper.button_dialog(
                'Please zoom on the time zero and press Next.',
                ['Cancel', 'Next'])
            if choice != 1:
                plt.close(fig)
                return

            ktk.mplhelper.message('Click on the sync event.')
            click = plt.ginput(1)
            ktk.mplhelper.message(None)
            plt.close(fig)
            self.shift(-click[0][0])

        else:  # Sync two TimeSeries together

            finished = False
            axes = []
            while finished is False:

                if len(axes) == 0:
                    axes.append(plt.subplot(2, 1, 1))
                    axes.append(plt.subplot(2, 1, 2, sharex=axes[0]))

                plt.sca(axes[0])
                axes[0].cla()
                self.plot(data_keys)
                plt.title('First TimeSeries (ts1)')
                plt.grid(True)
                plt.tight_layout()

                plt.sca(axes[1])
                axes[1].cla()
                ts2.plot(data_keys2)
                plt.title('Second TimeSeries (ts2)')
                plt.grid(True)
                plt.tight_layout()

                choice = ktk.mplhelper.button_dialog(
                    'Please select an option.',
                    choices=['Sync both TimeSeries on a common event',
                             'Zero both TimeSeries using ts1',
                             'Zero both TimeSeries using ts2',
                             'OK'])

                if choice == 0:  # Sync on a common event
                    ktk.mplhelper.message(
                        'Click on the common event in ts1.')
                    click_1 = plt.ginput(1)
                    ktk.mplhelper.message(
                        'Click on the common event in ts2.')
                    click_2 = plt.ginput(1)
                    ktk.mplhelper.message('')

                    self.shift(-click_1[0][0])
                    ts2.shift(-click_2[0][0])

                elif choice == 1:  # Zero using TimeSeries 1
                    ktk.mplhelper.message(
                        'Click on the time zero in ts1.')
                    click_1 = plt.ginput(1)
                    ktk.mplhelper.message('')

                    self.shift(-click_1[0][0])
                    ts2.shift(-click_1[0][0])

                elif choice == 2:  # Zero using TimeSeries 2
                    ktk.mplhelper.message(
                        'Click on the time zero in ts2.')
                    click_2 = plt.ginput(1)
                    ktk.mplhelper.message('')

                    self.shift(-click_2[0][0])
                    ts2.shift(-click_2[0][0])

                elif choice == 3 or choice < -1:  # OK or closed figure, quit.
                    plt.close(fig)
                    finished = True

    def get_subset(self, data_keys):
        """
        Return a subset of the TimeSeries.

        This method returns a TimeSeries that contains only selected data
        keys. The corresponding data_info keys are copied in the new
        TimeSeries. All events are also copied in the new TimeSeries.

        Parameters
        ----------
        data_keys : str or list of str
            The data keys to extract from the timeseries.

        Returns
        -------
        ts : TimeSeries
            A copy of the TimeSeries, minus the unspecified data keys.

        Example
        -------
            >>> ts = ktk.TimeSeries(time = np.arange(10))
            >>> ts.data['signal1'] = ts.time
            >>> ts.data['signal2'] = ts.time**2
            >>> ts.data['signal3'] = ts.time**3
            >>> ts.data.keys()
            dict_keys(['signal1', 'signal2', 'signal3'])

            >>> ts2 = ts.get_subset(['signal1', 'signal3'])
            >>> ts2.data.keys()
            dict_keys(['signal1', 'signal3'])

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

    def resample(self, new_time, kind='linear', fill_value=None):
        """
        Resample the TimeSeries.

        Parameters
        ----------
        new_time : np.array
            The new time vector to resample the TimeSeries to.
        kind : str (optional)
            The interpolation method. This input may take any value
            supported by scipy.interpolate.interp1d, such as:
                - 'linear'
                - 'nearest'
                - 'zero'
                - 'slinear'
                - 'quadratic'
                - 'cubic'
                - 'previous'
                - 'next'
        fill_value : array-like or 'extrapolate' (optional)
            The fill value to use if new_time vector contains point outside
            the current TimeSeries' time vector. Use 'extrapolate' to
            extrapolate.

        Returns
        -------
        None.
        """
        for key in self.data.keys():
            index = ~self.isnan(key)

            if sum(index) < 3:  # Only Nans, cannot interpolate.
                print(f'Warning: Almost only NaNs found in signal {key}.')
                # We generate an array of nans of the expected size.
                new_shape = [len(new_time)]
                for i in range(1, len(self.data[key].shape)):
                    new_shape.append(self.data[key].shape[i])
                self.data[key] = np.empty(new_shape)
                self.data[key][:] = np.nan
            else:  # Interpolate.
                if ~np.all(index):
                    warnings.warn('Some NaNs were found. They were interpolated.')
                f = sp.interpolate.interp1d(self.time[index],
                                            self.data[key][index],
                                            axis=0, fill_value=fill_value,
                                            kind=kind)
                self.data[key] = f(new_time)

        self.time = new_time

    def merge(self, ts, data_keys=None, resample=False, overwrite=True):
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
        resample : bool (optional
            Set to True to resample the source TimeSeries, in case the time
            vectors are not matched. If the time vectors are not matched and
            resample is False, an exception is raised. Default is False.
        overwrite : bool (optional)
            If duplicates are found and overwrite is True, then the source (ts)
            overwrites the destination. Otherwise (overwrite is False), the
            duplicated data is ignored. Default is True.

        Returns
        -------
        None.
        """
        ts = ts.copy()
        if data_keys is None or len(data_keys) == 0:
            data_keys = ts.data.keys()
        else:
            if isinstance(data_keys, list) or isinstance(data_keys, tuple):
                pass
            elif isinstance(data_keys, str):
                data_keys = [data_keys]
            else:
                raise TypeError(
                    'data_keys must be a string or list of strings')

        # Check if resampling is needed
        if ((self.time.shape == ts.time.shape) and
                np.all(self.time == ts.time)):
            must_resample = False
        else:
            must_resample = True

        if must_resample is True and resample is False:
            raise ValueError(
                'Time vectors do not match, resampling is required.')

        if must_resample is True:
            ts.resample(self.time, fill_value='extrapolate')

        for key in data_keys:

            # Check if this key is a duplicate, then continue to next key if
            # required.
            if (key in self.data) and (key in ts.data) and overwrite is False:
                continue

            # Add this data
            self.data[key] = ts.data[key]

            if key in ts.data_info:
                for info_name in ts.data_info[key].keys():
                    self.add_data_info(key, info_name,
                                       ts.data_info[key][info_name])

        # Merge events
        for event in ts.events:
            self.events.append(event)
        self.sort_events()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
