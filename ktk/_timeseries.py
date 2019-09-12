"""
Module that manages the TimeSeries class.

Author: Felix Chenier
Date: July 2019
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import os
import shutil
from ast import literal_eval

from copy import deepcopy

import ktk

from . import gui
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
        self.

        Examples
        --------
            >>> the_timeseries.add_info('Forces', 'Unit', 'N')
            >>> the_timeseries.add_info('Marker1', 'Color', [43, 2, 255])

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
            The time of the event, in the same unit as the time_info{'Unit'}
            attribute of the TimeSeries (default: 's').
        name : str (optional)
            The name of the event.

        Returns
        -------
        self.

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
        return self

    def ui_add_event(self, name='event', plot=[]):
        """
        Add an event interactively to the TimeSeries.

        Parameters
        ----------
        name : str (optional)
            The name of the event.
        plot : str, list of str or tuple of str (optional)
            A signal name of list of signal name to be plotted, similar to
            the argument of ktk.TimeSeries.plot().

        Returns
        -------
        The same timeseries, with the event added.

        Example
        -------
        >>> ts = ktk.TimeSeries()
        >>> ts.ui_add_event('event1')

        """
        ts = self.copy()

        fig = plt.figure()

        while True:

            plt.cla()
            ts.plot(plot)
            button = gui.button_dialog(
                    'TimeSeries.ui_add_event',
                    'Please zoom on the new event to add, then click Next.\n' +
                    'Click Finished when there is no more event to add.',
                    ['Cancel', 'Next', 'Finished'])

            if button == 0:  # Cancel
                plt.close(fig)
                print('No event was added.')
                return self.copy()
            elif button == 2:  # Finish
                plt.close(fig)
                return ts
            # Next

            plt.title(('Left-click to add events, right-click to delete, '
                       'then ENTER.'))
            plt.pause(0.001)  # Update the plot
            coordinates = plt.ginput(99999)

            # Add these events
            for i in range(len(coordinates)):
                ts.add_event(coordinates[i][0], name)
        # Continue until done.

    def to_dataframes(self):
        """
        Convert the TimeSeries to a dict of Pandas dataframe.

        Parameters
        ----------
        No parameter.

        Returns
        -------
        A dict with the following keys:
            'data' : contains a pandas DataFrame with columns time and every
                     data reshaped to 1-d or 2-d arrays.
            'events' : contains a pandas DataFrame with columns time and name.
            'info' : contains a pandas DataFrame with columns time and every
                     data, and where each line is an information from
                     TimeSeries.time_info or TimeSeries.data_info.

        Example
        -------
        >>> # Let create a sample TimeSeries
        >>> ts = ktk.TimeSeries()
        >>> ts.time = np.linspace(0, 9, 10)

        >>> # where the first series is a scalar timeseries,
        >>> ts.data['data1'] = np.random.rand(10)

        >>> # and where the second series is a matrix timeseries.
        >>> ts.data['data2'] = np.random.rand(10, 2, 2)

        >>> # Let add some metadata and events:
        >>> ts.add_data_info('data1', 'Unit', 'm/s')
        >>> ts.add_data_info('data2', 'Unit', 'km/h')
        >>> ts.add_event(1.53, 'test_event1')
        >>> ts.add_event(7.2, 'test_event2')

        >>> # Now let convert this TimeSeries to pandas dataframes
        >>> dataframes = ts.to_dataframes()

        >>> print(dataframes['data'])
           time     data1  data2[0,0]  data2[0,1]  data2[1,0]  data2[1,1]
        0   0.0  0.600247    0.736891    0.902195    0.905907    0.065458
        1   1.0  0.860783    0.875474    0.414270    0.696410    0.872808
        2   2.0  0.564258    0.697806    0.542093    0.093780    0.394655
        3   3.0  0.345313    0.132531    0.073543    0.036600    0.697872
        4   4.0  0.768984    0.713446    0.672632    0.599467    0.211884
        5   5.0  0.394562    0.860927    0.769096    0.278852    0.317487
        6   6.0  0.213062    0.998223    0.831627    0.024960    0.960739
        7   7.0  0.327657    0.573798    0.191601    0.797447    0.728639
        8   8.0  0.821492    0.774073    0.942711    0.868428    0.667369
        9   9.0  0.614514    0.530900    0.737578    0.224186    0.895926

        >>> print(dataframes['events'])
           time         name
        0  1.53  test_event1
        1  7.20  test_event2

        >>> print(dataframes['info'])
             time data1 data2
        Unit    s   m/s  km/h

        """
        dict_out = dict()

        # DATA
        # ----

        df_time = pd.DataFrame(self.time)
        df_time.columns = ['time']

        df_data = ktk.loadsave.dict_to_dataframe(self.data)

        # Merge these dataframes
        df_out = pd.concat([df_time, df_data], axis=1)

        dict_out['data'] = df_out

        # EVENTS
        # ------
        if len(self.events) > 0:
            df_events = pd.DataFrame(self.events)
            df_events.columns = ['time', 'name']
        else:
            df_events = pd.DataFrame(columns=['time', 'name'])

        dict_out['events'] = df_events

        # INFO
        # ----
        df_time_info = pd.DataFrame({'time': self.time_info})
        df_data_info = pd.DataFrame(self.data_info)
        dict_out['info'] = pd.concat([df_time_info, df_data_info],
                                     axis=1, sort=False)

        return dict_out

    def from_dataframes(data, events=pd.DataFrame(), info=pd.DataFrame()):
        """
        Generate a TimeSeries based on pandas DataFrames.

        Parameters
        ----------
        data : DataFrame
            Generates the TimeSeries' time and data fields.
        events : DataFrame, optional
            Generates the TimeSeries' events field.
        info : DataFrame, optional
            Generate the TimeSeries' time_info and data_info fields.

        Returns
        -------
        TimeSeries.

        See the ``TimeSeries.to_dataframes`` method for more information on the
        shapes of data, events and info DataFrames.

        """
        out = TimeSeries()

        # DATA AND TIME
        # -------------

        # Determine the time length
        time = data.time

        # Determine the data fields
        all_column_names = data.columns
        all_data_names = []
        all_data_highest_indices = []

        for one_column_name in all_column_names:
            opening_bracket_position = one_column_name.find('[')
            if opening_bracket_position == -1:
                # No dimension for this data
                all_data_names.append(one_column_name)
                all_data_highest_indices.append([len(time)-1])
            else:
                # Extract name and dimension
                data_name = one_column_name[0:opening_bracket_position]
                data_dimension = literal_eval(
                        '[' + str(len(time)-1) + ',' +
                        one_column_name[opening_bracket_position+1:])

                all_data_names.append(data_name)
                all_data_highest_indices.append(data_dimension)

        # Create the timeseries data

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
                            data[all_column_names[i]])
                    unique_data_highest_index.append(
                            all_data_highest_indices[i])

            # Sort the sub-dataframe's columns
            sub_dataframe.reindex(sorted(sub_dataframe.columns), axis=1)

            # Calculate the data dimension we must reshape to
            unique_data_dimension = np.max(
                    np.array(unique_data_highest_index)+1, axis=0)

            # Convert the dataframe to a np.array, then reshape.
            new_data = sub_dataframe.to_numpy()
            new_data = np.reshape(new_data, unique_data_dimension)

            if unique_data_name == 'time':
                out.time = new_data
            else:
                out.data[unique_data_name] = new_data

        # EVENTS
        # ------
        for i_event in range(0, len(events)):
            out.add_event(events.time[i_event], events.name[i_event])

        # INFO
        # ----
        n_rows = len(info)
        row_names = list(info.index)
        for column_name in info.columns:
            for i_row in range(0, n_rows):
                one_info = info[column_name][i_row]
                if str(one_info).lower() != 'nan':
                    if column_name == 'time':
                        out.time_info[row_names[i_row]] = one_info
                    else:
                        out.add_data_info(column_name, row_names[i_row],
                                          one_info)

        return out

    def save(self, filename):
        """
        Save the TimeSeries as a zip archive of tab-delimited text files.

        Parameters
        ----------
        filename : str
            Filename, including the path if desired.

        Returns
        -------
        None.

        This method saves all the TimeSeries information as a zip file that
        contains three separate tab-delimited text files:
            - data.txt :   an ascii file that contains the TimeSeries' time and
                           data in columns. TimeSeries of matrices are reshaped
                           so that each matrix element are in a column.
            - events.txt : an ascii file that contains the TimeSeries' events.
                           The time is in a column, the event names are in
                           another column.
            - info.txt :   an ascii file that contains all the time and data
                           info, where all data_info keys have their own
                           column.

        The values in the files at separated by tabs. The tables are generated
        using the ``to_dataframes`` method and the corresponding pandas
        DataFrames are saved to csv using the pandas' ``to_csv`` method.

        """
        # Create the pandas dataframes
        dataframes = self.to_dataframes()

        # Create a temporary folder
        try:
            # TODO be a little nicer and ask before deleting.
            shutil.rmtree(filename)
        except:
            pass

        os.mkdir(filename)

        # Create the csv files
        dataframes['data'].to_csv(filename + '/data.txt',
                                  sep='\t', quoting=csv.QUOTE_NONNUMERIC,
                                  index=False)
        dataframes['events'].to_csv(filename + '/events.txt',
                                    sep='\t', quoting=csv.QUOTE_NONNUMERIC,
                                    index=False)
        dataframes['info'].to_csv(filename + '/info.txt',
                                  sep='\t', quoting=csv.QUOTE_NONNUMERIC)

        # Create the archive
        shutil.make_archive(filename + '-temp', 'zip', filename)

        # Move the archive to its final name
        shutil.rmtree(filename)
        os.rename(filename + '-temp.zip', filename)

        # Just to be sure, load back the file and compare its content, so
        # we are sure everything has been saved correctly.
        ts = TimeSeries.load(filename)
        if ts != self:
            raise AssertionError('The file was saved, but its ' +
                                 'content may differ to the original data.')

    def load(filename):
        """
        Load a TimeSeries from a zip archive of tab-delimited files.

        Please see ``TimeSeries.save`` for information of the expected zip
        archive structure.
        """
        # TODO Support individual txt files
        # TODO Add a better help

        temp_folder = filename + '_temp_folder'
        try:
            shutil.rmtree(temp_folder)
        except:
            pass

        os.mkdir(temp_folder)
        shutil.unpack_archive(filename, temp_folder, 'zip')

        data = pd.read_csv(temp_folder + '/data.txt',
                           sep='\t', quoting=csv.QUOTE_NONNUMERIC)
        events = pd.read_csv(temp_folder + '/events.txt',
                             sep='\t', quoting=csv.QUOTE_NONNUMERIC)
        info = pd.read_csv(temp_folder + '/info.txt',
                           sep='\t', quoting=csv.QUOTE_NONNUMERIC,
                           index_col=0)

        dataframes = TimeSeries.from_dataframes(data, events, info)
        shutil.rmtree(temp_folder)
        return dataframes

    def copy(self):
        """
        Deep copy of a TimeSeries.

        Returns
        -------
        A deep copy of the original TimeSeries.

        """
        return deepcopy(self)

    def plot(self, data_keys=None):
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

            # Next plot
            i_plot += 1

        # Add labels and format
        plt.xlabel('Time (' + self.time_info['Unit'] + ')')
        plt.tight_layout(pad=0.05)
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
        fig.canvas.set_window_title('Click on the left of the desired zone.')
        left_point = plt.ginput(1)
        fig.canvas.set_window_title('Click on the right of the desired zone.')
        right_point = plt.ginput(1)
        plt.close(fig)
        return self.get_ts_between_times(left_point[0][0], right_point[0][0])
