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
Provide the TimeSeries and TimeSeriesEvent classes.

The classes defined in this module are accessible directly from the toplevel
Kinetics Toolkit's namespace (i.e. ktk.TimeSeries, ktk.TimeSeriesEvent)

"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import kineticstoolkit._repr
from kineticstoolkit.decorators import stable, experimental, unstable
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd
import warnings
from ast import literal_eval
from copy import deepcopy
from typing import Dict, List, Tuple, Any, Union

import kineticstoolkit as ktk  # For doctests


ts_listing = []  # type: List[str]


def dataframe_to_dict_of_arrays(
        dataframe: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Convert a pandas DataFrame to a dict of numpy ndarrays.

    This function mirrors the dict_of_arrays_to_dataframe function. It is
    mainly used by the TimeSeries.from_dataframe method.

    Parameters
    ----------
    pd_dataframe
        The dataframe to be converted.

    Returns
    -------
    Dict[str, np.ndarray]


    Examples
    --------
    In the simplest case, each dataframe column becomes a dict key.

        >>> df = pd.DataFrame([[0, 3], [1, 4], [2, 5]])
        >>> df.columns = ['column1', 'column2']
        >>> df
           column1  column2
        0        0        3
        1        1        4
        2        2        5

        >>> data = dataframe_to_dict_of_arrays(df)

        >>> data['column1']
        array([0, 1, 2])

        >>> data['column2']
        array([3, 4, 5])

    If the dataframe contains similar column names with indices in brackets
    (for example, Forces[0], Forces[1], Forces[2]), then these columns are
    combined in a single array.

        >>> df = pd.DataFrame([[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]])
        >>> df.columns = ['Forces[0]', 'Forces[1]', 'Forces[2]', 'Other']
        >>> df
           Forces[0]  Forces[1]  Forces[2]  Other
        0          0          3          6      9
        1          1          4          7     10
        2          2          5          8     11

        >>> data = dataframe_to_dict_of_arrays(df)

        >>> data['Forces']
        array([[0, 3, 6],
               [1, 4, 7],
               [2, 5, 8]])

        >>> data['Other']
        array([ 9, 10, 11])

    """
    # Remove spaces in indexes between brackets
    columns = dataframe.columns
    new_columns = []
    for i_column, column in enumerate(columns):
        splitted = column.split('[')
        if len(splitted) > 1:  # There are brackets
            new_columns.append(
                splitted[0] + '[' + splitted[1].replace(' ', ''))
        else:
            new_columns.append(column)
    dataframe.columns = columns

    # Search for the column names and their dimensions
    # At the end, we end with something like:
    #    dimensions['Data1'] = []
    #    dimensions['Data2'] = [[0], [1], [2]]
    #    dimensions['Data3'] = [[0,0],[0,1],[1,0],[1,1]]
    dimensions = dict()  # type: Dict[str, List]
    for column in dataframe.columns:
        splitted = column.split('[')
        if len(splitted) == 1:  # No brackets
            dimensions[column] = []
        else:  # With brackets
            key = splitted[0]
            index = literal_eval('[' + splitted[1])

            if key in dimensions:
                dimensions[key].append(index)
            else:
                dimensions[key] = [index]

    n_samples = len(dataframe)

    # Assign the columns to the output
    out = dict()  # type: Dict[str, np.array]
    for key in dimensions:
        if len(dimensions[key]) == 0:
            out[key] = dataframe[key].to_numpy()
        else:
            highest_dims = np.max(np.array(dimensions[key]), axis=0)

            columns = [key + str(dim).replace(' ', '')
                       for dim in sorted(dimensions[key])]
            out[key] = dataframe[columns].to_numpy()
            out[key] = np.reshape(out[key],
                                  [n_samples] + (highest_dims + 1).tolist())

    return out


def dict_of_arrays_to_dataframe(
        dict_of_arrays : Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Convert a dict of ndarray of any dimension to a pandas DataFrame.

    This function mirrors the dataframe_to_dict_of_arrays function. It is
    mainly used by the TimeSeries.to_dataframe method.

    The rows in the output DataFrame correspond to the first dimension of the
    numpy arrays.

    - Vectors are converted to single-column DataFrames.
    - 2-dimensional arrays are converted to multi-columns DataFrames.
    - 3-dimensional (or more) arrays are also converted to DataFrames, but
      indices in brackets are added to the column names.

    Parameters
    ----------
    dict_of_array
        A dict that contains numpy arrays. Each array must have the same
        first dimension's size.

    Returns
    -------
    DataFrame

    Example
    -------
    >>> data = dict()
    >>> data['Forces'] = np.arange(12).reshape((4, 3))
    >>> data['Other'] = np.arange(4)

    >>> data['Forces']
    array([[ 0,  1,  2],
           [ 3,  4,  5],
           [ 6,  7,  8],
           [ 9, 10, 11]])

    >>> data['Other']
    array([0, 1, 2, 3])

    >>> df = dict_of_arrays_to_dataframe(data)

    >>> df
       Forces[0]  Forces[1]  Forces[2]  Other
    0          0          1          2      0
    1          3          4          5      1
    2          6          7          8      2
    3          9         10         11      3

    It also works with higher dimensions:

    >>> data = {'3d_data': np.arange(8).reshape((2, 2, 2))}

    >>> data['3d_data']
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])

    >>> df = dict_of_arrays_to_dataframe(data)

    >>> df
       3d_data[0,0]  3d_data[0,1]  3d_data[1,0]  3d_data[1,1]
    0             0             1             2             3
    1             4             5             6             7

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
        # The strategy here is to build matrices of indices, that have
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
                    if i_indice == n_indices-1:
                        this_column_name += ']'
                    else:
                        this_column_name += ','

            column_names.append(this_column_name)

        df_data.columns = column_names

        # Merge this dataframe with the output dataframe
        df_out = pd.concat([df_out, df_data], axis=1)

    return df_out


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

    Example
    -------
    >>> event = ktk.TimeSeriesEvent()
    >>> event.time = 1.5
    >>> event.name = 'event_name'
    >>> event
    [1.5, 'event_name']

    """

    def __dir__(self) -> List[str]:
        return ['time', 'name']

    def __init__(self, time: float = 0., name: str = 'event'):
        list.__init__(self)
        self.append(float(time))
        self.append(str(name))

    @property
    def time(self):
        """The time at which the event happened."""
        return self[0]

    @time.setter
    def time(self, time):
        self[0] = float(time)

    @property
    def name(self):
        """The name of the event."""
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
    time : np.ndarray
        Time vector as 1-dimension np.array.

    data : Dict[str, np.ndarray]
        Contains the data, where each element contains a np.array
        which first dimension corresponds to time.

    time_info : Dict[str, Any]
        Contains metadata relative to time. The default is {'Unit': 's'}

    data_info : Dict[str, Dict[str, Any]]
        Contains facultative metadata relative to data. For example, the
        data_info attribute could indicate the unit of data['Forces']:

        data['Forces'] = {'Unit': 'N'}.

        To facilitate the management of data_info, please use
        `ktk.TimeSeries.add_data_info`.

    events : List[TimeSeriesEvent]
        List of events.

    Example
    -------
        >>> ts = ktk.TimeSeries(time=np.arange(0,100))

    """
    def __init__(
            self,
            time: np.ndarray = np.array([]),
            time_info: Dict[str, Any] = {'Unit': 's'},
            data: Dict[str, np.ndarray] = {},
            data_info: Dict[str, Dict[str, Any]] = {},
            events: List[TimeSeriesEvent] = []):

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
        return kineticstoolkit._repr._format_class_attributes(self)

    def __repr__(self):
        return kineticstoolkit._repr._format_class_attributes(self)

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

    @stable(ts_listing)
    def to_dataframe(self) -> pd.DataFrame:
        """
        Create a DataFrame by reshaping all data to one bidimensional table.

        Returns
        -------
        DataFrame
            DataFrame with the index as the TimeSeries' time. Vector data are
            converted to single columns, and two-dimensional (or more) data are
            converted to multiple columns with the additional dimensions in
            brackets. The TimeSeries's events and metadata such as time_info
            and data_info are not included in the resulting DataFrame.

        """
        df = dict_of_arrays_to_dataframe(self.data)
        df.index = self.time
        return df

    @stable(ts_listing)
    def from_dataframe(dataframe: pd.DataFrame, /) -> 'TimeSeries':
        """
        Create a new TimeSeries from a Pandas Dataframe

        Parameters
        ----------
        dataframe
            A Pandas DataFrame where the index corresponds to time, and
            where each column corresponds to a data key. As special cases,
            data in column which names end with bracketed indices such as
            [0], [1], [0,0], [0,1], etc. are converted to multidimensional
            arrays. For example, if a DataFrame has these column names:
            ['Forces[0]', 'Forces[1]', 'Forces[2]', 'Forces[3]'],
            then a single data key is created (Forces) and the data itself
            will be of shape Nx4, N being the number of samples (the length
            of the DataFrame).

        Example
        -------
            ts = ktk.TimeSeries.from_dataframe(the_dataframe)

        """
        ts = TimeSeries()
        ts.data = dataframe_to_dict_of_arrays(dataframe)
        ts.time = dataframe.index.to_numpy()
        return ts

    @stable(ts_listing)
    def add_data_info(self, data_key: str, info_key: str, value: Any) -> None:
        """
        Add metadata to TimeSeries' data.

        Parameters
        ----------
        data_key
            The data key the info corresponds to.
        info_key
            The key of the info dict.
        value
            The info.

        Returns
        -------
        None.

        Example
        -------
        We create a TimeSeries with some data info:

        >>> ts = ktk.TimeSeries()
        >>> ts.add_data_info('Forces', 'Unit', 'N')
        >>> ts.add_data_info('Marker1', 'Color', [43, 2, 255])

        Let's see the result:

        >>> ts.data_info['Forces']
        {'Unit': 'N'}

        >>> ts.data_info['Marker1']
        {'Color': [43, 2, 255]}

        """
        if data_key in self.data_info:   # Assign the value
            self.data_info[data_key][info_key] = value
        else:  # Create and assign value
            self.data_info[data_key] = {info_key: value}

    @stable(ts_listing)
    def remove_data_info(self, data_key: str, info_key: str, /) -> None:
        """
        Remove metadata from a TimeSeries' data.

        Parameters
        ----------
        data_key
            The data key the info corresponds to.
        info_key
            The key of the info dict.

        Note
        ----
        No warning or exception is raised if the data key does not exist.

        Example
        -------
        >>> ts = ktk.TimeSeries()
        >>> ts.add_data_info('Forces', 'Unit', 'N')
        >>> ts.data_info['Forces']
        {'Unit': 'N'}

        >>> ts.remove_data_info('Forces', 'Unit')
        >>> ts.data_info['Forces']
        {}

        """
        try:
            self.data_info[data_key].pop(info_key)
        except KeyError:
            pass

    @stable(ts_listing)
    def rename_data(self, old_data_key: str, new_data_key: str, /) -> None:
        """
        Rename a key in data and data_info.

        Parameters
        ----------
        old_data_key
            Name of the current data key.
        new_data_key
            New name of the data key.

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
        if old_data_key in self.data:
            self.data[new_data_key] = self.data.pop(old_data_key)
        if old_data_key in self.data_info:
            self.data_info[new_data_key] = self.data_info.pop(old_data_key)

    @stable(ts_listing)
    def remove_data(self, data_key: str, /) -> None:
        """
        Remove a data key and its associated metadata.

        Parameters
        ----------
        data_key
            Name of the data key.

        Note
        ----
        No warning or exception is raised if the data key does not exist.

        Example
        -------
        >>> # Prepare a test TimeSeries with data 'test'
        >>> ts = ktk.TimeSeries()
        >>> ts.data['test'] = np.arange(10)
        >>> ts.add_data_info('test', 'Unit', 'm')

        >>> ts.data
        {'test': array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}

        >>> ts.data_info
        {'test': {'Unit': 'm'}}

        >>> # Now remove data 'test'
        >>> ts.remove_data('test')

        >>> ts.data
        {}

        >>> ts.data_info
        {}

        """
        try:
            self.data.pop(data_key)
        except KeyError:
            pass
        try:
            self.data_info.pop(data_key)
        except KeyError:
            pass

    @stable(ts_listing)
    def add_event(self, time: float, name: str = 'event') -> None:
        """
        Add an event to the TimeSeries.

        Parameters
        ----------
        time
            The time of the event, in the same unit as `time_info['Unit']`.
        name
            Optional. The name of the event.

        Example
        -------
            >>> ts = ktk.TimeSeries()
            >>> ts.add_event(5.5, 'event1')
            >>> ts.add_event(10.8, 'event2')
            >>> ts.add_event(2.3, 'event2')

            >>> ts.events
            [[5.5, 'event1'], [10.8, 'event2'], [2.3, 'event2']]

        """
        self.events.append(TimeSeriesEvent(time, name))

    @stable(ts_listing)
    def remove_event(self,
                     event_name: str, event_occurrence : int = 0, /) -> None:
        """
        Remove an event.

        Parameters
        ----------
        event_name
            Name of the event to look for in the events list.
        event_occurrence
            Optional. i_th occurence of the event to look for in the events
            list, starting at 0, where the occurrences are sorted in time.

        Example
        -------
        >>> # Instanciate a timeseries with some events
        >>> ts = ktk.TimeSeries()
        >>> ts.add_event(5.5, 'event1')
        >>> ts.add_event(10.8, 'event2')
        >>> ts.add_event(2.3, 'event2')

        >>> ts.events
        [[5.5, 'event1'], [10.8, 'event2'], [2.3, 'event2']]

        >>> ts.remove_event('event1')
        >>> ts.events
        [[10.8, 'event2'], [2.3, 'event2']]

        >>> ts.remove_event('event2', 1)
        >>> ts.events
        [[2.3, 'event2']]

        """
        time_tolerance = 1E-15
        event_time = self.get_event_time(event_name, event_occurrence)
        new_events = []
        for event in self.events:
            if (np.abs(event.time - event_time) > time_tolerance or
                    event.name != event_name):
                new_events.append(event)
        self.events = new_events

    @stable(ts_listing)
    def ui_add_event(self,
                     name: str = 'event',
                     plot: Union[str, List[str]] = [], /, *,
                     multiple_events: bool = False) -> bool:
        """
        Add one or many events interactively to the TimeSeries.

        Parameters
        ----------
        name
            Optional. The name of the event.
        plot
            Optional. A signal name of list of signal name to be plotted,
            similar to the argument of ktk.TimeSeries.plot().
        multiple_events
            Optional. False to add only one event, True to add multiple events
            with the same name.

        Returns
        -------
        bool
            True if the event was added, False if the operation was cancelled
            by the user.
        """
        ts = self.copy()

        fig = plt.figure()
        ts.plot(plot)

        finished = False

        while finished is False:
            finished = True  # Only one pass by default

            button = kineticstoolkit.gui.button_dialog(
                f'Adding the event "{name}".\n'
                'Please zoom on the location to \n'
                'add the event, then click Next.',
                ['Cancel', 'Next'])

            if button <= 0:  # Cancel
                plt.close(fig)
                return False

            if multiple_events:
                kineticstoolkit.gui.message(
                    'Left-click to add events; \n'
                    'Right-click to delete; \n'
                    'ENTER to finish.')
                plt.pause(0.001)  # Update the plot
                coordinates = plt.ginput(99999)
                kineticstoolkit.gui.message('')

            else:
                kineticstoolkit.gui.message(
                    'Please left-click on the event to add.')
                coordinates = plt.ginput(1)
                kineticstoolkit.gui.message('')

            # Add these events
            for i in range(len(coordinates)):
                ts.add_event(coordinates[i][0], name)

            if multiple_events:
                plt.cla()
                ts.plot(plot)
                button = kineticstoolkit.gui.button_dialog(
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

        kineticstoolkit.gui.message('')
        plt.close(fig)
        self.events = ts.events  # Add the events to self.
        return True

    @stable(ts_listing)
    def sort_events(self, *, unique: bool = True) -> None:
        """
        Sorts the TimeSeries' events from the earliest to the latest.

        Parameters
        ----------
        unique
            Optional. True to make events unique so that no two events can
            have both the same name and the same time.

        Example
        -------
        >>> ts = ktk.TimeSeries(time=np.arange(100)/10)
        >>> ts.add_event(2, 'two')
        >>> ts.add_event(1, 'one')
        >>> ts.add_event(3, 'three')
        >>> ts.add_event(3, 'three')

        >>> ts.events
        [[2.0, 'two'], [1.0, 'one'], [3.0, 'three'], [3.0, 'three']]

        >>> ts.sort_events(unique=False)
        >>> ts.events
        [[1.0, 'one'], [2.0, 'two'], [3.0, 'three'], [3.0, 'three']]

        >>> ts.sort_events()
        >>> ts.events
        [[1.0, 'one'], [2.0, 'two'], [3.0, 'three']]

        """
        self.events = sorted(self.events)

        if unique is True:
            for i in range(len(self.events) - 1, 0, -1):
                if ((self.events[i].time == self.events[i - 1].time) and
                        (self.events[i].name == self.events[i - 1].name)):
                    self.events.pop(i)

    @stable(ts_listing)
    def copy(self) -> 'TimeSeries':
        """Deep copy of a TimeSeries."""
        return deepcopy(self)

    @stable(ts_listing)
    def plot(self,
             data_keys: Union[str, List[str]] = [],
             /, *,
             event_names: bool = True,
             legend: bool = True,
             **kwargs) -> None:
        """
        Plot the TimeSeries using matplotlib.

        Parameters
        ----------
        data_keys
            The data keys to plot. If left empty, all data is plotted.
        event_names
            Optional. True to plot the event names on top of the event lines.
        legend
            Optional. True to plot a legend, False otherwise.

        Additional keyboard arguments are passed to the pyplot's plot function.

        Example
        -------
        If a TimeSeries `ts`' data attribute as keys 'Forces', 'Moments' and
        'Angle', then:

            ts.plot(['Forces', 'Moments'])

        plots only the forces and moments, without plotting the angle.

        """
        if data_keys is None or len(data_keys) == 0:
            # Plot all
            ts = self.copy()
        else:
            ts = self.get_subset(data_keys)

        # Sort events to help finding each event's occurrence
        ts.sort_events(unique=False)

        df = ts.to_dataframe()
        labels = df.columns.to_list()

        axes = plt.gca()
        axes.set_prop_cycle(mpl.cycler(
            linewidth=[1, 2, 3, 4]) * mpl.cycler(
                linestyle=['-', '--', '-.', ':']) * mpl.cycler(
                    color=['r', 'g', 'b', 'c', 'm', 'y', 'k', 'tab:orange']))

        # Plot the curves
        for i_label, label in enumerate(labels):
            axes.plot(df.index.to_numpy(),
                     df[label].to_numpy(), label=label, **kwargs)

        # Add labels
        plt.xlabel('Time (' + ts.time_info['Unit'] + ')')

        # Make unique list of units
        unit_set = set()
        for data in ts.data_info:
            for info in ts.data_info[data]:
                if info == 'Unit':
                    unit_set.add(ts.data_info[data][info])
        # Plot this list
        unit_str = '('
        for unit in unit_set:
            if len(unit_str) > 1:
                unit_str += ', '
            unit_str += unit
        unit_str += ')'

        plt.ylabel(unit_str)

        # Plot the events
        n_events = len(ts.events)
        event_times = []
        for event in ts.events:
            event_times.append(event.time)

        if len(ts.events) > 0:
            a = plt.axis()
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

            plt.plot(event_line_x, event_line_y, ':k')

            if event_names:
                occurrences = {}  # type:Dict[str, int]

                for event in ts.events:
                    if event.name == '_':
                        name = '_'
                    elif event.name in occurrences:
                        occurrences[event.name] += 1
                        name = f"{event.name} {occurrences[event.name]}"
                    else:
                        occurrences[event.name] = 0
                        name = f"{event.name} 0"

                    plt.text(event.time, max_y, name,
                             rotation='vertical',
                             horizontalalignment='center',
                             fontsize='small')

        if legend:
            if len(labels) < 20:
                legend_location = 'best'
            else:
                legend_location = 'upper right'

            axes.legend(loc=legend_location,
                       ncol=1 + int(len(labels) / 40))  # Max 40 items per line

    @stable(ts_listing)
    def get_index_at_time(self, time: float, /) -> int:
        """
        Get the time index that is the closest to the specified time.

        Parameters
        ----------
        time
            Time to look for in the TimeSeries' time vector.

        Returns
        -------
        int
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
        return np.argmin(np.abs(self.time - float(time)))

    @stable(ts_listing)
    def get_index_before_time(self, time: float, /, *,
                              inclusive: bool = False) -> int:
        """
        Get the time index that is just before the specified time.

        Parameters
        ----------
        time
            Time to look for in the TimeSeries' time vector.
        inclusive
            Optional. True to include the given time in the comparison.

        Returns
        -------
        int
            The index in the time vector. If no value is before the specified
            time, a value of np.nan is returned.

        Example
        -------
        >>> ts = ktk.TimeSeries(time=np.array([0, 0.5, 1, 1.5, 2]))

        >>> ts.get_index_before_time(0.9)
        1

        >>> ts.get_index_before_time(1)
        1

        >>> ts.get_index_before_time(1, inclusive=True)
        2

        >>> ts.get_index_before_time(1.1)
        2

        >>> ts.get_index_before_time(-1)
        nan

        """
        diff = float(time) - self.time
        if inclusive:
            diff[diff < 0] = np.nan
        else:
            diff[diff <= 0] = np.nan
        if np.all(np.isnan(diff)):  # All nans
            return np.nan
        else:
            return np.nanargmin(diff)

    @stable(ts_listing)
    def get_index_after_time(self, time: float, /, *,
                             inclusive: bool = False) -> int:
        """
        Get the time index that is just after the specified time.

        Parameters
        ----------
        time
            Time to look for in the TimeSeries' time vector.
        inclusive
            Optional. True to include the given time in the comparison.

        Returns
        -------
        int
            The index in the time vector. If no value is after the
            specified time, a value of np.nan is returned.

        Example
        -------
        >>> ts = ktk.TimeSeries(time=np.array([0, 0.5, 1, 1.5, 2]))

        >>> ts.get_index_after_time(0.9)
        2

        >>> ts.get_index_after_time(1)
        3

        >>> ts.get_index_after_time(1, inclusive=True)
        2

        >>> ts.get_index_after_time(1.1)
        3

        >>> ts.get_index_after_time(13)
        nan

        """
        diff = self.time - float(time)
        if inclusive:
            diff[diff < 0] = np.nan
        else:
            diff[diff <= 0] = np.nan
        if np.all(np.isnan(diff)):  # All nans
            return np.nan
        else:
            return np.nanargmin(diff)

    @stable(ts_listing)
    def get_event_time(self, event_name: str,
                       event_occurrence: int = 0, /) -> float:
        """
        Get the time of the specified event.

        Parameters
        ----------
        event_name
            Name of the event to look for in the events list.
        event_occurrence
             Optional. i_th occurence of the event to look for in the events
             list, starting at 0.

        Returns
        -------
        float
            The time of the specified event. If no corresponding event is
            found, then np.nan is returned.

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

    @stable(ts_listing)
    def get_ts_at_time(self, time: float, /) -> 'TimeSeries':
        """
        Get a one-data subset of the TimeSeries at the nearest time.

        Parameters
        ----------
        time
            Time to look for in the TimeSeries' time vector.

        Returns
        -------
        TimeSeries
            A TimeSeries of length 1, at the time neasest to the specified
            time.

        Example
        -------
        >>> ts = ktk.TimeSeries(time=np.array([0, 0.5, 1, 1.5, 2]))
        >>> ts.time
        array([0. , 0.5, 1. , 1.5, 2. ])

        >>> ts.get_index_at_time(0.9)
        2

        >>> ts.get_index_at_time(1)
        2

        >>> ts.get_index_at_time(1.1)
        2

        """
        out_ts = self.copy()
        index = self.get_index_at_time(time)
        out_ts.time = out_ts.time[index]
        for the_data in out_ts.data.keys():
            out_ts.data[the_data] = out_ts.data[the_data][index]
        return out_ts

    @stable(ts_listing)
    def get_ts_at_event(self, event_name: str,
                        event_occurrence: int = 0, /) -> 'TimeSeries':
        """
        Get a one-data subset of the TimeSeries at the event's nearest time.

        Parameters
        ----------
        event_name
            Name of the event to look for in the events list.
        event_occurrence
            Optional. i_th occurence of the event to look for in the events
            list, starting at 0.

        Returns
        -------
        TimeSeries
            A TimeSeries of length 1, at the event's nearest time.

        """
        time = self.get_event_time(event_name, event_occurrence)
        return self.get_ts_at_time(time)

    @stable(ts_listing)
    def get_ts_before_index(self, index: int, /, *,
                            inclusive: bool = False) -> 'TimeSeries':
        """
        Get a subset of the TimeSeries before the specified time index.

        Parameters
        ----------
        index
            Time index
        inclusive
            Optional. True to include the given time index.

        Example
        -------
        >>> ts = ktk.TimeSeries(time=np.arange(10)/10)
        >>> ts.time
        array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        >>> ts.get_ts_before_index(2).time
        array([0. , 0.1])

        >>> ts.get_ts_before_index(2, inclusive=True).time
        array([0. , 0.1, 0.2])

        """
        out_ts = self.copy()
        if np.isnan(index):
            index_range = range(0)
        else:
            if inclusive:
                index_range = range(index + 1)
            else:
                index_range = range(index)

        out_ts.time = out_ts.time[index_range]
        for the_data in out_ts.data.keys():
            out_ts.data[the_data] = out_ts.data[the_data][index_range]
        return out_ts

    @stable(ts_listing)
    def get_ts_after_index(self, index: int, /, *,
                           inclusive: bool = False) -> 'TimeSeries':
        """
        Get a subset of the TimeSeries after the specified time index.

        Parameters
        ----------
        index
            Time index
        inclusive
            Optional. True to include the given time index.

        Example
        -------
        >>> ts = ktk.TimeSeries(time=np.arange(10)/10)
        >>> ts.time
        array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        >>> ts.get_ts_after_index(2).time
        array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        >>> ts.get_ts_after_index(2, inclusive=True).time
        array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        """
        out_ts = self.copy()
        if np.isnan(index):
            index_range = range(0)
        else:
            if inclusive:
                index_range = range(index, len(self.time))
            else:
                index_range = range(index + 1, len(self.time))

        out_ts.time = out_ts.time[index_range]
        for the_data in out_ts.data.keys():
            out_ts.data[the_data] = out_ts.data[the_data][index_range]
        return out_ts

    @stable(ts_listing)
    def get_ts_between_indexes(self, index1: int, index2: int, /, *,
                               inclusive: bool = False) -> 'TimeSeries':
        """
        Get a subset of the TimeSeries before two specified time indexes.

        Parameters
        ----------
        index1, index2
            Time indexes
        inclusive
            Optional. True to include the given time indexes.

        Example
        -------
        >>> ts = ktk.TimeSeries(time=np.arange(10)/10)
        >>> ts.time
        array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        >>> ts.get_ts_between_indexes(2, 5).time
        array([0.3, 0.4])

        >>> ts.get_ts_between_indexes(2, 5, inclusive=True).time
        array([0.2, 0.3, 0.4, 0.5])

        """
        out_ts = self.copy()
        if np.isnan(index1) or np.isnan(index2):
            index_range = range(0)
        else:
            if inclusive:
                index_range = range(index1, index2 + 1)
            else:
                index_range = range(index1 + 1, index2)

        out_ts.time = out_ts.time[index_range]
        for the_data in out_ts.data.keys():
            out_ts.data[the_data] = out_ts.data[the_data][index_range]
        return out_ts

    @stable(ts_listing)
    def get_ts_before_time(self, time: float, /, *,
                           inclusive: bool = False) -> 'TimeSeries':
        """
        Get a subset of the TimeSeries before the specified time.

        Parameters
        ----------
        time
            Time to look for in the TimeSeries' time vector.
        inclusive
            Optional. True to include the given time in the comparison.

        Example
        -------
        >>> ts = ktk.TimeSeries(time=np.arange(10)/10)
        >>> ts.time
        array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        >>> ts.get_ts_before_time(0.3).time
        array([0. , 0.1, 0.2])

        >>> ts.get_ts_before_time(0.3, inclusive=True).time
        array([0. , 0.1, 0.2, 0.3])

        """
        out_ts = self.copy()
        index = self.get_index_before_time(time, inclusive=inclusive)
        if np.isnan(index):
            index_range = range(0)
        else:
            index_range = range(0, index + 1)

        out_ts.time = out_ts.time[index_range]
        for the_data in out_ts.data.keys():
            out_ts.data[the_data] = out_ts.data[the_data][index_range]
        return out_ts

    @stable(ts_listing)
    def get_ts_after_time(self, time: float, /, *,
                          inclusive: bool = False) -> 'TimeSeries':
        """
        Get a subset of the TimeSeries after the specified time.

        Parameters
        ----------
        time
            Time to look for in the TimeSeries' time vector.
        inclusive
            Optional. True to include the given time in the comparison.

        Example
        -------
        >>> ts = ktk.TimeSeries(time=np.arange(10)/10)
        >>> ts.time
        array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        >>> ts.get_ts_after_time(0.3).time
        array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        >>> ts.get_ts_after_time(0.3, inclusive=True).time
        array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        """
        if inclusive:
            index = self.get_index_before_time(time, inclusive=True)
        else:
            index = self.get_index_after_time(time, inclusive=False)

        return self.get_ts_after_index(index, inclusive=True)

    @stable(ts_listing)
    def get_ts_between_times(self, time1: float, time2: float, /, *,
                             inclusive: bool = False) -> 'TimeSeries':
        """
        Get a subset of the TimeSeries between two specified times.

        Parameters
        ----------
        time1, time2
            Times to look for in the TimeSeries' time vector.
        inclusive
            Optional. True to include the given time in the comparison.

        Example
        -------
        >>> ts = ktk.TimeSeries(time=np.arange(10)/10)
        >>> ts.time
        array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        >>> ts.get_ts_between_times(0.2, 0.5).time
        array([0.3, 0.4])

        >>> ts.get_ts_between_times(0.2, 0.5, inclusive=True).time
        array([0.2, 0.3, 0.4, 0.5])

        """
        sorted_times = np.sort([time1, time2])
        new_ts = self.get_ts_after_time(sorted_times[0],
                                        inclusive=inclusive)
        new_ts = new_ts.get_ts_before_time(sorted_times[1],
                                           inclusive=inclusive)
        return new_ts

    @stable(ts_listing)
    def get_ts_before_event(self, event_name: str,
                            event_occurrence: int = 0, /, *,
                            inclusive: bool = False) -> 'TimeSeries':
        """
        Get a subset of the TimeSeries before the specified event.

        Parameters
        ----------
        event_name
            Name of the event to look for in the events list.
        event_occurrence
            Optional. i_th occurence of the event to look for in the events
            list, starting at 0.
        inclusive
            Optional. True to include the given time in the comparison.

        Example
        -------
        >>> ts = ktk.TimeSeries(time=np.arange(10)/10)
        >>> ts.add_event(0.2, 'event')
        >>> ts.add_event(0.35, 'event')
        >>> ts.time
        array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        >>> ts.get_ts_before_event('event').time
        array([0. , 0.1])

        >>> ts.get_ts_before_event('event', inclusive=True).time
        array([0. , 0.1, 0.2])

        >>> ts.get_ts_before_event('event', 1).time
        array([0. , 0.1, 0.2, 0.3])

        >>> ts.get_ts_before_event('event', 1, inclusive=True).time
        array([0. , 0.1, 0.2, 0.3, 0.4])

        """
        time = self.get_event_time(event_name, event_occurrence)
        if inclusive:
            index = self.get_index_after_time(time, inclusive=True)
        else:
            index = self.get_index_before_time(time, inclusive=False)

        return self.get_ts_before_index(index, inclusive=True)

    @stable(ts_listing)
    def get_ts_after_event(self, event_name: str,
                           event_occurrence: int = 0, /, *,
                           inclusive: bool = False) -> 'TimeSeries':
        """
        Get a subset of the TimeSeries after the specified event.

        Parameters
        ----------
        event_name
            Name of the event to look for in the events list.
        event_occurrence
            Optional. i_th occurence of the event to look for in the events
            list, starting at 0.
        inclusive
            Optional. True to include the given event in the comparison.

        Example
        -------
        >>> ts = ktk.TimeSeries(time=np.arange(10)/10)
        >>> ts.add_event(0.2, 'event')
        >>> ts.add_event(0.35, 'event')
        >>> ts.time
        array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        >>> ts.get_ts_after_event('event').time
        array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        >>> ts.get_ts_after_event('event', inclusive=True).time
        array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        >>> ts.get_ts_after_event('event', 1).time
        array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        >>> ts.get_ts_after_event('event', 1, inclusive=True).time
        array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        """
        time = self.get_event_time(event_name, event_occurrence)
        if inclusive:
            index = self.get_index_before_time(time, inclusive=True)
        else:
            index = self.get_index_after_time(time, inclusive=False)

        return self.get_ts_after_index(index, inclusive=True)

    @stable(ts_listing)
    def get_ts_between_events(self, event_name1: str, event_name2: str,
                              event_occurrence1: int = 0,
                              event_occurrence2: int = 0,
                              /, *, inclusive: bool = False) -> 'TimeSeries':
        """
        Get a subset of the TimeSeries between two specified events.

        Parameters
        ----------
        event_name1, event_name2
            Name of the events to look for in the events list.
        event_occurrence1, event_occurrence2
            Optional. i_th occurence of the event to look for in the events
            list, starting at 0.
        inclusive
            Optional. True to include the given time in the comparison.

        Example
        -------
        >>> ts = ktk.TimeSeries(time=np.arange(10)/10)
        >>> ts.add_event(0.2, 'event')
        >>> ts.add_event(0.55, 'event')
        >>> ts.time
        array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        >>> ts.get_ts_between_events('event', 'event', 0, 1).time
        array([0.3, 0.4, 0.5])

        >>> ts.get_ts_between_events('event', 'event', 0, 1, \
                                     inclusive=True).time
        array([0.2, 0.3, 0.4, 0.5, 0.6])

        """
        ts = self.get_ts_after_event(event_name1, event_occurrence1,
                                     inclusive=inclusive)
        ts = ts.get_ts_before_event(event_name2, event_occurrence2,
                                    inclusive=inclusive)
        return ts

    @stable(ts_listing)
    def ui_get_ts_between_clicks(
            self,
            data_keys: Union[str, List[str]] = [], /, *,
            inclusive: bool = False) -> 'TimeSeries':
        """
        Get a subset of the TimeSeries between two mouse clicks.

        Parameters
        ----------
        data_keys
            Optional. String or list of strings corresponding to the signals
            to plot. See TimeSeries.plot() for more information.
        inclusive
            Optional. True to include the given time in the comparison.

        """
        fig = plt.figure()
        self.plot(data_keys)
        kineticstoolkit.gui.message('Click on both sides of the portion to keep.')
        plt.pause(0.001)  # Redraw
        points = plt.ginput(2)
        kineticstoolkit.gui.message('')
        times = [points[0][0], points[1][0]]
        plt.close(fig)
        return self.get_ts_between_times(min(times), max(times),
                                         inclusive=inclusive)

    @stable(ts_listing)
    def isnan(self, data_key: str, /) -> np.ndarray:
        """
        Return a boolean array of missing samples.

        Parameters
        ----------
        data_key
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

    @experimental(ts_listing)
    def fill_missing_samples(self, max_missing_samples: int, *,
                             method: str = 'linear') -> None:
        """
        Fill missing samples with the given method.

        Note
        ----
        The sample rate must be constant.

        Parameters
        ----------
        max_missing_samples
            Maximal number of consecutive missing samples to fill. Set to
            zero to fill all missing samples.
        method
            Optional. The interpolation method. This input may take any value
            supported by scipy.interpolate.interp1d, such as 'linear',
            'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'previous' or
            'next'.

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

    @stable(ts_listing)
    def shift(self, time: float, /) -> None:
        """
        Shift time and events.time.

        Parameters
        ----------
        time_shift
            Time to be added to time and events.time.

        """
        for event in self.events:
            event.time = event.time + time
        self.time = self.time + time

    @stable(ts_listing)
    def sync_event(self, event_name: str,
                   event_occurrence: int = 0, /) -> None:
        """
        Shift time and events.time so that event_name is the new time zero.

        Parameters
        ----------
        event_name
            Name of the event to sync on.
        event_occurrence
            Optional. Occurrence of the event to sync on, starting with 0.

        """
        self.shift(-self.get_event_time(event_name, event_occurrence))

    @stable(ts_listing)
    def trim_events(self) -> None:
        """
        Delete the events that are outside the TimeSeries' time vector.

        Example
        -------
        >>> ts = ktk.TimeSeries(time = np.arange(10))
        >>> ts.time
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        >>> ts.add_event(-2)
        >>> ts.add_event(0)
        >>> ts.add_event(5)
        >>> ts.add_event(9)
        >>> ts.add_event(10)
        >>> ts.events
        [[-2.0, 'event'], [0.0, 'event'], [5.0, 'event'], [9.0, 'event'], [10.0, 'event']]

        >>> ts.trim_events()
        >>> ts.events
        [[0.0, 'event'], [5.0, 'event'], [9.0, 'event']]

        """
        events = self.events
        self.events = []
        for event in events:
            if event.time >= self.time[0] and event.time <= self.time[-1]:
                self.add_event(event.time, event.name)

    @experimental(ts_listing)
    def ui_sync(self,
                data_keys: Union[str, List[str]] = [],
                ts2: Union['TimeSeries', None] = None,
                data_keys2: Union[str, List[str]] = [], /) -> None:
        """
        Synchronize one or two TimeSeries by shifting their time.

        If this method is called on only one TimeSeries, an interactive
        interface asks the user to click on the time to set to zero.

        If another TimeSeries is given, an interactive interface allows
        synchronizing both TimeSeries together.

        Parameters
        ----------
        data_keys
            Optional. The data keys to plot. If kept tempy, all data is
            plotted.
        ts2
            Optional. A second TimeSeries that contains both a recognizable
            zero-time event and a common event with the first TimeSeries.
        data_keys2
            Optional. The data keys from the second TimeSeries to plot. If
            kept tempy, all data is plotted.

        Note
        ----
        Although this method does not return anything, the TimeSeries ts2 is
        modified by this method.

        """
        fig = plt.figure('ktk.TimeSeries.ui_sync')

        if ts2 is None:
            # Synchronize ts1 only
            self.plot(data_keys)
            choice = kineticstoolkit.gui.button_dialog(
                'Please zoom on the time zero and press Next.',
                ['Cancel', 'Next'])
            if choice != 1:
                plt.close(fig)
                return

            kineticstoolkit.gui.message('Click on the sync event.')
            click = plt.ginput(1)
            kineticstoolkit.gui.message('')
            plt.close(fig)
            self.shift(-click[0][0])

        else:  # Sync two TimeSeries together

            finished = False
            # List of axes:
            axes = []  # type: List[Any]
            while finished is False:

                if len(axes) == 0:
                    axes.append(fig.add_subplot(2, 1, 1))
                    axes.append(fig.add_subplot(2, 1, 2, sharex=axes[0]))

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

                choice = kineticstoolkit.gui.button_dialog(
                    'Please select an option.',
                    choices=['Zero ts1 only, using ts1',
                             'Zero ts2 only, using ts2',
                             'Zero both TimeSeries, using ts1',
                             'Zero both TimeSeries, using ts2',
                             'Sync both TimeSeries on a common event',
                             'Finished'])

                if choice == 0:  # Zero ts1 only
                    kineticstoolkit.gui.message(
                        'Click on the time zero in ts1.')
                    click_1 = plt.ginput(1)
                    kineticstoolkit.gui.message('')

                    self.shift(-click_1[0][0])

                elif choice == 1:  # Zero ts2 only
                    kineticstoolkit.gui.message(
                        'Click on the time zero in ts2.')
                    click_1 = plt.ginput(1)
                    kineticstoolkit.gui.message('')

                    ts2.shift(-click_1[0][0])

                elif choice == 2:  # Zero ts1 and ts2 using ts1
                    kineticstoolkit.gui.message(
                        'Click on the time zero in ts1.')
                    click_1 = plt.ginput(1)
                    kineticstoolkit.gui.message('')

                    self.shift(-click_1[0][0])
                    ts2.shift(-click_1[0][0])

                elif choice == 3:  # Zero ts1 and ts2 using ts2
                    kineticstoolkit.gui.message(
                        'Click on the time zero in ts2.')
                    click_2 = plt.ginput(1)
                    kineticstoolkit.gui.message('')

                    self.shift(-click_2[0][0])
                    ts2.shift(-click_2[0][0])

                elif choice == 4:  # Sync on a common event
                    kineticstoolkit.gui.message(
                        'Click on the sync event in ts1.')
                    click_1 = plt.ginput(1)
                    kineticstoolkit.gui.message(
                        'Now click on the same event in ts2.')
                    click_2 = plt.ginput(1)
                    kineticstoolkit.gui.message('')

                    self.shift(-click_1[0][0])
                    ts2.shift(-click_2[0][0])

                elif choice == 5 or choice < -1:  # OK or closed figure, quit.
                    plt.close(fig)
                    finished = True

    @stable(ts_listing)
    def get_subset(self, data_keys: Union[str, List[str]]) -> 'TimeSeries':
        """
        Return a subset of the TimeSeries.

        This method returns a TimeSeries that contains only selected data
        keys. The corresponding data_info keys are copied in the new
        TimeSeries. All events are also copied in the new TimeSeries.

        Parameters
        ----------
        data_keys
            The data keys to extract from the timeseries.

        Returns
        -------
        TimeSeries
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

    @unstable(ts_listing)
    def resample(self,
                 new_time: np.ndarray,
                 kind: str = 'linear', *,
                 fill_value: Union[np.ndarray, str, None] = None) -> None:
        """
        Resample the TimeSeries.

        Parameters
        ----------
        new_time
            The new time vector to resample the TimeSeries to.
        kind
            Optional. The interpolation method. This input may take any value
            supported by scipy.interpolate.interp1d, such as 'linear',
            'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'previous',
            'next'. Additionally, kind can be 'pchip'.
        fill_value
            Optional. The fill value to use if new_time vector contains point
            outside the current TimeSeries' time vector. Use 'extrapolate' to
            extrapolate.

        Example
        --------
        >>> ts = ktk.TimeSeries(time=np.arange(6.))
        >>> ts.data['data'] = ts.time ** 2
        >>> ts.data['data_with_nans'] = ts.time ** 2
        >>> ts.data['data_with_nans'][3] = np.nan
        >>> ts.time
        array([0., 1., 2., 3., 4., 5.])
        >>> ts.data['data']
        array([ 0.,  1.,  4.,  9., 16., 25.])
        >>> ts.data['data_with_nans']
        array([ 0.,  1.,  4., nan, 16., 25.])

        >>> ts.resample(np.arange(0, 5.5, 0.5))

        >>> ts.time
        array([0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. ])

        >>> ts.data['data']
        array([ 0. ,  0.5,  1. ,  2.5,  4. ,  6.5,  9. , 12.5, 16. , 20.5, 25. ])

        >>> ts.data['data_with_nans']
        array([ 0. ,  0.5,  1. ,  2.5,  4. ,  nan,  nan,  nan, 16. , 20.5, 25. ])

        """
        if np.any(np.isnan(new_time)):
            raise ValueError('new_time must not contain nans')

        for key in self.data.keys():
            index = ~self.isnan(key)

            if sum(index) < 3:  # Only Nans, cannot interpolate.
                warnings.warn(
                    f'Warning: Almost only NaNs found in signal {key}.')
                # We generate an array of nans of the expected size.
                new_shape = [len(new_time)]
                for i in range(1, len(self.data[key].shape)):
                    new_shape.append(self.data[key].shape[i])
                self.data[key] = np.empty(new_shape)
                self.data[key][:] = np.nan

            else:  # Interpolate.

                # Express nans as a range of times to
                # remove from the final, interpolated timeseries
                nan_indexes = np.argwhere(~index)
                time_ranges_to_remove = []  # type: List[Tuple[int, int]]
                length = self.time.shape[0]
                for i in nan_indexes:
                    if i > 0 and i < length -1:
                        time_range = (self.time[i-1], self.time[i+1])
                    elif i == 0:
                        time_range = (-np.inf, self.time[i+1])
                    else:
                        time_range = (self.time[i-1], np.inf)
                    time_ranges_to_remove.append(time_range)

                if kind == 'pchip':
                    P = sp.interpolate.PchipInterpolator(
                        self.time[index],
                        self.data[key][index],
                        axis=0,
                        extrapolate=(
                            True if fill_value == 'extrapolate' else False))
                    self.data[key] = P(new_time)
                else:
                    f = sp.interpolate.interp1d(self.time[index],
                                                self.data[key][index],
                                                axis=0, fill_value=fill_value,
                                                kind=kind)
                    self.data[key] = f(new_time)

                # Put back nans
                for j in time_ranges_to_remove:
                    self.data[key][
                        (new_time > j[0]) & (new_time < j[1])] = np.nan

        self.time = new_time

    @experimental(ts_listing)
    def merge(self,
              ts: 'TimeSeries',
              data_keys: Union[str, List[str]] = [], /, *,
              resample: bool = False,
              overwrite: bool = True) -> None:
        """
        Merge another TimeSeries into the current TimeSeries.

        This method merges a TimeSeries into the current TimeSeries, copying
        the data, data_info and events.

        Parameters
        ----------
        ts
            The TimeSeries to merge into the current TimeSeries.
        data_keys
            Optional. The data keys to merge from ts. If left empty, all the
            data keys are merged.
        resample
            Optional. Set to True to resample the source TimeSeries, in case
            the time vectors are not matched. If the time vectors are not
            matched and resample is False, an exception is raised.
        overwrite
            Optional. If duplicates are found and overwrite is True, then the
            source (ts) overwrites the destination. Otherwise (overwrite is
            False), the duplicate data in ts is ignored.

        """
        ts = ts.copy()
        if len(data_keys) == 0:
            data_keys = list(ts.data.keys())
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
                for info_key in ts.data_info[key].keys():
                    self.add_data_info(key, info_key,
                                       ts.data_info[key][info_key])

        # Merge events
        for event in ts.events:
            self.events.append(event)
        self.sort_events()

    def __dir__(self):
        """Dir for the TimeSeries."""
        print('allo')
        return ts_listing


if __name__ == "__main__":
    import doctest
    import kineticstoolkit as ktk
    import numpy as np
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
