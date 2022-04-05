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
from kineticstoolkit.decorators import unstable, deprecated, directory
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd
import limitedinteraction as li
from dataclasses import dataclass

import warnings
from ast import literal_eval
from copy import deepcopy
from typing import Dict, List, Tuple, Any, Union, Optional

import kineticstoolkit as ktk  # For doctests


WINDOW_PLACEMENT = {'top': 50, 'right': 0}


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
                splitted[0] + '[' + splitted[1].replace(' ', '')
            )
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
    out = dict()  # type: Dict[str, np.ndarray]
    for key in dimensions:
        if len(dimensions[key]) == 0:
            out[key] = dataframe[key].to_numpy()
        else:
            highest_dims = np.max(np.array(dimensions[key]), axis=0)

            columns = [
                key + str(dim).replace(' ', '')
                for dim in sorted(dimensions[key])
            ]
            out[key] = dataframe[columns].to_numpy()
            out[key] = np.reshape(
                out[key], [n_samples] + (highest_dims + 1).tolist()
            )

    return out


def dict_of_arrays_to_dataframe(
        dict_of_arrays: Dict[str, np.ndarray]) -> pd.DataFrame:
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

        if original_data.shape[0] > 0:  # Not empty

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
            reshaped_indices = np.reshape(
                original_indices, (-1, reshaped_data_shape[1])
            )

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
                            reshaped_indices[i_indice, i_column]
                        )
                        if i_indice == n_indices - 1:
                            this_column_name += ']'
                        else:
                            this_column_name += ','

                column_names.append(this_column_name)

            df_data.columns = column_names

        else:  # empty data
            df_data = pd.DataFrame(columns=[the_key])

        # Merge this dataframe with the output dataframe
        df_out = pd.concat([df_out, df_data], axis=1)

    return df_out


@dataclass
class TimeSeriesEvent:
    """
    Define an event in a timeseries.

    This class is rarely used by itself, it is easier to use the `TimeSeries``
    methods to deal with events.

    Example
    -------
    >>> event = ktk.TimeSeriesEvent(time=1.5, name='event_name')
    >>> event
    TimeSeriesEvent(time=1.5, name='event_name')

    """

    time: float = 0.0
    name: str = 'event'

    def __lt__(self, other):
        """Define < operator."""
        return self.time < other.time

    def __le__(self, other):
        """Define <= operator."""
        return self.time <= other.time

    def __gt__(self, other):
        """Define > operator."""
        return self.time > other.time

    def __ge__(self, other):
        """Define >= operator."""
        return self.time >= other.time

    def _to_list(self):
        """
        Convert a TimeSeriesEvent to a list.

        Example
        -------
        >>> event = ktk.TimeSeriesEvent(time=1.5, name='event_name')
        >>> event._to_list()
        [1.5, 'event_name']

        """
        return [self.time, self.name]

    def _to_dict(self):
        """
        Convert a TimeSeriesEvent to a dict.

        Example
        -------
        >>> event = ktk.TimeSeriesEvent(time=1.5, name='event_name')
        >>> event._to_dict()
        {'Time': 1.5, 'Name': 'event_name'}

        """
        return {'Time': self.time, 'Name': self.name}


class TimeSeries:
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
        events: List[TimeSeriesEvent] = [],
    ):

        self.time = time.copy()
        self.data = data.copy()
        self.time_info = time_info.copy()
        self.data_info = data_info.copy()
        self.events = events.copy()

    def __dir__(self):
        """Generate the class directory."""
        return directory(TimeSeries.__dict__)

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
        """Generate the class representation."""
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

        for data in [self.data, ts.data]:
            for one_data in data:
                try:
                    if not np.isclose(
                        self.data[one_data], ts.data[one_data], rtol=1e-15
                    ).all():
                        print(f'{one_data} is not equal')
                        return False
                except KeyError:
                    print(f'{one_data} is missing in one of the TimeSeries')
                    return False
                except ValueError:
                    print(
                        f'{one_data} does not have the same size in both '
                        'TimeSeries'
                    )
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

    def from_dataframe(dataframe: pd.DataFrame, /) -> 'TimeSeries':
        """
        Create a new TimeSeries from a Pandas Dataframe.

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

    def add_data_info(
            self,
            data_key: str,
            info_key: str,
            value: Any,
            *,
            in_place: bool = False) -> 'TimeSeries':
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
        in_place
            Optional. True to modify the original TimeSeries. False
            to return a modified copy of the TimeSeries while leaving the
            original TimeSeries intact. The default is False.

        Returns
        -------
        TimeSeries
            The TimeSeries with the added data info.

        Example
        -------
        We create a TimeSeries with some data info:

        >>> ts = ktk.TimeSeries()
        >>> ts = ts.add_data_info('Forces', 'Unit', 'N')
        >>> ts = ts.add_data_info('Marker1', 'Color', [43, 2, 255])

        Let's see the result:

        >>> ts.data_info['Forces']
        {'Unit': 'N'}

        >>> ts.data_info['Marker1']
        {'Color': [43, 2, 255]}

        """
        ts = self if in_place else self.copy()
        try:
            ts.data_info[data_key][info_key] = value
        except KeyError:
            ts.data_info[data_key] = {info_key: value}
        return ts

    def remove_data_info(
            self,
            data_key: str,
            info_key: str,
            *,
            in_place: bool = False) -> 'TimeSeries':
        """
        Remove metadata from a TimeSeries' data.

        Parameters
        ----------
        data_key
            The data key the info corresponds to.
        info_key
            The key of the info dict.
        in_place
            Optional. True to modify the original TimeSeries. False
            to return a modified copy of the TimeSeries while leaving the
            original TimeSeries intact. The default is False.

        Returns
        -------
        TimeSeries
            The TimeSeries with the removed data info.

        Note
        ----
        No warning or exception is raised if the data key does not exist.

        Example
        -------
        >>> ts = ktk.TimeSeries()
        >>> ts = ts.add_data_info('Forces', 'Unit', 'N')
        >>> ts.data_info['Forces']
        {'Unit': 'N'}

        >>> ts = ts.remove_data_info('Forces', 'Unit')
        >>> ts.data_info['Forces']
        {}

        """
        ts = self if in_place else self.copy()
        try:
            ts.data_info[data_key].pop(info_key)
        except KeyError:
            pass
        return ts

    def rename_data(
            self,
            old_data_key: str,
            new_data_key: str,
            *,
            in_place: bool = False) -> 'TimeSeries':
        """
        Rename a key in data and data_info.

        Parameters
        ----------
        old_data_key
            Name of the current data key.
        new_data_key
            New name of the data key.
        in_place
            Optional. True to modify and return the original TimeSeries. False
            to return a modified copy of the TimeSeries while leaving the
            original TimeSeries intact. The default is False.

        Returns
        -------
        TimeSeries
            The TimeSeries with the renamed data.

        Example
        -------
        >>> ts = ktk.TimeSeries()
        >>> ts.data['test'] = np.arange(10)
        >>> ts = ts.add_data_info('test', 'Unit', 'm')

        >>> ts.data
        {'test': array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}

        >>> ts.data_info
        {'test': {'Unit': 'm'}}

        >>> ts = ts.rename_data('test', 'signal')

        >>> ts.data
        {'signal': array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}

        >>> ts.data_info
        {'signal': {'Unit': 'm'}}

        """
        ts = self if in_place else self.copy()
        try:
            ts.data[new_data_key] = ts.data.pop(old_data_key)
        except KeyError:
            pass
        try:
            ts.data_info[new_data_key] = ts.data_info.pop(old_data_key)
        except KeyError:
            pass
        return ts

    def remove_data(
            self,
            data_key: str,
            *,
            in_place: bool = False) -> 'TimeSeries':
        """
        Remove a data key and its associated metadata.

        Parameters
        ----------
        data_key
            Name of the data key.
        in_place
            Optional. True to modify and return the original TimeSeries. False
            to return a modified copy of the TimeSeries while leaving the
            original TimeSeries intact. The default is False.

        Returns
        -------
        TimeSeries
            The TimeSeries with the removed data.

        Note
        ----
        No warning or exception is raised if the data key does not exist.

        Example
        -------
        >>> # Prepare a test TimeSeries with data 'test'
        >>> ts = ktk.TimeSeries()
        >>> ts.data['test'] = np.arange(10)
        >>> ts = ts.add_data_info('test', 'Unit', 'm')

        >>> ts.data
        {'test': array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}

        >>> ts.data_info
        {'test': {'Unit': 'm'}}

        >>> # Now remove data 'test'
        >>> ts = ts.remove_data('test')

        >>> ts.data
        {}

        >>> ts.data_info
        {}

        """
        ts = self if in_place else self.copy()
        try:
            ts.data.pop(data_key)
        except KeyError:
            pass
        try:
            ts.data_info.pop(data_key)
        except KeyError:
            pass
        return ts

    def add_event(
            self,
            time: float,
            name: str = 'event',
            *,
            in_place=False) -> 'TimeSeries':
        """
        Add an event to the TimeSeries.

        Parameters
        ----------
        time
            The time of the event, in the same unit as `time_info['Unit']`.
        name
            Optional. The name of the event. The default is 'event'.
        in_place
            Optional. True to modify and return the original TimeSeries. False
            to return a modified copy of the TimeSeries while leaving the
            original TimeSeries intact. The default is False.

        Returns
        -------
        TimeSeries
            A copy of the TimeSeries with the added event.

        Example
        -------
        >>> ts = ktk.TimeSeries()
        >>> ts = ts.add_event(5.5, 'event1')
        >>> ts = ts.add_event(10.8, 'event2')
        >>> ts = ts.add_event(2.3, 'event2')

        >>> ts.events
        [TimeSeriesEvent(time=5.5, name='event1'),
         TimeSeriesEvent(time=10.8, name='event2'),
         TimeSeriesEvent(time=2.3, name='event2')]

        """
        ts = self if in_place else self.copy()
        ts.events.append(TimeSeriesEvent(time, name))
        return ts

    def rename_event(
            self,
            old_name: str,
            new_name: str,
            occurrence: Optional[int] = None,
            *,
            in_place: bool = False) -> 'TimeSeries':
        """
        Rename an event occurrence or all events of a same name.

        Parameters
        ----------
        old_name
            Name of the event to look for in the events list.
        new_name
            New event name
        occurrence
            Optional. i_th occurence of the event to look for in the events
            list, starting at 0, where the occurrences are sorted in time.
            If None (default), all occurences of this event name are renamed.
        in_place
            Optional. True to modify and return the original TimeSeries. False
            to return a modified copy of the TimeSeries while leaving the
            original TimeSeries intact. The default is False.

        Returns
        -------
        TimeSeries
            The TimeSeries with the renamed event.

        Example
        -------
        >>> ts = ktk.TimeSeries()
        >>> ts = ts.add_event(5.5, 'event1')
        >>> ts = ts.add_event(10.8, 'event2')
        >>> ts = ts.add_event(2.3, 'event2')

        >>> ts.events
        [TimeSeriesEvent(time=5.5, name='event1'),
         TimeSeriesEvent(time=10.8, name='event2'),
         TimeSeriesEvent(time=2.3, name='event2')]

        >>> ts = ts.rename_event('event2', 'event3')
        >>> ts.events
        [TimeSeriesEvent(time=5.5, name='event1'),
         TimeSeriesEvent(time=10.8, name='event3'),
         TimeSeriesEvent(time=2.3, name='event3')]

        >>> ts = ts.rename_event('event3', 'event4', 0)
        >>> ts.events
        [TimeSeriesEvent(time=5.5, name='event1'),
         TimeSeriesEvent(time=10.8, name='event3'),
         TimeSeriesEvent(time=2.3, name='event4')]

        """
        ts = self if in_place else self.copy()

        if old_name == new_name:
            return ts  # Nothing to do.

        if occurrence is None:
            # Rename every occurrence of this event
            index = ts.get_event_index(old_name, 0)
            while ~np.isnan(index):
                ts.events[index].name = new_name  # type: ignore
                index = ts.get_event_index(old_name, 0)

        else:
            index = ts.get_event_index(old_name, occurrence)
            if ~np.isnan(index):
                ts.events[int(index)].name = new_name  # type: ignore
            else:
                warnings.warn(
                    f"The occurrence {occurrence} of event "
                    f"{old_name} could not be found."
                )
        return ts

    def remove_event(
            self,
            name: str,
            occurrence: Optional[int] = None,
            *,
            in_place: bool = False) -> 'TimeSeries':
        """
        Remove an event occurrence or all events of a same name.

        Parameters
        ----------
        name
            Name of the event to look for in the events list.
        occurrence
            Optional. i_th occurence of the event to look for in the events
            list, starting at 0, where the occurrences are sorted in time.
            If None (default), all occurences of this event name or removed.
        in_place
            Optional. True to modify and return the original TimeSeries. False
            to return a modified copy of the TimeSeries while leaving the
            original TimeSeries intact. The default is False.

        Returns
        -------
        TimeSeries
            The TimeSeries with the removed event.

        Example
        -------
        >>> # Instanciate a timeseries with some events
        >>> ts = ktk.TimeSeries()
        >>> ts = ts.add_event(5.5, 'event1')
        >>> ts = ts.add_event(10.8, 'event2')
        >>> ts = ts.add_event(2.3, 'event2')

        >>> ts.events
        [TimeSeriesEvent(time=5.5, name='event1'),
         TimeSeriesEvent(time=10.8, name='event2'),
         TimeSeriesEvent(time=2.3, name='event2')]

        >>> ts = ts.remove_event('event1')
        >>> ts.events
        [TimeSeriesEvent(time=10.8, name='event2'),
         TimeSeriesEvent(time=2.3, name='event2')]

        >>> ts = ts.remove_event('event2', 1)
        >>> ts.events
        [TimeSeriesEvent(time=2.3, name='event2')]

        """
        ts = self if in_place else self.copy()

        if occurrence is None:  # Remove all occurrences
            event_index = ts.get_event_index(name, 0)
            while ~np.isnan(event_index):
                ts.events.pop(event_index)  # type: ignore
                event_index = ts.get_event_index(name, 0)

        else:  # Remove only the specified occurrence
            event_index = ts.get_event_index(name, occurrence)
            if ~np.isnan(event_index):
                ts.events.pop(event_index)  # type: ignore
            else:
                warnings.warn(
                    f"The occurrence {occurrence} of event "
                    f"{name} could not be found."
                )
        return ts

    def ui_edit_events(
            self,
            name: Union[str, List[str]] = [],
            data_keys: Union[str, List[str]] = []
    ) -> 'TimeSeries':  # pragma: no cover
        """
        Edit events interactively.

        Warning
        -------
        This function, which has been introduced in 0.6, is still experimental
        and may change signature or behaviour in the future.

        Parameters
        ----------
        name
            Optional. The name of the event(s) to add. May be a string
            or a list of strings. Event names can also be defined
            interactively.
        data_keys
            Optional. A signal name of list of signal name to be plotted,
            similar to the data_keys argument of ktk.TimeSeries.plot().

        Returns
        -------
        TimeSeries
            The original TimeSeries with the modified events. If
            the operation was cancelled by the user, this is a pure copy of
            the original TimeSeries.

        Note
        ----
        Matplotlib must be in interactive mode for this function to work.

        """

        def add_this_event(ts: 'TimeSeries', name: str) -> 'TimeSeries':
            kineticstoolkit.gui.message(
                "Place the event on the figure.", **WINDOW_PLACEMENT
            )
            this_time = plt.ginput(1)[0][0]
            ts = ts.add_event(this_time, name)
            kineticstoolkit.gui.message("")
            return ts

        def get_event_index(ts: 'TimeSeries') -> int:
            kineticstoolkit.gui.message(
                "Select an event on the figure.", **WINDOW_PLACEMENT
            )
            this_time = plt.ginput(1)[0][0]
            event_times = np.array([event.time for event in ts.events])
            kineticstoolkit.gui.message("")
            return int(np.argmin(np.abs(event_times - this_time)))

        # Set Matplotlib interactive mode
        isinteractive = plt.isinteractive()
        plt.ion()

        ts = self.copy()

        if isinstance(name, str):
            event_names = [name]
        else:
            event_names = deepcopy(name)

        fig = plt.figure()
        ts.plot(data_keys, _raise_on_no_data=True)

        while True:
            # Populate the choices to the user
            choices = [f"Add '{s}'" for s in event_names]

            choice_index = {}
            choice_index['add'] = len(choices)
            if len(event_names) == 0:
                choices.append("Add event")
            else:
                choices.append("Add event with another name")

            if len(ts.events) > 0:
                choice_index['remove'] = len(choices)
                choices.append("Remove event")

            if len(ts.events) > 0:
                choice_index['remove_all'] = len(choices)
                choices.append("Remove all events")

                choice_index['move'] = len(choices)
                choices.append("Move event")

            choice_index['close'] = len(choices)
            choices.append("Save and close")

            choice_index['cancel'] = len(choices)
            choices.append("Cancel")

            # Show the button dialog
            choice = kineticstoolkit.gui.button_dialog(
                "Move and zoom on the figure,\n"
                "then select an option below.",
                choices,
                **WINDOW_PLACEMENT,
            )

            # Execute
            if choice < choice_index['add']:
                ts = add_this_event(ts, event_names[choice])

            elif choice == choice_index['add']:
                event_names.append(
                    li.input_dialog(
                        "Please enter the event name:", **WINDOW_PLACEMENT
                    )
                )
                # Add this event name to the list of recently added events
                if len(event_names) > 5:
                    event_names = event_names[-5:]

                # Add the event
                ts = add_this_event(ts, event_names[-1])

            elif ('remove' in choice_index) and (
                choice == choice_index['remove']
            ):
                event_index = get_event_index(ts)
                try:
                    ts.events.pop(event_index)
                except IndexError:
                    li.button_dialog(
                        "No event was removed.",
                        choices=['OK'],
                        icon='error',
                        **WINDOW_PLACEMENT,
                    )

            elif ('remove_all' in choice_index) and (
                choice == choice_index['remove_all']
            ):
                if (
                    li.button_dialog(
                        "Do you really want to remove all events from this "
                        "TimeSeries?",
                        ['Yes, remove all events', 'No'],
                        icon='alert',
                        **WINDOW_PLACEMENT,
                    )
                    == 0
                ):
                    ts.events = []

            elif ('move' in choice_index) and (choice == choice_index['move']):
                event_index = get_event_index(ts)
                event_name = ts.events[event_index].name
                try:
                    ts.events.pop(event_index)
                    ts = add_this_event(ts, event_name)
                except IndexError:
                    li.button_dialog(
                        "Could not move this event.",
                        choices=['OK'],
                        icon='error',
                        **WINDOW_PLACEMENT,
                    )

            elif ('close' in choice_index) and (
                choice == choice_index['close']
            ):
                plt.close(fig)
                if not isinteractive:
                    plt.ioff()
                return ts

            elif (choice == -1) or (
                ('cancel' in choice_index)
                and (choice == choice_index['cancel'])
            ):
                plt.close(fig)
                if not isinteractive:
                    plt.ioff()
                return self.copy()

            # Refresh
            ts.sort_events(unique=False, in_place=True)
            axes = plt.axis()
            plt.cla()
            ts.plot(data_keys, _raise_on_no_data=True)
            plt.axis(axes)

    @deprecated(
        since="0.5",
        until="0.7",
        details=(
            "Please use the ui_edit_events method, which is the "
            "more powerful replacement."
        ),
    )
    def ui_add_event(
        self,
        name: str = 'event',
        data_key: Union[str, List[str]] = [],
        *,
        multiple_events: bool = False,
    ) -> 'TimeSeries':
        """
        Add one or many events interactively to the TimeSeries.

        Parameters
        ----------
        name
            Optional. The name of the event.
        data_key
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
        return self.ui_edit_events(name, data_key)

    def sort_events(
            self,
            *,
            unique: bool = True,
            in_place: bool = False) -> 'TimeSeries':
        """
        Sorts the TimeSeries' events from the earliest to the latest.

        Parameters
        ----------
        unique
            Optional. True to make events unique so that no two events can
            have both the same name and the same time.
        in_place
            Optional. True to modify and return the original TimeSeries. False
            to return a modified copy of the TimeSeries while leaving the
            original TimeSeries intact. The default is False.

        Returns
        -------
        TimeSeries
            The TimeSeries with the sorted events.

        Example
        -------
        >>> ts = ktk.TimeSeries(time=np.arange(100)/10)
        >>> ts = ts.add_event(2, 'two')
        >>> ts = ts.add_event(1, 'one')
        >>> ts = ts.add_event(3, 'three')
        >>> ts = ts.add_event(3, 'three')

        >>> ts.events
        [TimeSeriesEvent(time=2, name='two'),
         TimeSeriesEvent(time=1, name='one'),
         TimeSeriesEvent(time=3, name='three'),
         TimeSeriesEvent(time=3, name='three')]

        >>> ts = ts.sort_events(unique=False)
        >>> ts.events
        [TimeSeriesEvent(time=1, name='one'),
         TimeSeriesEvent(time=2, name='two'),
         TimeSeriesEvent(time=3, name='three'),
         TimeSeriesEvent(time=3, name='three')]

        >>> ts = ts.sort_events()
        >>> ts.events
        [TimeSeriesEvent(time=1, name='one'),
         TimeSeriesEvent(time=2, name='two'),
         TimeSeriesEvent(time=3, name='three')]

        """
        ts = self if in_place else self.copy()

        ts.events = sorted(ts.events)

        if unique is True:
            for i in range(len(ts.events) - 1, 0, -1):
                if (ts.events[i].time == ts.events[i - 1].time) and (
                    ts.events[i].name == ts.events[i - 1].name
                ):
                    ts.events.pop(i)

        return ts

    def copy(
        self,
        *,
        copy_time=True,
        copy_data=True,
        copy_time_info=True,
        copy_data_info=True,
        copy_events=True,
    ) -> 'TimeSeries':
        """
        Deep copy of a TimeSeries.

        Parameters
        ----------
        copy_data
            Optional. True to copy data to the new TimeSeries,
            False to keep the data attribute empty. Default is True.
        copy_time_info
            Optional. True to copy time_info to the new TimeSeries,
            False to keep the time_info attribute empty. Default is True.
        copy_data_info
            Optional. True to copy data_into to the new TimeSeries,
            False to keep the data_info attribute empty. Default is True.
        copy_events
            Optional. True to copy events to the new TimeSeries,
            False to keep the events attribute empty. Default is True.

        Returns
        -------
        TimeSeries
            A deep copy of the TimeSeries.

        """
        if copy_data and copy_time_info and copy_data_info and copy_events:
            # General case
            return deepcopy(self)
        else:
            # Specific cases
            ts = ktk.TimeSeries()
            if copy_time:
                ts.time = deepcopy(self.time)
            if copy_data:
                ts.data = deepcopy(self.data)
            if copy_time_info:
                ts.time_info = deepcopy(self.time_info)
            if copy_data_info:
                ts.data_info = deepcopy(self.data_info)
            if copy_events:
                ts.events = deepcopy(self.events)
            return ts

    def plot(
        self,
        data_keys: Union[str, List[str]] = [],
        *args,
        event_names: bool = True,
        legend: bool = True,
        **kwargs,
    ) -> None:
        """
        Plot the TimeSeries in the current matplotlib figure.

        Parameters
        ----------
        data_keys
            The data keys to plot. If left empty, all data is plotted.
        event_names
            Optional. True to plot the event names on top of the event lines.
        legend
            Optional. True to plot a legend, False otherwise.

        Example
        -------
        If a TimeSeries `ts`' data attribute as keys 'Forces', 'Moments' and
        'Angle', then:

            ts.plot()

        plots all data (Forces, Moments and Angle), whereas:

            ts.plot(['Forces', 'Moments'])

        plots only the forces and moments, without plotting the angle.

        Additional positional and keyboard arguments are passed to
        matplotlib's pyplot.plot function:

            ts.plot(['Forces'], '--')

        plots the forces using a dashed line style.

        """
        # Private argument _raise_on_no_data: Raise a ValueError instead of
        # warning when no data is available to plot.
        if '_raise_on_no_data' in kwargs:
            raise_on_no_data = kwargs.pop('_raise_on_no_data')
        else:
            raise_on_no_data = False

        if data_keys is None or len(data_keys) == 0:
            # Plot all
            ts = self.copy()
        else:
            ts = self.get_subset(data_keys)

        if len(ts.data) == 0:
            if raise_on_no_data:
                raise ValueError("No data available to plot.")
            else:
                warnings.warn("No data available to plot.")
            return

        # Sort events to help finding each event's occurrence
        ts.sort_events(unique=False)

        df = ts.to_dataframe()
        labels = df.columns.to_list()

        axes = plt.gca()
        axes.set_prop_cycle(
            mpl.cycler(linewidth=[1, 2, 3, 4])
            * mpl.cycler(linestyle=['-', '--', '-.', ':'])
            * mpl.cycler(
                color=['r', 'g', 'b', 'c', 'm', 'y', 'k', 'tab:orange']
            )
        )

        # Plot the curves
        for i_label, label in enumerate(labels):
            axes.plot(
                df.index.to_numpy(),
                df[label].to_numpy(),
                *args,
                label=label,
                **kwargs,
            )

        # Add labels
        plt.xlabel('Time (' + ts.time_info['Unit'] + ')')

        # Make unique list of units
        unit_set = set()
        for data in ts.data_info:
            for info in ts.data_info[data]:
                if info == 'Unit':
                    unit_set.add(ts.data_info[data][info])
        # Plot this list
        unit_str = ''
        for unit in unit_set:
            if len(unit_str) > 0:
                unit_str += ', '
            unit_str += unit

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

                    plt.text(
                        event.time,
                        max_y,
                        name,
                        rotation='vertical',
                        horizontalalignment='center',
                        fontsize='small',
                    )

        if legend:
            if len(labels) < 20:
                legend_location = 'best'
            else:
                legend_location = 'upper right'

            axes.legend(
                loc=legend_location, ncol=1 + int(len(labels) / 40)
            )  # Max 40 items per line

    def get_index_at_time(self, time: float) -> int:
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
        return int(np.argmin(np.abs(self.time - float(time))))

    def get_index_before_time(
        self, time: float, *, inclusive: bool = False
    ) -> Union[int, float]:
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

        >>> ts.get_index_before_time(1.1)
        2

        >>> ts.get_index_before_time(1.1, inclusive=True)
        3

        >>> ts.get_index_before_time(0)
        nan

        >>> ts.get_index_before_time(0, inclusive=True)
        0

        """
        # Edge case
        try:
            if inclusive and time == self.time[0]:
                return 0
        except IndexError:  # If time was empty
            return np.nan

        # Other cases
        diff = float(time) - self.time
        diff[diff <= 0] = np.nan

        if np.all(np.isnan(diff)):  # All nans
            return np.nan

        index = np.nanargmin(diff)

        if inclusive and self.time[index] < time:
            index += 1

        if index < self.time.shape[0]:
            return int(index)
        else:
            return np.nan

    def get_index_after_time(
        self, time: float, *, inclusive: bool = False
    ) -> Union[int, float]:
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

        >>> ts.get_index_after_time(0.9, inclusive=True)
        1

        >>> ts.get_index_after_time(1)
        3

        >>> ts.get_index_after_time(1, inclusive=True)
        2

        >>> ts.get_index_after_time(2)
        nan

        >>> ts.get_index_after_time(2, inclusive=True)
        4

        """
        # Edge case
        try:
            if inclusive and time == self.time[-1]:
                return self.time.shape[0] - 1
        except IndexError:  # If time was empty
            return np.nan

        # Other cases
        diff = self.time - float(time)
        diff[diff <= 0] = np.nan

        if np.all(np.isnan(diff)):  # All nans
            return np.nan

        index = np.nanargmin(diff)

        if inclusive and self.time[index] > time:
            index -= 1

        if index >= 0:
            return int(index)
        else:
            return np.nan

    def get_event_index(self, name: str, occurrence: int) -> Union[int, float]:
        """
        Get the index of a given occurrence of an event name.

        Parameters
        ----------
        name
            Name of the event to look for in the events list.
        occurrence
            i_th occurence of the event to look for in the events
            list, starting at 0, where the occurrences are sorted in time.

        Returns
        -------
        int
            The index of the event or np.nan if no event found.

        """
        occurrence = int(occurrence)

        if occurrence < 0:
            raise ValueError('occurrence must be positive')

        # Make a list of event times, with NaNs for events whose name doesn't
        # match what we're looking for.
        event_times = [
            event.time if event.name == name else np.nan
            for event in self.events
        ]

        # Sort in time
        event_indexes = np.argsort(event_times)

        # Remove indexes that correspond to nan time (wrong events)
        clean_event_indexes = [
            index if ~np.isnan(event_times[index]) else np.nan
            for index in event_indexes
        ]

        # Get the event occurrence
        try:
            return clean_event_indexes[occurrence]
        except IndexError:
            return np.nan

    def get_event_time(self, name: str, occurrence: int = 0) -> float:
        """
        Get the time of the specified event.

        Parameters
        ----------
        name
            Name of the event to look for in the events list.
        occurrence
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
        >>> ts = ts.add_event(5.5, 'event1')
        >>> ts = ts.add_event(10.8, 'event2')
        >>> ts = ts.add_event(2.3, 'event2')

        >>> ts.get_event_time('event1')
        5.5

        >>> ts.get_event_time('event2', 0)
        2.3

        >>> ts.get_event_time('event2', 1)
        10.8

        """
        event_index = self.get_event_index(name, occurrence)
        if ~np.isnan(event_index):
            return self.events[event_index].time  # type: ignore
        else:
            return np.nan

    def get_ts_at_time(self, time: float) -> 'TimeSeries':
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

    def get_ts_at_event(self, name: str, occurrence: int = 0) -> 'TimeSeries':
        """
        Get a one-data subset of the TimeSeries at the event's nearest time.

        Parameters
        ----------
        name
            Name of the event to look for in the events list.
        occurrence
            Optional. i_th occurence of the event to look for in the events
            list, starting at 0.

        Returns
        -------
        TimeSeries
            A TimeSeries of length 1, at the event's nearest time.

        """
        time = self.get_event_time(name, occurrence)
        return self.get_ts_at_time(time)

    def get_ts_before_index(
        self, index: int, *, inclusive: bool = False
    ) -> 'TimeSeries':
        """
        Get a subset of the TimeSeries before the specified time index.

        Parameters
        ----------
        index
            Time index
        inclusive
            Optional. True to include the given time index.

        Returns
        -------
        TimeSeries
            A new TimeSeries that fulfils the specified conditions.

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
        out_ts = self.copy(copy_data=False, copy_time=False)

        if index < 0:
            index += len(self.time)

        if np.isnan(index):
            index_range = range(0)
        else:
            if inclusive:
                index_range = range(index + 1)
            else:
                index_range = range(index)

        out_ts.time = self.time[index_range]
        for the_data in self.data:
            out_ts.data[the_data] = self.data[the_data][index_range]
        return out_ts

    def get_ts_after_index(
        self, index: int, *, inclusive: bool = False
    ) -> 'TimeSeries':
        """
        Get a subset of the TimeSeries after the specified time index.

        Parameters
        ----------
        index
            Time index
        inclusive
            Optional. True to include the given time index.

        Returns
        -------
        TimeSeries
            A new TimeSeries that fulfils the specified conditions.

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
        out_ts = self.copy(copy_data=False, copy_time=False)

        if index < 0:
            index += len(self.time)

        if inclusive:
            index_range = range(index, len(self.time))
        else:
            index_range = range(index + 1, len(self.time))

        out_ts.time = self.time[index_range]
        for the_data in self.data:
            out_ts.data[the_data] = self.data[the_data][index_range]
        return out_ts

    def get_ts_between_indexes(
        self, index1: int, index2: int, *, inclusive: bool = False
    ) -> 'TimeSeries':
        """
        Get a subset of the TimeSeries before two specified time indexes.

        Parameters
        ----------
        index1, index2
            Time indexes
        inclusive
            Optional. True to include the given time indexes.

        Returns
        -------
        TimeSeries
            A new TimeSeries that fulfils the specified conditions.

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
        out_ts = self.copy(copy_time=False, copy_data=False)
        if np.isnan(index1) or np.isnan(index2):
            index_range = range(0)
        else:
            if inclusive:
                index_range = range(index1, index2 + 1)
            else:
                index_range = range(index1 + 1, index2)

        out_ts.time = self.time[index_range]
        for the_data in self.data.keys():
            out_ts.data[the_data] = self.data[the_data][index_range]
        return out_ts

    def get_ts_before_time(
        self, time: float, *, inclusive: bool = False
    ) -> 'TimeSeries':
        """
        Get a subset of the TimeSeries before the specified time.

        Parameters
        ----------
        time
            Time to look for in the TimeSeries' time vector.
        inclusive
            Optional. True to include the given time in the comparison.

        Returns
        -------
        TimeSeries
            A new TimeSeries that fulfils the specified conditions.

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
        # Edge case
        if len(self.time) == 0 or time > self.time[-1]:
            return self.copy()

        # Other cases
        index = self.get_index_before_time(time, inclusive=inclusive)
        if ~np.isnan(index):
            return self.get_ts_before_index(index, inclusive=True)  # type: ignore # noqa
        else:
            return self.get_ts_before_index(0, inclusive=False)

    def get_ts_after_time(
        self, time: float, *, inclusive: bool = False
    ) -> 'TimeSeries':
        """
        Get a subset of the TimeSeries after the specified time.

        Parameters
        ----------
        time
            Time to look for in the TimeSeries' time vector.
        inclusive
            Optional. True to include the given time in the comparison.

        Returns
        -------
        TimeSeries
            A new TimeSeries that fulfils the specified conditions.

        Example
        -------
        >>> ts = ktk.TimeSeries(time=np.arange(10)/10)
        >>> ts.time
        array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        >>> ts.get_ts_after_time(0.3).time
        array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        >>> ts.get_ts_after_time(0.3, inclusive=True).time
        array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        >>> ts.get_ts_after_time(0.25, inclusive=True).time
        array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        """
        # Edge case
        if len(self.time) == 0 or time < self.time[0]:
            return self.copy()

        # Other cases
        index = self.get_index_after_time(time, inclusive=inclusive)
        if ~np.isnan(index):
            return self.get_ts_after_index(index, inclusive=True)  # type: ignore # noqa
        else:
            return self.get_ts_after_index(-1, inclusive=False)

    def get_ts_between_times(
        self, time1: float, time2: float, *, inclusive: bool = False
    ) -> 'TimeSeries':
        """
        Get a subset of the TimeSeries between two specified times.

        Parameters
        ----------
        time1, time2
            Times to look for in the TimeSeries' time vector.
        inclusive
            Optional. True to include the given time in the comparison.

        Returns
        -------
        TimeSeries
            A new TimeSeries that fulfils the specified conditions.

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
        new_ts = self.get_ts_after_time(sorted_times[0], inclusive=inclusive)
        new_ts = new_ts.get_ts_before_time(
            sorted_times[1], inclusive=inclusive
        )
        return new_ts

    def get_ts_before_event(
        self, name: str, occurrence: int = 0, *, inclusive: bool = False
    ) -> 'TimeSeries':
        """
        Get a subset of the TimeSeries before the specified event.

        Parameters
        ----------
        name
            Name of the event to look for in the events list.
        occurrence
            Optional. i_th occurence of the event to look for in the events
            list, starting at 0.
        inclusive
            Optional. True to include the given time in the comparison.

        Returns
        -------
        TimeSeries
            A new TimeSeries that fulfils the specified conditions.

        Example
        -------
        >>> ts = ktk.TimeSeries(time=np.arange(10)/10)
        >>> ts = ts.add_event(0.2, 'event')
        >>> ts = ts.add_event(0.35, 'event')
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
        time = self.get_event_time(name, occurrence)
        return self.get_ts_before_time(time, inclusive=inclusive)

    def get_ts_after_event(
        self, name: str, occurrence: int = 0, *, inclusive: bool = False
    ) -> 'TimeSeries':
        """
        Get a subset of the TimeSeries after the specified event.

        Parameters
        ----------
        name
            Name of the event to look for in the events list.
        occurrence
            Optional. i_th occurence of the event to look for in the events
            list, starting at 0.
        inclusive
            Optional. True to include the given event in the comparison.

        Returns
        -------
        TimeSeries
            A new TimeSeries that fulfils the specified conditions.

        Example
        -------
        >>> ts = ktk.TimeSeries(time=np.arange(10)/10)
        >>> ts = ts.add_event(0.2, 'event')
        >>> ts = ts.add_event(0.35, 'event')
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
        time = self.get_event_time(name, occurrence)
        return self.get_ts_after_time(time, inclusive=inclusive)

    def get_ts_between_events(
        self,
        name1: str,
        name2: str,
        occurrence1: int = 0,
        occurrence2: int = 0,
        *,
        inclusive: bool = False,
    ) -> 'TimeSeries':
        """
        Get a subset of the TimeSeries between two specified events.

        Parameters
        ----------
        name1, name2
            Name of the events to look for in the events list.
        occurrence1, occurrence2
            Optional. i_th occurence of the event to look for in the events
            list, starting at 0.
        inclusive
            Optional. True to include the given time in the comparison.

        Returns
        -------
        TimeSeries
            A new TimeSeries that fulfils the specified conditions.

        Example
        -------
        >>> ts = ktk.TimeSeries(time=np.arange(10)/10)
        >>> ts = ts.add_event(0.2, 'event')
        >>> ts = ts.add_event(0.55, 'event')
        >>> ts.time
        array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        >>> ts.get_ts_between_events('event', 'event', 0, 1).time
        array([0.3, 0.4, 0.5])

        >>> ts.get_ts_between_events('event', 'event', 0, 1, \
                                     inclusive=True).time
        array([0.2, 0.3, 0.4, 0.5, 0.6])

        """
        ts = self.get_ts_after_event(name1, occurrence1, inclusive=inclusive)
        ts = ts.get_ts_before_event(name2, occurrence2, inclusive=inclusive)
        return ts

    def ui_get_ts_between_clicks(
        self, data_keys: Union[str, List[str]] = [], *, inclusive: bool = False
    ) -> 'TimeSeries':  # pragma: no cover
        """
        Get a subset of the TimeSeries between two mouse clicks.

        Parameters
        ----------
        data_keys
            Optional. String or list of strings corresponding to the signals
            to plot. See TimeSeries.plot() for more information.
        inclusive
            Optional. True to include the given time in the comparison.

        Returns
        -------
        TimeSeries
            A new TimeSeries that fulfils the specified conditions.

        """
        fig = plt.figure()
        self.plot(data_keys)
        kineticstoolkit.gui.message(
            'Click on both sides of the portion to keep.', **WINDOW_PLACEMENT
        )
        plt.pause(0.001)  # Redraw
        points = plt.ginput(2)
        kineticstoolkit.gui.message('')
        times = [points[0][0], points[1][0]]
        plt.close(fig)
        return self.get_ts_between_times(
            min(times), max(times), inclusive=inclusive
        )

    def isnan(self, data_key: str) -> np.ndarray:
        """
        Return a boolean array of missing samples.

        Parameters
        ----------
        data_key
            Key value of the data signal to analyze.

        Returns
        -------
        np.ndarray
            A boolean array of the same size as the time vector, where True
            values represent missing samples (samples that contain at least
            one nan value).
        """
        values = self.data[data_key].copy()
        # Reduce the dimension of values while keeping the time dimension.
        while len(values.shape) > 1:
            values = np.sum(values, 1)  # type: ignore
        return np.isnan(values)

    def fill_missing_samples(
            self,
            max_missing_samples: int,
            *, method: str = 'linear',
            in_place: bool = False,
    ) -> 'TimeSeries':
        """
        Fill missing samples with the given method.

        The sample rate must be constant.

        Warning
        -------
        This function, which has been introduced in 0.1, is still experimental
        and may change signature or behaviour in the future.

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
        in_place
            Optional. True to modify and return the original TimeSeries. False
            to return a modified copy of the TimeSeries while leaving the
            original TimeSeries intact. The default is False.

        Returns
        -------
        TimeSeries
            The TimeSeries with the missing samples filled.

        """
        ts_out = self if in_place else self.copy()
        max_missing_samples = int(max_missing_samples)

        for data in ts_out.data:

            # Fill missing samples
            is_visible = ~ts_out.isnan(data)
            ts = ts_out.get_subset(data)
            ts.data[data] = ts.data[data][is_visible]
            ts.time = ts.time[is_visible]
            ts = ts.resample(ts_out.time, method, fill_value='extrapolate')

            # Put back missing samples in holes longer than max_missing_samples
            if max_missing_samples > 0:
                hole_start_index = 0
                to_keep = np.ones(self.time.shape)
                for current_index in range(ts.time.shape[0]):
                    if is_visible[current_index]:
                        hole_start_index = current_index
                    elif (
                        current_index - hole_start_index > max_missing_samples
                    ):
                        to_keep[hole_start_index + 1: current_index + 1] = 0

                ts.data[data][to_keep == 0] = np.nan

            ts_out.data[data] = ts.data[data]

        return ts_out

    def shift(self, time: float, *, in_place: bool = False) -> 'TimeSeries':
        """
        Shift time and events.time.

        Parameters
        ----------
        time_shift
            Time to be added to time and events.time.
        in_place
            Optional. True to modify and return the original TimeSeries. False
            to return a modified copy of the TimeSeries while leaving the
            original TimeSeries intact. The default is False.

        Returns
        -------
        TimeSeries
            The TimeSeries with the time being shifted.

        """
        ts = self if in_place else self.copy()
        for event in ts.events:
            event.time = event.time + time
        ts.time = ts.time + time
        return ts

    def sync_event(
            self,
            name: str,
            occurrence: int = 0,
            *,
            in_place: bool = False) -> 'TimeSeries':
        """
        Shift time and events.time so that this event is at the new time zero.

        Parameters
        ----------
        name
            Name of the event to sync on.
        occurrence
            Optional. Occurrence of the event to sync on, starting with 0.
        in_place
            Optional. True to modify and return the original TimeSeries. False
            to return a modified copy of the TimeSeries while leaving the
            original TimeSeries intact. The default is False.

        Returns
        -------
        TimeSeries
            The TimeSeries with the time being shifted.

        """
        ts = self if in_place else self.copy()
        ts.shift(-ts.get_event_time(name, occurrence), in_place=True)
        return ts

    def trim_events(self, *, in_place: bool = False) -> 'TimeSeries':
        """
        Delete the events that are outside the TimeSeries' time vector.

        Parameters
        ----------
        in_place
            Optional. True to modify and return the original TimeSeries. False
            to return a modified copy of the TimeSeries while leaving the
            original TimeSeries intact. The default is False.

        Returns
        -------
        TimeSeries
            The TimeSeries without the trimmed events.

        Example
        -------
        >>> ts = ktk.TimeSeries(time = np.arange(10))
        >>> ts.time
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        >>> ts = ts.add_event(-2)
        >>> ts = ts.add_event(0)
        >>> ts = ts.add_event(5)
        >>> ts = ts.add_event(9)
        >>> ts = ts.add_event(10)
        >>> ts.events
        [TimeSeriesEvent(time=-2, name='event'),
         TimeSeriesEvent(time=0, name='event'),
         TimeSeriesEvent(time=5, name='event'),
         TimeSeriesEvent(time=9, name='event'),
         TimeSeriesEvent(time=10, name='event')]

        >>> ts = ts.trim_events()
        >>> ts.events
        [TimeSeriesEvent(time=0, name='event'),
         TimeSeriesEvent(time=5, name='event'),
         TimeSeriesEvent(time=9, name='event')]

        """
        ts = self if in_place else self.copy()
        if ts.time.shape[0] == 0:  # no time, thus no event to keep.
            ts.events = []
            return ts

        events = deepcopy(ts.events)
        ts.events = []
        for event in events:
            if event.time >= ts.time[0] and event.time <= ts.time[-1]:
                ts.add_event(event.time, event.name, in_place=True)
        return ts

    def ui_sync(
            self,
            data_keys: Union[str, List[str]] = [],
            ts2: Union['TimeSeries', None] = None,
            data_keys2: Union[str, List[str]] = []
    ) -> 'TimeSeries':  # pragma: no cover
        """
        Synchronize one or two TimeSeries by shifting their time.

        If this method is called on only one TimeSeries, an interactive
        interface asks the user to click on the time to set to zero.

        If another TimeSeries is given, an interactive interface allows
        synchronizing both TimeSeries together.

        Warning
        -------
        This function, which has been introduced in 0.1, is still experimental
        and may change signature or behaviour in the future.

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

        Returns
        -------
        TimeSeries
            A copy of the TimeSeries after synchronization.

        Note
        ----
        The TimeSeries ts2 is modified in place.

        """
        ts1 = self.copy()

        fig = plt.figure('ktk.TimeSeries.ui_sync')

        if ts2 is None:
            # Synchronize ts1 only
            ts1.plot(data_keys)
            choice = kineticstoolkit.gui.button_dialog(
                'Please zoom on the time zero and press Next.',
                ['Cancel', 'Next'],
                **WINDOW_PLACEMENT,
            )
            if choice != 1:
                plt.close(fig)
                return ts1

            kineticstoolkit.gui.message(
                'Click on the sync event.', **WINDOW_PLACEMENT
            )
            click = plt.ginput(1)
            kineticstoolkit.gui.message('')
            plt.close(fig)
            ts1 = ts1.shift(-click[0][0])

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
                ts1.plot(data_keys)
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
                    choices=[
                        'Zero ts1 only, using ts1',
                        'Zero ts2 only, using ts2',
                        'Zero both TimeSeries, using ts1',
                        'Zero both TimeSeries, using ts2',
                        'Sync both TimeSeries on a common event',
                        'Finished',
                    ],
                    **WINDOW_PLACEMENT,
                )

                if choice == 0:  # Zero ts1 only
                    kineticstoolkit.gui.message(
                        'Click on the time zero in ts1.', **WINDOW_PLACEMENT
                    )
                    click_1 = plt.ginput(1)
                    kineticstoolkit.gui.message('')

                    ts1 = ts1.shift(-click_1[0][0])

                elif choice == 1:  # Zero ts2 only
                    kineticstoolkit.gui.message(
                        'Click on the time zero in ts2.', **WINDOW_PLACEMENT
                    )
                    click_1 = plt.ginput(1)
                    kineticstoolkit.gui.message('')

                    ts2 = ts2.shift(-click_1[0][0])

                elif choice == 2:  # Zero ts1 and ts2 using ts1
                    kineticstoolkit.gui.message(
                        'Click on the time zero in ts1.', **WINDOW_PLACEMENT
                    )
                    click_1 = plt.ginput(1)
                    kineticstoolkit.gui.message('')

                    ts1 = ts1.shift(-click_1[0][0])
                    ts2 = ts2.shift(-click_1[0][0])

                elif choice == 3:  # Zero ts1 and ts2 using ts2
                    kineticstoolkit.gui.message(
                        'Click on the time zero in ts2.', **WINDOW_PLACEMENT
                    )
                    click_2 = plt.ginput(1)
                    kineticstoolkit.gui.message('')

                    ts1 = ts1.shift(-click_2[0][0])
                    ts2 = ts2.shift(-click_2[0][0])

                elif choice == 4:  # Sync on a common event
                    kineticstoolkit.gui.message(
                        'Click on the sync event in ts1.', **WINDOW_PLACEMENT
                    )
                    click_1 = plt.ginput(1)
                    kineticstoolkit.gui.message(
                        'Now click on the same event in ts2.',
                        **WINDOW_PLACEMENT,
                    )
                    click_2 = plt.ginput(1)
                    kineticstoolkit.gui.message('')

                    ts1 = ts1.shift(-click_1[0][0])
                    ts2 = ts2.shift(-click_2[0][0])

                elif choice == 5 or choice < -1:  # OK or closed figure, quit.
                    plt.close(fig)
                    finished = True

        return ts1

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
            try:
                ts.data[key] = self.data[key].copy()
            except KeyError:
                pass

            try:
                ts.data_info[key] = deepcopy(self.data_info[key])
            except KeyError:
                pass

        return ts

    @unstable
    def resample(
            self,
            new_time: np.ndarray,
            kind: str = 'linear',
            *,
            fill_value: Union[np.ndarray, str, None] = None,
            in_place: bool = False) -> 'TimeSeries':
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
        in_place
            Optional. True to modify and return the original TimeSeries. False
            to return a modified copy of the TimeSeries while leaving the
            original TimeSeries intact. The default is False.

        Returns
        -------
        TimeSeries
            The TimeSeries with a new sample rate.

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

        >>> ts = ts.resample(np.arange(0, 5.5, 0.5))

        >>> ts.time
        array([0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. ])

        >>> ts.data['data']
        array([ 0. ,  0.5,  1. ,  2.5,  4. ,  6.5,  9. , 12.5, 16. , 20.5, 25. ])

        >>> ts.data['data_with_nans']
        array([ 0. ,  0.5,  1. ,  2.5,  4. ,  nan,  nan,  nan, 16. , 20.5, 25. ])

        """
        ts = self if in_place else self.copy()

        if np.any(np.isnan(new_time)):
            raise ValueError('new_time must not contain nans')

        for key in ts.data.keys():
            index = ~ts.isnan(key)

            if sum(index) < 3:  # Only Nans, cannot interpolate.
                warnings.warn(
                    f'Warning: Almost only NaNs found in signal "{key}.'
                )
                # We generate an array of nans of the expected size.
                new_shape = [len(new_time)]
                for i in range(1, len(self.data[key].shape)):
                    new_shape.append(self.data[key].shape[i])
                ts.data[key] = np.empty(new_shape)
                ts.data[key][:] = np.nan

            else:  # Interpolate.

                # Express nans as a range of times to
                # remove from the final, interpolated timeseries
                nan_indexes = np.argwhere(~index)
                time_ranges_to_remove = []  # type: List[Tuple[int, int]]
                length = ts.time.shape[0]
                for i in nan_indexes:
                    if i > 0 and i < length - 1:
                        time_range = (ts.time[i - 1], ts.time[i + 1])
                    elif i == 0:
                        time_range = (-np.inf, ts.time[i + 1])
                    else:
                        time_range = (ts.time[i - 1], np.inf)
                    time_ranges_to_remove.append(time_range)

                if kind == 'pchip':
                    P = sp.interpolate.PchipInterpolator(
                        ts.time[index],
                        ts.data[key][index],
                        axis=0,
                        extrapolate=(
                            True if fill_value == 'extrapolate' else False
                        ),
                    )
                    ts.data[key] = P(new_time)
                else:
                    f = sp.interpolate.interp1d(
                        ts.time[index],
                        ts.data[key][index],
                        axis=0,
                        fill_value=fill_value,
                        kind=kind,
                    )
                    ts.data[key] = f(new_time)

                # Put back nans
                for j in time_ranges_to_remove:
                    ts.data[key][
                        (new_time > j[0]) & (new_time < j[1])
                    ] = np.nan

        ts.time = new_time
        return ts

    def merge(
            self,
            ts: 'TimeSeries',
            data_keys: Union[str, List[str]] = [],
            *,
            resample: bool = False,
            overwrite: bool = False,
            in_place: bool = False) -> 'TimeSeries':
        """
        Merge the TimeSeries with another TimeSeries.

        Warning
        -------
        This function, which has been introduced in 0.2, is still experimental and
        may change signature or behaviour in the future.

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
            False), the duplicate data in ts is ignored. The default is False.
        in_place
            Optional. True to modify and return the original TimeSeries. False
            to return a modified copy of the TimeSeries while leaving the
            original TimeSeries intact. The default is False.

        Returns
        -------
        TimeSeries
            The merged TimeSeries.

        """
        ts_out = self if in_place else self.copy()
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
                    'data_keys must be a string or list of strings'
                )

        # Check if resampling is needed
        if len(ts_out.time) == 0:
            ts_out.time = deepcopy(ts.time)

        if (ts_out.time.shape == ts.time.shape) and np.all(
            ts_out.time == ts.time
        ):
            must_resample = False
        else:
            must_resample = True

        if must_resample is True and resample is False:
            raise ValueError(
                'Time vectors do not match, resampling is required.'
            )

        if must_resample is True:
            ts.resample(ts_out.time, fill_value='extrapolate', in_place=True)

        for key in data_keys:

            # Check if this key is a duplicate, then continue to next key if
            # required.
            if (
                (key in ts_out.data)
                and (overwrite is False)
            ):
                pass

            else:
                # Add this data
                ts_out.data[key] = ts.data[key]

                if key in ts.data_info:
                    for info_key in ts.data_info[key].keys():
                        ts_out.add_data_info(
                            key,
                            info_key,
                            ts.data_info[key][info_key],
                            in_place=True,
                        )

        # Merge events
        for event in ts.events:
            ts_out.events.append(event)
        ts_out.sort_events(in_place=True)
        return ts_out


if __name__ == "__main__":  # pragma: no cover
    import doctest
    import kineticstoolkit as ktk
    import numpy as np

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
