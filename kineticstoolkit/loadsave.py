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
Provide functions to load and save data.

The classes defined in this module are accessible directly from the toplevel
Kinetics Toolkit namespace (i.e. ktk.load, ktk.save).

"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"

from kineticstoolkit.timeseries import TimeSeries
from kineticstoolkit.timeseries import dataframe_to_dict_of_arrays
from kineticstoolkit.decorators import stable, private
import kineticstoolkit.config

import scipy.io as spio
import os
import numpy as np
import pandas as pd
from ast import literal_eval
import csv
import warnings
import shutil
import json
from datetime import datetime
import time
import getpass
import zipfile

from typing import Any, List


listing = []  # type: List[str]


@stable(listing)
def save(filename: str, variable: Any) -> None:
    """
    Save a variable to a ktk.zip file.

    A ktk.zip file is a zipped folder containing two files:

    - metadata.json, which includes save date, user, etc.
    - data.json, which includes the data.

    The following classes are supported:

    - dict containing any supported class
    - list containing any supported class
    - str
    - int
    - float
    - True
    - False
    - None
    - numpy.array
    - pandas.DataFrame
    - pandas.Series
    - ktk.TimeSeries

    Tuples are also supported but will be loaded back as lists, without
    warning.
    """

    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return {'class__': 'numpy.array',
                        'value': obj.tolist()}

            elif str(type(obj)) == \
                    "<class 'kineticstoolkit.timeseries.TimeSeries'>":
                out = {}
                out['class__'] = 'ktk.TimeSeries'
                out['time'] = obj.time.tolist()
                out['time_info'] = obj.time_info
                out['data_info'] = obj.data_info
                out['data'] = {}
                for key in obj.data:
                    out['data'][key] = obj.data[key].tolist()
                out['events'] = []
                for event in obj.events:
                    out['events'].append({
                        'time': event.time,
                        'name': event.name,
                    })
                return out

            elif isinstance(obj, pd.Series):
                # return {'class__': 'pandas.Series',
                #         'value': json.loads(
                #             pd.DataFrame(obj).to_json(orient='table'))}
                return {'class__': 'pandas.Series',
                        'name': str(obj.name),
                        'dtype': str(obj.dtype),
                        'index': obj.index.tolist(),
                        'data': obj.tolist(),
                        }

            elif isinstance(obj, pd.DataFrame):
                return {'class__': 'pandas.DataFrame',
                        'columns': obj.columns.tolist(),
                        'dtypes': [str(dtype) for dtype in obj.dtypes],
                        'index': obj.index.tolist(),
                        'data': obj.to_numpy().tolist(),
                        }

            elif isinstance(obj, complex):
                return {'class__': 'complex',
                        'real': obj.real,
                        'imag': obj.imag,
                        }

            else:
                return super().default(obj)

    now = datetime.now()
    if kineticstoolkit.config.is_pc:
        computer = 'PC'
    elif kineticstoolkit.config.is_mac:
        computer = 'Mac'
    elif kineticstoolkit.config.is_linux:
        computer = 'Linux'
    else:
        computer = 'Unknown'

    metadata = {
        'Software': 'Kinetics Toolkit',
        'Version': kineticstoolkit.config.version,
        'Computer': computer,
        'FileFormat': 1.0,
        'SaveDate': now.strftime('%Y-%m-%d'),
        'SaveTime': now.strftime('%H:%M:%S'),
        'User': getpass.getuser(),
    }

    # Save
    temp_folder = kineticstoolkit.config.temp_folder + '/save' + str(time.time())

    try:
        shutil.rmtree(temp_folder)
    except:
        pass
    os.mkdir(temp_folder)

    with open(temp_folder + '/metadata.json', "w") as fid:
        json.dump(metadata, fid, indent='\t')

    with open(temp_folder + '/data.json', "w") as fid:
        json.dump(variable, fid, cls=CustomEncoder, indent='\t')

    shutil.make_archive(temp_folder, 'zip', temp_folder)
    os.rename(temp_folder + '.zip', filename)
    shutil.rmtree(temp_folder)


@private(listing)
def _load(filename):
    """
    Load the contents of a folder or filename.

    This is a deprecated function as ktk.zip is to be removed soon.

    Returns a tuple where the first element is the suffix
    (.eval.txt, .dict, etc) and the second element is the contents.
    """

    # Easiest case:
    if filename.endswith('.str.txt'):
        with open(filename, 'r') as fid:
            return ('.str.txt', fid.read())

    # Next easiest:
    elif filename.endswith('.eval.txt'):
        with open(filename, 'r') as fid:
            return ('.eval.txt', literal_eval(fid.read()))

    elif filename.endswith('.dict'):
        variable = dict()
        list_of_files = os.listdir(filename)
        for subfilename in list_of_files:
            contents = _load(filename + '/' + subfilename)
            key = subfilename[0:-len(contents[0])]
            variable[key] = contents[1]
        return ('.dict', variable)

    elif filename.endswith('.list'):
        variable = list()
        file_list = os.listdir(filename)
        indexes = [int(file.split('.')[0]) for file in file_list]
        sorted_file_list = [x for (_, x) in sorted(zip(indexes, file_list))]

        for file in sorted_file_list:
            contents = _load(filename + '/' + file)
            variable.append(contents[1])
        return ('.list', variable)

    elif filename.endswith('.tuple'):
        variable = list()
        file_list = os.listdir(filename)
        indexes = [int(file.split('.')[0]) for file in file_list]
        sorted_file_list = [x for (_, x) in sorted(zip(indexes, file_list))]

        for file in sorted_file_list:
            contents = _load(filename + '/' + file)
            variable.append(contents[1])
        return ('.tuple', tuple(variable))

    elif filename.endswith('.ndarray.txt'):
        dataframe = pd.read_csv(filename, sep='\t',
                                quoting=csv.QUOTE_NONNUMERIC)
        dict_of_arrays = dataframe_to_dict_of_arrays(dataframe)
        return ('.ndarray.txt', dict_of_arrays['Data'])

    elif filename.endswith('.TimeSeries'):

        data = pd.read_csv(filename + '/data.txt',
                           sep='\t', quoting=csv.QUOTE_NONNUMERIC)
        events = pd.read_csv(filename + '/events.txt',
                             sep='\t', quoting=csv.QUOTE_NONNUMERIC)
        info = pd.read_csv(filename + '/info.txt',
                           sep='\t', quoting=csv.QUOTE_NONNUMERIC,
                           index_col=0)

        out = TimeSeries()

        # DATA AND TIME
        # -------------
        out.data = dataframe_to_dict_of_arrays(data)
        out.time = out.data['time']
        out.data.pop('time', None)

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

        return ('.TimeSeries', out)

    else:
        warnings.warn(f'Could not load contents in {filename}')
        return ('', None)


@private(listing)
def _load_object_hook(obj):
    if 'class__' in obj:
        to_class = obj['class__']
        if to_class == 'numpy.array':
            return np.array(obj['value'])

        elif to_class == 'ktk.TimeSeries':
            out = TimeSeries()
            out.time = np.array(obj['time'])
            out.time_info = obj['time_info']
            out.data_info = obj['data_info']
            for key in obj['data']:
                out.data[key] = np.array(obj['data'][key])
            for event in obj['events']:
                out.add_event(event['time'], event['name'])
            return out

        elif to_class == 'pandas.DataFrame':
            return pd.DataFrame(obj['data'],
                                dtype=obj['dtypes'][0],
                                columns=obj['columns'],
                                index=obj['index'])

        elif to_class == 'pandas.Series':
            return pd.Series(obj['data'],
                             dtype=obj['dtype'],
                             name=obj['name'],
                             index=obj['index'])

        elif to_class == 'complex':
            return obj['real'] + obj['imag'] * 1j

        else:
            warnings.warn(f'The "{to_class}" class is not supported by '
                          'this version of Kinetics Toolkit. Please check '
                          'that Kinetics Toolkit is up to date.')
            return obj

    else:
        return obj


@private(listing)
def _load_ktk_zip(filename, include_metadata=False):
    """Read the ktk.zip file format."""

    archive = zipfile.ZipFile(filename, 'r')
    try:
        data = json.loads(archive.read('data.json').decode(),
                          object_hook=_load_object_hook)

        if include_metadata:
            metadata = json.loads(archive.read('metadata.json').decode(),
                                  object_hook=_load_object_hook)
            return data, metadata

        else:
            return data


    except Exception:
        # No data.json. It seems to be the old format.

        basename = os.path.basename(filename)
        temp_folder_name = kineticstoolkit.config.temp_folder + '/' + basename

        # We will rename the folder to .dict to uniformize loading using _load
        # if needed.
        new_temp_folder_name = (
            temp_folder_name[0:-len('.ktk.zip')] + '.dict')

        try:
            shutil.rmtree(temp_folder_name)
        except Exception:
            pass
        try:
            shutil.rmtree(new_temp_folder_name)
        except Exception:
            pass

        os.mkdir(temp_folder_name)
        shutil.unpack_archive(filename, extract_dir=temp_folder_name)

        new_temp_folder_name = (
            temp_folder_name[0:-len('.ktk.zip')] + '.dict')

        os.rename(temp_folder_name, new_temp_folder_name)

        variable = _load(new_temp_folder_name)[1]

        shutil.rmtree(new_temp_folder_name)

        # Return the entry that corresponds to the contents
        for key in variable:
            if key != 'metadata':
                return variable[key]


@stable(listing)
def load(filename: str) -> Any:
    """
    Load a ktk.zip file.

    Load a data file as saved using the ktk.save function.

    Parameters
    ----------
    filename : str
        The path of the zip file to load.

    Returns
    -------
    Any
        The loaded variable.
    """

    # NOTE: THIS FUNCTION CAN ALSO LOAD MAT FILES, BUT THIS IS A TRANSITIONAL
    # FEATURE FOR MY LAB'S MIGRATION FROM MATLAB TO PYTHON. PLEASE DO NOT
    # RELY ON THIS FEATURE AS IT COULD BE REMOVED SOON.
    # Check hdf5storage for a nice alternative.
    if filename is None:
        raise ValueError('filename is empty.')
    if not isinstance(filename, str):
        raise ValueError('filename must be a string.')

    if filename.lower().endswith('.zip'):
        return _load_ktk_zip(filename)

    elif filename.lower().endswith('.mat'):
        return _loadmat(filename)

    else:
        raise ValueError('The file must be either zip or mat.')


@private(listing)
def _loadmat(filename):
    """
    Load a Matlab's MAT file.

    Parameters
    ----------
    filename : str
        Path of the MAT file to load.

    Returns
    -------
    The saved variable.
    """
    # Load the Matlab file
    data = spio.loadmat(filename, struct_as_record=False,
                        squeeze_me=True)

    # Correct the keys
    data = _recursive_matstruct_to_dict(data)

    # Return contents and metadata if it exists, if not return data as is.
    if ('contents' in data) and ('metadata' in data):

        # Since it was saved using ktksave, then also convert structures of
        # timeseries to ktk.TimeSeries.
        contents = data['contents']
        # metadata = data['metadata']  # Not sure what to do with it yet.
        contents = _convert_to_timeseries(contents)
        contents = _convert_cell_arrays_to_lists(contents)
        return contents
    else:
        return data


@private(listing)
def _convert_cell_arrays_to_lists(the_input):
    """
    Convert cell arrays to lists.

    This function recursively goes into the_input and checks for dicts that
    contains a 'OriginalClass' == 'cell' field. These dicts are then converted
    to lists.
    """
    if isinstance(the_input, dict):
        if (('OriginalClass' in the_input) and
                the_input['OriginalClass'] == 'cell'):
            # This should be converted to a list.
            # Let's try with a single dimension list. If it fails, try as a
            # bidimensional list.
            try:
                length = len(the_input.keys()) - 1
                the_list = []
                for i_cell in range(length):
                    the_list.append(_convert_cell_arrays_to_lists(
                            the_input[f'cell{i_cell+1}']))
                the_input = the_list

            except KeyError:
                # Convert to a dict with addresses in keys

                def extract_address(key):
                    if key != 'OriginalClass':
                        return key[4:].split('_')
                    else:
                        return None

                # Get the number of rows and columns
                n_rows = 0
                n_cols = 0
                for cell in the_input.keys():
                    address = extract_address(cell)
                    if address is not None:
                        n_rows = max(n_rows, int(address[0]))
                        n_cols = max(n_cols, int(address[1]))

                # Create and populate the list
                the_list = [
                        [None for i_col in range(n_cols)]
                        for i_row in range(n_rows)]
                for cell in the_input.keys():
                    address = extract_address(cell)
                    if address is not None:
                        the_list[int(address[0])-1][int(address[1])-1] = \
                                _convert_cell_arrays_to_lists(the_input[cell])

                the_input = the_list

        else:
            for key in the_input.keys():
                the_input[key] = _convert_cell_arrays_to_lists(the_input[key])
    elif isinstance(the_input, list):
        for i in range(len(the_input)):
            the_input[key] = _convert_cell_arrays_to_lists(the_input[key])

    return the_input


@private(listing)
def _convert_to_timeseries(the_input):
    """
    Convert dicts of Matlab timeseries to Kinetics Toolkit TimeSeries.

    This function recursively goes into the_input and checks for dicts that
    result from obvious Matlab's structures of timeseries. These structures are
    converted to Kinetics Toolkit Timeseries.

    Parameters
    ----------
    the_input : any variable
        The input to be checked, usually a dict.

    Returns
    -------
    A copy of the input, with the converted timeseries.
    """
    if isinstance(the_input, dict):

        # Check if this dict should become a timeseries
        is_a_timeseries = True

        for the_key in the_input:
            if (isinstance(the_input[the_key], dict)
                    and ('OriginalClass' in the_input[the_key])
                    and (the_input[the_key]['OriginalClass'] == 'timeseries')):
                pass
            else:
                is_a_timeseries = False

        if is_a_timeseries is True:
            # We checked if each key is a Matlab timeseries and it is.
            # So we get here.
            the_output = TimeSeries()
            for the_key in the_input:

                if (isinstance(the_input[the_key], dict) and
                        ('OriginalClass' in the_input[the_key]) and
                        (the_input[the_key]['OriginalClass'] == 'timeseries')):
                    # This is a matlab timeseries.

                    # Extract Time
                    the_output.time = the_input[the_key]['Time']

                    # Extract Data
                    the_data = the_input[the_key]['Data']

                    if the_input[the_key]['IsTimeFirst'] is True:
                        the_output.data[the_key] = the_data
                    else:  # We must reshape to ensure time is on first dim.
                        the_shape = the_data.shape
                        if len(the_shape) == 2:
                            the_output.data[the_key] = \
                                the_data.transpose((1, 0))
                        elif len(the_shape) == 3:
                            the_output.data[the_key] = \
                                the_data.transpose((2, 0, 1))
                        else:
                            the_output.data[the_key] = the_data

                    # Extract Events
                    for event in the_input[the_key]['Events']:
                        the_output.add_event(event['Time'], event['Name'])

            return the_output

        else:
            # It is only a dict, not a dict of matlab timeseries.
            for the_key in the_input:
                the_input[the_key] = _convert_to_timeseries(the_input[the_key])
            return the_input

    elif isinstance(the_input, list):
        for i in range(len(the_input)):
            the_input[i] = _convert_to_timeseries(the_input[i])
        return the_input

    else:
        return the_input


@private(listing)
def _recursive_matstruct_to_dict(variable):
    """Recursively converts Mat-objects in dicts or arrays to nested dicts."""
    if isinstance(variable, spio.matlab.mio5_params.mat_struct):
        variable = _todict(variable)
        variable = _recursive_matstruct_to_dict(variable)
    elif isinstance(variable, dict):
        for key in variable:
            variable[key] = _recursive_matstruct_to_dict(variable[key])
    elif isinstance(variable, np.ndarray):
        for index in np.ndindex(np.shape(variable)):
            variable[index] = _recursive_matstruct_to_dict(variable[index])
    return variable


@private(listing)
def _todict(variable):
    """Construct dicts from Mat-objects."""
    dict = {}
    for strg in variable._fieldnames:
        elem = variable.__dict__[strg]
        dict[strg] = elem
    return dict


def __dir__():
    return listing
