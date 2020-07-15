#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020 Félix Chénier
#
# This file is not for redistribution.
"""
Provides functions to save and load data.

Most of the functions in this module are temporary and only helpers for
my transition from Matlab code to Python code. For example, the _loadmat
function is very specific for my laboratory. When this module will have
settled and all my lab's data will be converted to JSON, only the save and
load functions will remain, and these functions will save and load using the
JSON format.
"""

from ktk.timeseries import TimeSeries
from ktk.timeseries import dataframe_to_dict_of_arrays
from ktk.timeseries import dict_of_arrays_to_dataframe
import ktk.config

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
import getpass


def save(filename, variable):

    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return {'OriginalClass': 'numpy.array',
                        'value': obj.tolist()}

            elif isinstance(obj, complex):
                return {'OriginalClass': 'complex',
                        'real': obj.real,
                        'imag': obj.imag,
                        }

            elif str(type(obj)) == "<class 'ktk.timeseries.TimeSeries'>":
                out = {}
                out['OriginalClass'] = 'ktk.TimeSeries'
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

            else:
                return super().default(obj)

    now = datetime.now()
    if ktk.config.is_pc:
        computer = 'PC'
    elif ktk.config.is_mac:
        computer = 'Mac'
    elif ktk.config.is_linux:
        computer = 'Linux'
    else:
        computer = 'Unknown'

    metadata = {
        'Software': 'Kinetics Toolkit (ktk)',
        'Version': ktk.config.version,
        'Computer': computer,
        'FileFormat': 1.0,
        'SaveDate': now.strftime('%Y-%m-%d'),
        'SaveTime': now.strftime('%H:%M:%S'),
        'User': getpass.getuser(),
    }

    with open(filename, "w") as fid:
        json.dump([metadata, variable], fid,
                  cls=CustomEncoder,
                  indent='\t')


def _load_json(filename):
    """Main function to load ktk's JSON file format."""

    def object_hook(obj):
        if 'OriginalClass' in obj:
            if obj['OriginalClass'] == 'numpy.array':
                return np.array(obj['value'])

            elif obj['OriginalClass'] == 'complex':
                return obj['real'] + obj['imag'] * 1j

            elif obj['OriginalClass'] == 'ktk.TimeSeries':
                out = ktk.TimeSeries()
                out.time = obj['time']
                out.time_info = obj['time_info']
                out.data_info = obj['data_info']
                for key in obj['data']:
                    out.data[key] = np.array(obj['data'][key])
                for event in obj['events']:
                    out.add_event(event['time'], event['name'])
                return out

            else:
                return obj

        else:
            return obj

    with open(filename, 'r') as fid:
        obj = json.load(fid, object_hook=object_hook)

    return obj[1]


def _load_ktk_zip(filename):
    """
    Read the older ktk.zip file format.

    This is a deprecated function as ktk.zip is to be removed soon.
    """
    warnings.warn('This is a deprecated function. ktk.zip support is to be '
                  'removed soon.')

    def _load(filename):
        """
        Load the contents of a folder or filename.

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

    # Function begins here.

    basename = os.path.basename(filename)
    temp_folder_name = ktk.config.temp_folder + '/' + basename

    # We will rename the folder to .dict to uniformize loading using _load
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


def load(filename):
    """
    Load a KTK json, zip or mat file.

    Load a data file as saved using the ktk.save function.

    Parameters
    ----------
    filename : str
        The path of the zip file to load.

    Returns
    -------
    The loaded variable.
    """
    if filename is None:
        raise ValueError('filename is empty.')
    if not isinstance(filename, str):
        raise ValueError('filename must be a string.')

    if filename.lower().endswith('.json'):
        return _load_json(filename)

    elif filename.lower().endswith('.ktk.zip'):
        return _load_ktk_zip(filename)

    elif filename.lower().endswith('.mat'):
        return _loadmat(filename)

    else:
        raise ValueError('The file must be either JSON, ktk.zip or MAT.')


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
        contents = convert_to_timeseries(contents)
        contents = convert_cell_arrays_to_lists(contents)
        return contents
    else:
        return data


def convert_cell_arrays_to_lists(the_input):
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
                    the_list.append(convert_cell_arrays_to_lists(
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
                                convert_cell_arrays_to_lists(the_input[cell])

                the_input = the_list

        else:
            for key in the_input.keys():
                the_input[key] = convert_cell_arrays_to_lists(the_input[key])
    elif isinstance(the_input, list):
        for i in range(len(the_input)):
            the_input[key] = convert_cell_arrays_to_lists(the_input[key])

    return the_input


def convert_to_timeseries(the_input):
    """
    Convert dicts of Matlab timeseries to KTK TimeSeries.

    This function recursively goes into the_input and checks for dicts that
    result from obvious Matlab's structures of timeseries. These structures are
    converted to KTK Timeseries.

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
                the_input[the_key] = convert_to_timeseries(the_input[the_key])
            return the_input

    elif isinstance(the_input, list):
        for i in range(len(the_input)):
            the_input[i] = convert_to_timeseries(the_input[i])
        return the_input

    else:
        return the_input


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


def _todict(variable):
    """Construct dicts from Mat-objects."""
    dict = {}
    for strg in variable._fieldnames:
        elem = variable.__dict__[strg]
        dict[strg] = elem
    return dict
