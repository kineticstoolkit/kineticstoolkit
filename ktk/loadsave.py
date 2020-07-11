#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020 Félix Chénier
#
# This file is not for redistribution.
"""
Provides functions to load .mat and .ktk.zip files, and to save .ktk.zip files.
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


def _save(folder_name, variable_name, variable):
    """Save a variable in a folder. This function is called by save."""

    # Try the easiest way: save the string representation of the variable.
    try:
        string = str(variable)
        test_variable = literal_eval(string)
        assert test_variable == variable

        with open(folder_name + '/' + variable_name + '.eval.txt', 'w') as fid:
            fid.write(string)
        return
    except Exception:
        pass

    # It didn't work, we will have some work to do.
    if type(variable) == dict:
        this_folder_name = folder_name + '/' + variable_name + '.dict'
        os.mkdir(this_folder_name)
        for dict_key, dict_variable in variable.items():
            _save(folder_name=this_folder_name,
                  variable_name=dict_key,
                  variable=dict_variable)

    elif type(variable) == list or type(variable) == tuple:
        this_folder_name = folder_name + '/' + variable_name + '.list'
        os.mkdir(this_folder_name)
        for i_item, item in enumerate(variable):
            _save(folder_name=this_folder_name,
                  variable_name=str(i_item),
                  variable=item)

    elif type(variable) == np.ndarray:
        dataframe = dict_of_arrays_to_dataframe({'Data': variable})
        dataframe.to_csv(folder_name + '/' + variable_name + '.ndarray.txt',
                         sep='\t', quoting=csv.QUOTE_NONNUMERIC,
                         header=True)

    elif type(variable) == str:
        with open(folder_name + '/' + variable_name + '.str.txt', 'w') as fid:
            fid.write(variable)

    elif str(type(variable)) == "<class 'ktk.timeseries.TimeSeries'>":
        # This string comparison instead of a type check is because ktk is
        # often reloading in iPython with %autoreload activated. This causes
        # new class definitions to be declared and old instances of
        # TimeSeries are then not recognized as the same class than the new
        # class definition. To work around this, I compare the string
        # representation of type(variable), which does not change between
        # class redefinitions.
        this_folder_name = folder_name + '/' + variable_name + '.TimeSeries'
        os.mkdir(this_folder_name)

        # data and time
        variable = variable.copy()
        np_data = variable.data
        np_data['time'] = variable.time
        dataframe = dict_of_arrays_to_dataframe(np_data)
        dataframe.to_csv(this_folder_name + '/data.txt',
                         sep='\t', quoting=csv.QUOTE_NONNUMERIC,
                         index=False)

        # events
        if len(variable.events) > 0:
            df_events = pd.DataFrame(variable.events)
            df_events.columns = ['time', 'name']
        else:
            df_events = pd.DataFrame(columns=['time', 'name'])

        df_events.to_csv(this_folder_name + '/events.txt',
                         sep='\t', quoting=csv.QUOTE_NONNUMERIC,
                         index=False)

        # info
        df_time_info = pd.DataFrame({'time': variable.time_info})
        df_data_info = pd.DataFrame(variable.data_info)
        df_info = pd.concat([df_time_info, df_data_info],
                            axis=1, sort=False)

        df_info.to_csv(this_folder_name + '/info.txt',
                       sep='\t', quoting=csv.QUOTE_NONNUMERIC)

    else:
        warnings.warn(f'The variable {variable_name} could not be '
                      f'saved because its type or contents is not '
                      'supported.')


def save(filename, variable):
    """
    Save a variable in a zip file.

    The supported variable types are:

        - Any basic builtin type that can be reconstructed using its string
          representation. For example, if str(the_variable) evaluates to the
          variable, then this variable can be saved.
          This includes str, int, float, complex and bool.
          This also includes lists and tuples if they contain such variables
          (if they can also be reconstructed completely by evaluating their
          string representation).
        - Multidimensional NumPy Arrays. They are saved as txt files, where
          each line correspond to the first dimension of the array, and where
          all other dimensions are reshaped as columns. The column headers
          include brackets so that it is clear what column corresponds to
          what dimension of the original multidimensional array.
        - ktkTimeSeries. They are saved as a folder that contains:
            - data.txt : The time and data as a table. Multidimensional arrays
              are reshaped as for the NumPy arrays.
            - events.txt : A list of events as a table, with a column of time
              and a column of event names.
            - info.txt : A list of time and data info as a table.
        - Dictionaries. They are saved as folders, where the content of the
          folder corresponds to the keys of the dict. Thus, nested dicts
          are saved as nested folders, and Dictionaries of NumPy arrays or
          ktk.TimeSeries are saved as txt files inside a structure file
          hierarchy.

    The function generates a warning if a variable type is unsupported, and
    the corresponding variable is not saved.

    Parameters
    ----------
    filename : str
        The name of the output file. The '.ktk.zip' extension is optional, it
        is added automatically.
    variable : <any supported type>
        The variable to be saved. To save multiple variables at once, consider
        saving a dict.

    Returns
    -------
    None.
    """
    save_folder = os.path.dirname(filename)
    if save_folder == '':
        save_folder = '.'

    filename = os.path.basename(filename)

    # Convert 'filename' and 'filename.ktk.zip' to 'filename.ktk'
    if filename.lower().endswith('.zip'):
        filename = filename[0:-len('.zip')]
    if not filename.lower().endswith('.ktk'):
        filename = filename + '.ktk'

    # Create and empty temp folder
    temp_folder_name = ktk.config.temp_folder + '/' + filename
    try:
        shutil.rmtree(temp_folder_name)
    except Exception:
        pass
    os.mkdir(temp_folder_name)

    # Save the file hierarchy
    _save(folder_name=temp_folder_name,
          variable_name=filename,
          variable=variable)

    os.mkdir(temp_folder_name + '/metadata.dict')
    with open(temp_folder_name +
              '/metadata.dict/file_version.str.txt', 'w') as fid:
        fid.write('1.0\n')

    # Zip it into its final destination
    shutil.make_archive(save_folder + '/' + filename,
                        'zip',
                        temp_folder_name)

    # Clear the temporary folder
    shutil.rmtree(temp_folder_name)


def _add_to_current_variable(variable, key, value):
    """Used by _load_current_folder"""
    if type(variable) == dict:
        variable[key] = value
    else:
        variable.append(value)


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



    # list_of_files = os.listdir(folder)
    # for filename in sorted(list_of_files):

    #     elif filename.endswith('.list'):
    #         key = filename[0:-len('.list')]
    #         os.chdir(filename)
    #         _add_to_current_variable(variable, key, _load_current_folder(load_as='list'))
    #         os.chdir('..')

    #     elif filename.endswith('.tuple'):
    #         key = filename[0:-len('.tuple')]
    #         os.chdir(filename)
    #         _add_to_current_variable(variable, key, _load_current_folder(load_as='tuple'))
    #         os.chdir('..')

    #     elif filename.endswith('.set'):
    #         key = filename[0:-len('.set')]
    #         os.chdir(filename)
    #         _add_to_current_variable(variable, key, _load_current_folder(load_as='set'))
    #         os.chdir('..')

    #     elif filename.endswith('.str.txt'):
    #         key = filename[0:-len('.str.txt')]
    #         file = open(filename, 'r')
    #         _add_to_current_variable(variable, key, file.read())
    #         file.close()

    #     elif filename.endswith('.eval.txt'):
    #         key = filename[0:-len('.eval.txt')]
    #         file = open(filename, 'r')
    #         _add_to_current_variable(variable, key, literal_eval(file.read()))
    #         file.close()

    #     elif filename.endswith('.ndarray.txt'):
    #         key = filename[0:-len('.ndarray.txt')]
    #         dataframe = pd.read_csv(filename, sep='\t',
    #                                 quoting=csv.QUOTE_NONNUMERIC)
    #         dict_of_arrays = dataframe_to_dict_of_arrays(dataframe)
    #         _add_to_current_variable(variable, key, dict_of_arrays['Data'])

    #         os.chdir('..')

    # if load_as == 'tuple':
    #     variable = tuple(variable)
    # elif load_as == 'set':
    #     variable = set(variable)

    # return variable


def load(filename):
    """
    Load a KTK zip data file.

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

def loadmat(filename):
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
            if (isinstance(the_input[the_key], dict) and
                    ('OriginalClass' in the_input[the_key]) and
                    (the_input[the_key]['OriginalClass'] == 'timeseries')):
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


