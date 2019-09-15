#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 13:29:24 2019

@author: felix
"""

import ktk
import scipy.io as _spio
import os as _os
import subprocess as _subprocess
import numpy as np
import pandas as pd
from ast import literal_eval
import csv
import warnings
import shutil


def _save_to_current_folder(variable, variable_name):
    if type(variable) == dict:
        _os.mkdir(variable_name + '.dict')
        _os.chdir(variable_name + '.dict')
        for dict_key, dict_variable in variable.items():
            _save_to_current_folder(dict_variable, dict_key)
        _os.chdir('..')

    elif type(variable) == str:
        file = open(variable_name + '.str.txt', 'w')
        file.write(str(variable))
        file.close()

    elif type(variable) == np.ndarray:
        dataframe = dict_of_arrays_to_dataframe({'Data': variable})
        dataframe.to_csv(variable_name + '.ndarray.txt',
                         sep='\t', quoting=csv.QUOTE_NONNUMERIC,
                         header=True)

    elif type(variable) == ktk.TimeSeries:
        _os.mkdir(variable_name + '.TimeSeries')
        _os.chdir(variable_name + '.TimeSeries')

        # data and time
        variable = variable.copy()
        np_data = variable.data
        np_data['time'] = variable.time
        dataframe = dict_of_arrays_to_dataframe(np_data)
        dataframe.to_csv('data.txt',
                         sep='\t', quoting=csv.QUOTE_NONNUMERIC,
                         index=False)

        # events
        if len(variable.events) > 0:
            df_events = pd.DataFrame(variable.events)
            df_events.columns = ['time', 'name']
        else:
            df_events = pd.DataFrame(columns=['time', 'name'])

        df_events.to_csv('events.txt',
                         sep='\t', quoting=csv.QUOTE_NONNUMERIC,
                         index=False)

        # info
        df_time_info = pd.DataFrame({'time': variable.time_info})
        df_data_info = pd.DataFrame(variable.data_info)
        df_info = pd.concat([df_time_info, df_data_info],
                            axis=1, sort=False)

        df_info.to_csv('info.txt',
                       sep='\t', quoting=csv.QUOTE_NONNUMERIC)

        _os.chdir('..')

    else:
        # If the variable's string form is sufficient to load it back, then
        # save it.
        string = str(variable)
        try:
            test_variable = literal_eval(string)
        except Exception:
            test_variable = None

        if test_variable == variable:
            file = open(variable_name + '.eval.txt', 'w')
            file.write(string)
            file.close()
        else:
            warnings.warn(f'The variable {variable_name} could not be saved '
                          'because its type or contents is not supported.')


def save(filename, variable):
    """Save data in a KTK-supported way."""
    # Remove .zip extension if present (to obtain only the base name)
    if filename.lower().endswith('.zip'):
        filename = filename[0:-len('.zip')]

    try:
        shutil.rmtree('KTK_SAVE_TEMPORARY_FOLDER')
    except Exception:
        pass
    _os.mkdir('KTK_SAVE_TEMPORARY_FOLDER')
    _os.chdir('KTK_SAVE_TEMPORARY_FOLDER')
    _save_to_current_folder(variable, filename)
    _os.chdir('..')
    shutil.make_archive(filename, 'zip', 'KTK_SAVE_TEMPORARY_FOLDER')
    shutil.rmtree('KTK_SAVE_TEMPORARY_FOLDER')


def _load_current_folder():
    variable = dict()
    list_of_files = _os.listdir('.')
    for file_name in list_of_files:

        if file_name.endswith('.dict'):
            key = file_name[0:-len('.dict')]
            _os.chdir(file_name)
            variable[key] = _load_current_folder()
            _os.chdir('..')

        elif file_name.endswith('.str.txt'):
            key = file_name[0:-len('.str.txt')]
            file = open(file_name, 'r')
            variable[key] = file.read()
            file.close()

        elif file_name.endswith('.eval.txt'):
            key = file_name[0:-len('.eval.txt')]
            file = open(file_name, 'r')
            variable[key] = literal_eval(file.read())
            file.close()

        elif file_name.endswith('.ndarray.txt'):
            key = file_name[0:-len('.ndarray.txt')]
            dataframe = pd.read_csv(file_name, sep='\t',
                                    quoting=csv.QUOTE_NONNUMERIC)
            dict_of_arrays = dataframe_to_dict_of_arrays(dataframe)
            variable[key] = dict_of_arrays['Data']

        elif file_name.endswith('.TimeSeries'):
            key = file_name[0:-len('.TimeSeries')]

            _os.chdir(file_name)

            data = pd.read_csv('data.txt',
                               sep='\t', quoting=csv.QUOTE_NONNUMERIC)
            events = pd.read_csv('events.txt',
                                 sep='\t', quoting=csv.QUOTE_NONNUMERIC)
            info = pd.read_csv('info.txt',
                               sep='\t', quoting=csv.QUOTE_NONNUMERIC,
                               index_col=0)

            out = ktk.TimeSeries()

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

            variable[key] = out
            _os.chdir('..')

    return variable


def load(filename):
    """Load a KTK data file."""
    _os.mkdir('KTK_LOAD_TEMPORARY_FOLDER')
    shutil.unpack_archive(filename, extract_dir='KTK_LOAD_TEMPORARY_FOLDER')
    _os.chdir('KTK_LOAD_TEMPORARY_FOLDER')
    variable = _load_current_folder()
    _os.chdir('..')
    shutil.rmtree('KTK_LOAD_TEMPORARY_FOLDER')
    # Extract the first element (and only one) of this variable, to mirror
    # save function.
    for key, value in variable.items():
        return value


def dataframe_to_dict_of_arrays(dataframe):
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
            all_data_highest_indices.append([length-1])
        else:
            # Extract name and dimension
            data_name = one_column_name[0:opening_bracket_position]
            data_dimension = literal_eval(
                    '[' + str(length-1) + ',' +
                    one_column_name[opening_bracket_position+1:])

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
                np.array(unique_data_highest_index)+1, axis=0)

        # Convert the dataframe to a np.array, then reshape.
        new_data = sub_dataframe.to_numpy()
        new_data = np.reshape(new_data, unique_data_dimension)
        out[unique_data_name] = new_data

    return out


def dict_of_arrays_to_dataframe(dict_of_arrays):
    """
    Convert a numpy ndarray of any dimension to a pandas DataFrame.

    Parameters
    ----------
    dict_of_array : dict
        A dict that contains numpy arrays. Each array must have the same
        first dimension's size.

    Returns
    -------
    pd.DataFrame

    The rows in the output DataFrame correspond to the first dimension of the
    numpy arrays.
    - Vectors are converted to single-column DataFrames.
    - 2-dimensional arrays are converted to multi-columns DataFrames.
    - 3-dimensional (or more) arrays are also converted to DataFrames, but
      indices in brackets are added to the column names.

    Example
    -------
        >>> datadict = {'data': np.random.rand(10, 2, 2)}
        >>> dataframe = ktk.loadsave.dict_of_arrays_to_dataframe(datadict)

        >>> print(dataframe)
            data[0,0]   data[0,1]   data[1,0]   data[1,1]
        0   0.736891    0.902195    0.905907    0.065458
        1   0.875474    0.414270    0.696410    0.872808
        2   0.697806    0.542093    0.093780    0.394655
        3   0.132531    0.073543    0.036600    0.697872
        4   0.713446    0.672632    0.599467    0.211884
        5   0.860927    0.769096    0.278852    0.317487
        6   0.998223    0.831627    0.024960    0.960739
        7   0.573798    0.191601    0.797447    0.728639
        8   0.774073    0.942711    0.868428    0.667369
        9   0.530900    0.737578    0.224186    0.895926
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


def loadmat(filename):
    """
    Load a Matlab's MAT file.
    """
    # The MAT file should first be converted using Matlab's own runtime
    # engine, so that Matlab's timeseries are converted to structures.
    converted_filename = ktk.config['RootFolder'] + '/loadsave_converted.mat'

    if ktk.config['IsMac']:
        script_name = '/external/ktkMATtoPython/run_ktkMATtoPython.sh'
        runtime_path = '/Applications/MATLAB/MATLAB_Runtime/v91/'
    else:
        raise(NotImplementedError('loadmat is only available on Mac for now.'))

    _subprocess.run([ktk.config['RootFolder'] + script_name, runtime_path,
                      filename, converted_filename],
                      stderr=_subprocess.DEVNULL,
                      stdout=_subprocess.DEVNULL)

    # Now load it with scipy.io then delete file
    data = _spio.loadmat(converted_filename, struct_as_record=False,
                         squeeze_me=True)
    _os.remove(converted_filename)


    # Correct the keys
    data = _check_keys(data)

    # Assign contents to data
    data = data['contents']

    return data



def convert_to_timeseries(the_input):

    if isinstance(the_input, dict):
#c        print("This is a dict. Checking if it's a timeseries.")

        is_a_timeseries = False

        for the_key in the_input.keys():

            if isinstance(the_input[the_key], dict):
                if 'type' in the_input[the_key].keys():
                    if the_input[the_key]['type'] == 'timeseries':
                        is_a_timeseries = True
#                    else:
#                        is_a_timeseries = False
#                else:
#                    is_a_timeseries = False
#            else:
#                is_a_timeseries = False
        # end for the_key


        if is_a_timeseries == True:
            #After checking if each key is a timeseries, and it is, we get here.
            the_output = ktk.TimeSeries()
            for the_key in the_input.keys():
                try:
                    the_output.time = the_input[the_key]['Time']
                    the_data = the_input[the_key]['Data']
                    the_shape = the_data.shape
                    if len(the_shape) == 2:
                        the_output.data[the_key] = the_data.transpose((1,0))
                    elif len(the_shape) == 3:
                        the_output.data[the_key] = the_data.transpose((2,0,1))
                    else:
                        the_output.data[the_key] = the_data

                except:
                    pass

            return the_output
        else:
            print('This was not a timeseries.')

            for the_key in the_input.keys():
                print('  Now processing key %s' % the_key)
                the_input[the_key] = convert_to_timeseries(the_input[the_key])
            return the_input

    else:
        return the_input




def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], _spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, _spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict
