#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020 Félix Chénier
#
# This file is not for redistribution.
"""
Provides functions related to kinematics analysis.
"""

from ktk.timeseries import TimeSeries

import numpy as np
import warnings
import struct  # To unpack data from N3D files

try:
    from ezc3d import c3d as ezc3d
except ModuleNotFoundError:
    warnings.warn('Could not load ezc3d. Function kinematics.read_c3d_file '
                  'will not work.')


def read_c3d_file(filename):
    """
    Read markers from a C3D file.

    The markers positions are returned in a TimeSeries where each marker
    corresponds to a data key. Each marker position is expressed in this form:

            array([[x0, y0, z0, 1.],
                   [x1, y1, z1, 1.],
                   [x2, y2, z2, 1.],
                   [...           ]])

    Parameters
    ----------
    filename : str
        Path of the C3D file

    Returns
    -------
    TimeSeries

    Notes
    -----
    - This function relies on pariterre's `ezc3d`, which is available on
      conda-forge and on git-hub. Please install ezc3d before using
      read_c3d_file. https://github.com/pyomeca/ezc3d

    - The point unit must be either mm or m. In both cases, the final unit
      returned in the output TimeSeries is m.

    - As for any instrument, please check that your data loads correctly on
      your first use (e.g., sampling frequency, position unit). It is highly
      possible that read_c3d_file misses some corner cases.

    """
    # Create the reader
    reader = ezc3d(filename)

    # Create the output timeseries
    output = TimeSeries()

    # Get the marker label names and create a timeseries data entry for each
    # Get the labels
    labels = reader['parameters']['POINT']['LABELS']['value']
    n_frames = reader['parameters']['POINT']['FRAMES']['value'][0]
    point_rate = reader['parameters']['POINT']['RATE']['value'][0]
    point_unit = reader['parameters']['POINT']['UNITS']['value'][0]

    if point_unit == 'mm':
        point_factor = 0.001
    elif point_unit == 'm':
        point_factor = 1
    else:
        raise(ValueError("Point unit must be 'm' or 'mm'."))

    n_labels = len(labels)
    for i_label in range(n_labels):
        # Strip leading and ending spaces
        labels[i_label] = labels[i_label].strip()

        label_name = labels[i_label]

        output.data[label_name] = (
            [point_factor, point_factor, point_factor, 1] *
            reader['data']['points'][:, i_label, :].T)

        output.add_data_info(label_name, 'Unit', 'm')

    # Create the timeseries time vector
    output.time = np.arange(n_frames) / point_rate

    return output


def read_n3d_file(filename, labels=[]):
    """
    Read markers from an NDI N3D file.

    The markers positions are returned in a TimeSeries where each marker
    corresponds to a data key. Each marker position is expressed in this form:

            array([[x0, y0, z0, 1.],
                   [x1, y1, z1, 1.],
                   [x2, y2, z2, 1.],
                   [...           ]])

    Parameters
    ----------
    filename : str
        Path of the N3D file.
    labels : list of str (optional)
        Marker names

    Returns
    -------
    TimeSeries

    """
    with open(filename, 'rb') as fid:
        _ = fid.read(1)  # 32
        n_markers = struct.unpack('h', fid.read(2))[0]
        n_data_per_marker = struct.unpack('h', fid.read(2))[0]
        n_columns = n_markers * n_data_per_marker

        n_frames = struct.unpack('i', fid.read(4))[0]

        collection_frame_frequency = struct.unpack('f', fid.read(4))[0]
        user_comments = struct.unpack('60s', fid.read(60))[0]
        system_comments = struct.unpack('60s', fid.read(60))[0]
        file_description = struct.unpack('30s', fid.read(30))[0]
        cutoff_filter_frequency = struct.unpack('h', fid.read(2))[0]
        time_of_collection = struct.unpack('8s', fid.read(8))[0]
        _ = fid.read(2)
        date_of_collection = struct.unpack('8s', fid.read(8))[0]
        extended_header = struct.unpack('73s', fid.read(73))[0]

        # Read the rest and put it in an array
        ndi_array = np.ones((n_frames, n_columns)) * np.NaN

        for i_frame in range(n_frames):
            for i_column in range(n_columns):
                data = struct.unpack('f', fid.read(4))[0]
                if (data < -1E25):  # technically, it is -3.697314e+28
                    data = np.NaN
                ndi_array[i_frame, i_column] = data

        # Conversion from mm to meters
        ndi_array /= 1000

        # Transformation to a TimeSeries
        ts = TimeSeries(
                time=np.linspace(0, n_frames / collection_frame_frequency,
                                 n_frames))

        for i_marker in range(n_markers):
            if labels != []:
                label = labels[i_marker]
            else:
                label = f'Marker{i_marker}'

            ts.data[label] = np.block([[
                    ndi_array[:, 3*i_marker:3*i_marker+3],
                    np.ones((n_frames, 1))]])
            ts.add_data_info(label, 'Unit', 'm')

    return ts
