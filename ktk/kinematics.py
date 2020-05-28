#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020 Félix Chénier
#
# This file is not for redistribution.
"""
Kinematics analysis.
"""

import numpy as np
import ktk
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

    Parameters
    ----------
    filename : str
        Path of the C3D file

    Returns
    -------
    markers : TimeSeries
        A TimeSeries where each point in the C3D correspond to a data key in
        the TimeSeries.
    """
    # Create the reader
    reader = ezc3d(filename)

    # Create the output timeseries
    output = ktk.TimeSeries()

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

        output.data[label_name] = (point_factor *
                                   reader['data']['points'][:, i_label, :].T)

        output.add_data_info(label_name, 'Unit', 'm')

    # Create the timeseries time vector
    output.time = np.arange(n_frames) / point_rate

    return output


def read_n3d_file(filename, labels=[]):
    """
    Read markers from an Optitrak N3D file.

    Parameters
    ----------
    filename : str
        Path of the N3D file.
    labels : list of str (optional)
        Marker names

    Returns
    -------
    markers : TimeSeries
        A TimeSeries where each point in the N3D correspond to a data key in
        the TimeSeries.
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
        ts = ktk.TimeSeries(
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

    return ts


# def read_xml_file(file_name):
#
#    # Define a helper function that reads the file line by line until it finds
#    # one of the strings in a given list.
#    def read_until(list_strings):
#        while True:
#            one_line = fid.readline()
#            if len(one_line) == 0:
#                return(one_line)
#
#            for one_string in list_strings:
#                if one_line.find(one_string) >= 0:
#                    return(one_line)
#
#
#    # Open the file
#    if os.path.isfile(file_name) == False:
#        raise(FileNotFoundError)
#
#    fid = open(file_name, 'r')
#
#    # Reading loop
#    the_timeseries = []
#
#    while True:
#
#        # Wait for next label
#        one_string = read_until(['>Label :</Data>'])
#
#        if len(one_string) > 0:
#            # A new label was found
#
#            # Isolate the label name
#            one_string = read_until(['<Data'])
#            label_name = one_string[
#                    (one_string.find('"String">')+9):one_string.find('</Data>')]
#            print(label_name)
#
#            # Isolate the data format
#            one_string = read_until(['>Coords'])
#            one_string = one_string[
#                    (one_string.find('<Data>')+6):one_string.find('</Data>')]
#            data_unit = one_string[
#                    (one_string.find('x,y:')+4):one_string.find('; ')]
#
#            # Ignore the next data lines (header)
#            one_string = read_until(['<Data'])
#            one_string = read_until(['<Data'])
#            one_string = read_until(['<Data'])
#
#            # Find all data for this marker
#            time = np.array([1.0]) # Dummy init
#            data = np.zeros((1,2)) # Dummy init
#            sample_index = 0
#
#            while(True):
#
#                # Find the next x data
#                one_string = read_until(['<Data'])
#                one_string = one_string[
#                        (one_string.find('"Number">')+9):one_string.find('</Data>')]
#
#                try:
#                    # If it's a float, then add it.
#                    # Add a new row to time and data
#                    if sample_index > 0:
#                        time = np.block([time, np.array(1)])
#                        data = np.block([[data], [np.zeros((1,2))]])
#
#                    data[sample_index, 0] = float(one_string)
#
#                except:
#                    the_timeseries.append(ts.TimeSeries(name=label_name, data=data, time=time, dataunit=data_unit, timeunit='s'))
#                    break #No data
#
#                # Find the next y data
#                one_string = read_until(['<Data'])
#                one_string = one_string[
#                        (one_string.find('"Number">')+9):one_string.find('</Data>')]
#                data[sample_index, 1] = float(one_string)
#
#                # Find the next t data
#                one_string = read_until(['<Data'])
#                one_string = one_string[
#                        (one_string.find('">')+2):one_string.find('</Data>')]
#
#                if one_string.find(':') < 0:
#                    time[sample_index] = float(one_string) # milliseconds or #frame
#                else:
#                    index = one_string.find(':')
#                    hours = one_string[0:index]
#                    one_string = one_string[index+1:]
#
#                    index = one_string.find(':')
#                    minutes = one_string[0:index]
#                    one_string = one_string[index+1:]
#
#                    index = one_string.find(':')
#                    seconds = one_string[0:index]
#                    milliseconds = one_string[index+1:]
#
#                    time[sample_index] = (3600. * float(hours) +
#                        60. * float(minutes) + float(seconds) +
#                        (int(milliseconds) % 1000) / 1000)
#
#
#                sample_index = sample_index + 1
#
#        else:
#            # EOF
#            return(the_timeseries)


def create_rigid_body_config(markers, marker_names):
    """
    Create a rigid body configuration based on a static acquisition.

    Parameters
    ----------
    markers : TimeSeries
        Markers trajectories during the static acquisition.
    marker_names : list of str
        The markers that define the rigid body.

    Returns
    -------
    rigid_body_config : dict
        Dictionary with the following keys:
            - MarkerNames : the same as marker_names
            - LocalPoints : a 1x4xM array that indicates the local position of
                            each M marker in the created rigid body config.
    """
    n_samples = len(markers.time)
    n_markers = len(marker_names)

    # Construct the global points array
    global_points = np.empty((n_samples, 4, n_markers))

    for i_marker, marker in enumerate(marker_names):
        global_points[:, :, i_marker] = \
                markers.data[marker][:, :]

    # Remove samples where at least one marker is invisible
    global_points = global_points[~ktk.geometry.isnan(global_points)]

    rigid_bodies = ktk.geometry.create_reference_frames(global_points)
    local_points = ktk.geometry.get_local_coordinates(
            global_points, rigid_bodies)

    # Take the average
    local_points = np.mean(local_points, axis=0)
    local_points = local_points[np.newaxis]

    return {
            'LocalPoints' : local_points,
            'MarkerNames' : marker_names
            }


def register_markers(markers, rigid_body_configs, verbose=False):
    """
    Calculates the trajectory of rigid bodies.

    Calculates the trajectory of rigid bodies using
    `ktk.geometry.register_points`.

    Parameters
    ----------
    markers : TimeSeries
        Markers trajectories to calculate the rigid body trajectories on.
    rigid_body_configs : dict of dict
        A dict where each key is a rigid body configuration, and where
        each rigid body configuration is a dict with the following
        keys: 'MarkerNames' and 'LocalPoints'.
    verbose : bool (optional)
        True to print the rigid body being computed. Default is False.

    Returns
    -------
    rigid_bodies : TimeSeries
        TimeSeries where each data key is a Nx4x4 series of rigid
        transformations.
    """
    rigid_bodies = ktk.TimeSeries(time=markers.time,
                                  time_info=markers.time_info,
                                  events=markers.events)

    for rigid_body_name in rigid_body_configs:
        if verbose is True:
            print(f'Computing trajectory of rigid body {rigid_body_name}...')

        # Set local and global points
        local_points = rigid_body_configs[rigid_body_name]['LocalPoints']

        global_points = np.empty(
                (len(markers.time), 4, local_points.shape[2]))
        for i_marker in range(global_points.shape[2]):
            marker_name = rigid_body_configs[
                    rigid_body_name]['MarkerNames'][i_marker]
            if marker_name in markers.data:
                global_points[:, :, i_marker] = markers.data[marker_name]
            else:
                global_points[:, :, i_marker] = np.nan

        (local_points, global_points) = ktk.geometry.match_size(
                local_points, global_points)

        # Compute the rigid body trajectory
        rigid_bodies.data[rigid_body_name] = ktk.geometry.register_points(
                global_points, local_points)

    return rigid_bodies


def create_virtual_marker_config(markers, rigid_bodies,
                                 marker_name, rigid_body_name):
    """
    Create a virtual marker configuration based on a probing acquisition.

    Parameters
    ----------
    markers : TimeSeries
        Markers trajectories during the probing acquisition. This
        TimeSeries must have at least marker_name in its data keys.
    rigid_bodies : TimeSeries
        Rigid body trajectories during this probing acquisition. This
        TimeSeries must have at least rigid_body_name in its data keys.
    marker_name : str
        Name of the marker to express in local coordinates.
    rigid_body_name : str
        Name of the rigid body to express marker_name in relation to.

    Returns
    -------
    virtual_marker_config : dict
        Dictionary with the following keys:
            - RigidBodyName : Name of the virtual marker's rigid body
            - LocalPoint : Local position of this marker in the reference frame
                           defined by the rigid body RigidBodyName. LocalPoint
                           is expressed as a 1x4 array.

    """
    marker = markers.data[marker_name]
    rigid_body = rigid_bodies.data[rigid_body_name]

    local_points = ktk.geometry.get_local_coordinates(marker, rigid_body)
    to_keep = ~ktk.geometry.isnan(local_points)

    if np.all(to_keep is False):
        warnings.warn(f'There are no frame where both {marker_name} and'
                      f'{rigid_body_name} are visible at the same time.')

    local_points = local_points[to_keep]
    local_points = np.mean(local_points, axis=0)[np.newaxis]

    return {'RigidBodyName': rigid_body_name,
            'LocalPoint': local_points}
