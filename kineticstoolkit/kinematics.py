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
Provide functions related to kinematics analysis.
"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import kineticstoolkit.geometry as geometry
from kineticstoolkit import TimeSeries
from kineticstoolkit.decorators import unstable, directory, deprecated
from os.path import exists
from typing import Sequence, Dict, Any, List, Union
from copy import deepcopy

import numpy as np
import warnings
import struct  # To unpack data from N3D files

try:
    import ezc3d
except ModuleNotFoundError:
    warnings.warn(
        "Could not load ezc3d. Function kinematics.read_c3d_file "
        "will not work."
    )


def read_c3d_file(filename: str) -> TimeSeries:
    """
    Read markers from a C3D file.

    The markers positions are returned in a TimeSeries where each marker
    corresponds to a data key. Each marker position is expressed in this form:

    array([[x0, y0, z0, 1.], [x1, y1, z1, 1.], [x2, y2, z2, 1.], ...])

    Parameters
    ----------
    filename
        Path of the C3D file

    Notes
    -----
    - This function relies on `ezc3d`, which is available on
      conda-forge and on git-hub. Please install ezc3d before using
      read_c3d_file. https://github.com/pyomeca/ezc3d

    - The point unit must be either mm or m. In both cases, the final unit
      returned in the output TimeSeries is m.

    - As for any instrument, please check that your data loads correctly on
      your first use (e.g., sampling frequency, position unit). It is
      possible that read_c3d_file misses some corner cases.

    """
    # Create the reader
    if isinstance(filename, str) and exists(filename):
        reader = ezc3d.c3d(filename)
    else:
        raise FileNotFoundError(f"File {filename} was not found.")

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
        raise (ValueError("Point unit must be 'm' or 'mm'."))

    n_labels = len(labels)
    for i_label in range(n_labels):
        # Strip leading and ending spaces
        labels[i_label] = labels[i_label].strip()

        label_name = labels[i_label]

        output.data[label_name] = np.array(
            [point_factor, point_factor, point_factor, 1]
            * reader['data']['points'][:, i_label, :].T
        )

        output = output.add_data_info(label_name, 'Unit', 'm')

    # Create the timeseries time vector
    output.time = np.arange(n_frames) / point_rate

    return output


def write_c3d_file(filename: str, markers: TimeSeries) -> None:
    """
    Write a markers TimeSeries to a C3D file.

    Parameters
    ----------
    filename
        Path of the C3D file

    markers
        Markers trajectories, following the same form as the TimeSeries
        read by read_c3d_file.

    Notes
    -----
    This function relies on `ezc3d`. Please install ezc3d before using
    write_c3d_file. https://github.com/pyomeca/ezc3d

    """
    sample_rate = (markers.time.shape[0] - 1) / (
        markers.time[-1] - markers.time[0]
    )

    marker_list = []
    marker_data = np.zeros((4, len(markers.data), len(markers.time)))

    for i_marker, marker in enumerate(markers.data):
        marker_list.append(marker)
        marker_data[0, i_marker, :] = markers.data[marker][:, 0]
        marker_data[1, i_marker, :] = markers.data[marker][:, 1]
        marker_data[2, i_marker, :] = markers.data[marker][:, 2]
        marker_data[3, i_marker, :] = markers.data[marker][:, 3]

    # Load an empty c3d structure
    c3d = ezc3d.c3d()

    # Fill it with data
    c3d['parameters']['POINT']['RATE']['value'] = [sample_rate]
    c3d['parameters']['POINT']['LABELS']['value'] = tuple(marker_list)
    c3d['data']['points'] = marker_data

    # Add a custom parameter to the POINT group
    c3d.add_parameter('POINT', 'UNITS', 'm')

    # Write the data
    c3d.write(filename)


def read_n3d_file(filename: str, labels: Sequence[str] = []) -> TimeSeries:
    """
    Read markers from an NDI N3D file.

    The markers positions are returned in a TimeSeries where each marker
    corresponds to a data key. Each marker position is expressed in this form:

    array([[x0, y0, z0, 1.], [x1, y1, z1, 1.], [x2, y2, z2, 1.], ...])

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
                if data < -1e25:  # technically, it is -3.697314e+28
                    data = np.NaN
                ndi_array[i_frame, i_column] = data

        # Conversion from mm to meters
        ndi_array /= 1000

        # Transformation to a TimeSeries
        ts = TimeSeries(
            time=np.linspace(
                0, n_frames / collection_frame_frequency, n_frames
            )
        )

        for i_marker in range(n_markers):
            if labels != []:
                label = labels[i_marker]
            else:
                label = f'Marker{i_marker}'

            ts.data[label] = np.block(
                [
                    [
                        ndi_array[:, 3 * i_marker: 3 * i_marker + 3],
                        np.ones((n_frames, 1)),
                    ]
                ]
            )
            ts = ts.add_data_info(label, 'Unit', 'm')

    return ts


# def read_xml_file(filename):
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
#    if os.path.isfile(filename) == False:
#        raise(FileNotFoundError)
#
#    fid = open(filename, 'r')
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


def create_cluster(
        markers: TimeSeries,
        /,
        marker_names: Sequence[str]) -> Dict[str, np.ndarray]:
    """
    Create a cluster definition based on a static acquisition.

    Warning
    -------
    This function, which has been introduced in 0.7, is still experimental and
    may change signature or behaviour in the future.

    Parameters
    ----------
    markers
        Markers trajectories during a static acquisition.
    marker_names
        The markers that define the cluster.

    Returns
    -------
    Dict
        Dictionary where each entry represents the local position of a marker
        in an arbitrary coordinate system.

    """
    n_samples = len(markers.time)
    n_markers = len(marker_names)

    # Construct the global points array
    global_points = np.empty((n_samples, 4, n_markers))

    for i_marker, marker in enumerate(marker_names):
        global_points[:, :, i_marker] = markers.data[marker][:, :]

    # Remove samples where at least one marker is invisible
    global_points = global_points[~geometry.isnan(global_points)]

    rigid_bodies = geometry.create_frames(
        origin=global_points[:, :, 0],
        x=global_points[:, :, 1] - global_points[:, :, 0],
        xy=global_points[:, :, 2] - global_points[:, :, 0],
    )
    local_points = geometry.get_local_coordinates(global_points, rigid_bodies)

    # Take the average
    local_points = np.array(np.mean(local_points, axis=0))
    local_points = local_points[np.newaxis]

    # Create the output
    output = {}
    for i_marker, marker_name in enumerate(marker_names):
        output[marker_name] = local_points[:, :, i_marker]

    return output


def extend_cluster(
        markers: TimeSeries,
        /,
        cluster: Dict[str, np.ndarray],
        new_point: str) -> Dict[str, np.ndarray]:
    """
    Add a point to an existing cluster.

    Warning
    -------
    This function, which has been introduced in 0.7, is still experimental and
    may change signature or behaviour in the future.

    Parameters
    ----------
    markers
        TimeSeries that includes the new point trajectory, along with point
        trajectories from the cluster definition.
    cluster
        The source cluster to add a new point to.
    new_point
        The name of the point to add (data key of the markers TimeSeries).

    Returns
    -------
    Dict[str, np.ndarray]
        The cluster with the added point.

    """
    cluster = deepcopy(cluster)
    frames = _track_cluster_frames(markers, cluster)
    local_coordinates = geometry.get_local_coordinates(
        markers.data[new_point], frames)
    cluster[new_point] = np.nanmean(local_coordinates, axis=0)[np.newaxis]
    return cluster


def track_cluster(
        markers: TimeSeries,
        /,
        cluster: Dict[str, np.ndarray],
        *,
        include_lcs: bool = False,
        lcs_name: str = 'LCS') -> TimeSeries:
    """
    Fit a cluster to a TimeSeries of point trajectories.

    Fits a cluster to a TimeSeries and reconstructs a solidified version of
    all the points defined in this cluster.

    Warning
    -------
    This function, which has been introduced in 0.7, is still experimental and
    may change signature or behaviour in the future.

    Parameters
    ----------
    markers
        A TimeSeries that contains point trajectories as Nx4 arrays.
    cluster
        A cluster definition as returned by ktk.kinematics.create_cluster().
    include_lcs
        Optional. If True, return an additional entry in the output
        TimeSeries, that is the Nx4x4 frame series corresponding to the
        tracked cluster's local coordinate system. The default is False.
    lcs_name
        Optional. Name of the TimeSeries data entry for the tracked local
        coordinate system. The default is 'LCS'.

    Returns
    -------
    TimeSeries
        A TimeSeries with the trajectories of all cluster points.

    """
    out = markers.copy(copy_data=False, copy_data_info=False)

    # Track the cluster
    frames = _track_cluster_frames(markers, cluster)

    for marker in cluster:
        out.data[marker] = geometry.get_global_coordinates(
            cluster[marker], frames)

    if include_lcs:
        out.data[lcs_name] = frames

    return out


def _track_cluster_frames(
        markers: TimeSeries,
        cluster: Dict[str, np.ndarray]) -> np.ndarray:
    """Track a cluster and return its frame series."""
    # Set local and global points
    marker_names = cluster.keys()
    stacked_local_points = np.dstack(
        [cluster[_] for _ in marker_names])

    global_points = np.empty(
        (len(markers.time), 4, stacked_local_points.shape[2])
    )

    for i_marker, marker_name in enumerate(marker_names):
        if marker_name in markers.data:
            global_points[:, :, i_marker] = markers.data[marker_name]
        else:
            global_points[:, :, i_marker] = np.nan

    (stacked_local_points, global_points) = geometry._match_size(
        stacked_local_points, global_points
    )

    # Track the cluster
    frames = geometry.register_points(global_points, stacked_local_points)
    return frames


@unstable
def write_trc_file(markers: TimeSeries, filename: str) -> None:
    """
    Export a markers TimeSeries to OpenSim's TRC file format.

    Parameters
    ----------
    markers
        Markers trajectories.

    filename
        Name of the trc file to create.

    """
    markers = markers.copy()
    markers.fill_missing_samples(0)

    n_markers = len(markers.data)
    n_frames = markers.time.shape[0]
    data_rate = n_frames / (markers.time[1] - markers.time[0])
    camera_rate = data_rate
    units = 'm'

    # Open file
    with open(filename, 'w') as fid:
        fid.write(f'PathFileType\t4\t(X/Y/Z)\t{filename}\n')
        fid.write(
            'DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t'
            'OrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n'
        )
        fid.write(
            f'{data_rate}\t{camera_rate}\t{n_frames}\t{n_markers}\t'
            f'{units}\t{data_rate}\t1\t{n_frames}\n'
        )

        # Write marker names
        fid.write('Frame#\tTime')
        for key in markers.data:
            fid.write(f'\t{key}\t\t')
        fid.write('\n')

        # Write coordinate names
        fid.write('\t')
        for i, key in enumerate(markers.data):
            fid.write(f'\tX{i+1}\tY{i+1}\tZ{i+1}')
        fid.write('\n\n')

        # Write trajectories
        for i_frame in range(n_frames):
            fid.write(f'{i_frame+1}\t' '{:.3f}'.format(markers.time[i_frame]))
            for key in markers.data:
                fid.write(
                    '\t{:.5f}'.format(markers.data[key][i_frame, 0])
                    + '\t{:.5f}'.format(markers.data[key][i_frame, 1])
                    + '\t{:.5f}'.format(markers.data[key][i_frame, 2])
                )
            fid.write('\n')


# %% Deprecated
@unstable
@deprecated(since='master', until='January 2023', details="Use create_cluster().")
def define_rigid_body(
        kinematics: TimeSeries,
        marker_names: Sequence[str]) -> Dict[str, np.ndarray]:
    """
    Create a generic rigid body definition based on a static acquisition.

    Parameters
    ----------
    kinematics
        Markers trajectories during a static acquisition.
    marker_names
        The markers that define the rigid body.

    Returns
    -------
    Dict
        Dictionary where each entry represents the local position of a point
        (e.g., marker name). The key is the name of the point, the value is a
        1x4 array that indicates the local position of this point in the rigid
        body's local coordinate system.

    """
    return create_cluster(kinematics, marker_names)


@unstable
@deprecated(since='master', until='January 2023', details="Use track_clusters().")
def track_rigid_body(
        kinematics: TimeSeries,
        /,
        local_points: Dict[str, np.ndarray],
        label: str = 'Trajectory',
        *,
        include_rigid_body: bool = True,
        include_markers: bool = False) -> TimeSeries:
    """
    Track a rigid body from markers trajectories.

    This function tracks the specified rigid body in a TimeSeries that
    contains the required markers, and adds the tracked rigid body to a copy
    of the input TimeSeries as a Nx4x4 series of frames.

    Parameters
    ----------
    kinematics
        TimeSeries that contains at least the trajectories of the markers
        specified in rigid_body_definition.
    local_points
        A dict where each key is a point name and its corresponding value is
        its local coordinates, as returned by
        `ktk.kinematics.define_rigid_body()`.
    label
        Name of the rigid body, that will be the data key in the output
        TimeSeries.
    include_rigid_body: Optional.
        Include the tracked rigid body in the output TimeSeries as an Nx4x4
        series. The default is True.
    include_markers: Optional.
        Include the reconstructed markers in the output TimeSeries as a Nx4
        series. Every marker of the rigid body definition is recontructed. The
        default is False.

    Returns
    -------
    TimeSeries
        A TimeSeries that contains the trajectory of the tracked rigid body.
    """
    return track_rigid_bodies(
        kinematics,
        {label: local_points},
        include_rigid_bodies=include_rigid_body,
        include_markers=include_markers
    )


@unstable
@deprecated(since='master', until='January 2023', details="No replacement yet.")
def track_rigid_bodies(
        markers: TimeSeries,
        /,
        definitions: Dict[str, Dict[str, np.ndarray]],
        *,
        include_rigid_bodies: bool = True,
        include_markers: bool = False) -> TimeSeries:
    """
    Track rigid bodies from markers trajectories.

    Parameters
    ----------
    markers
        TimeSeries that contains the trajectories of the markers specified
        in the rigid body definitions.
    definitions
        A dict where each key is a rigid body name and its value is a dict
        of local coordinates as returned by
        `ktk.kinematics.define_rigid_body()`.
    include_rigid_bodies: Optional.
        Include the tracked rigid bodies in the output TimeSeries as an Nx4x4
        series. The default is True.
    include_markers: Optional.
        Include the reconstructed markers in the output TimeSeries as a Nx4
        series. Every marker of the rigid body definitions is recontructed. The
        default is False.

    Returns
    -------
    TimeSeries
        A TimeSeries that contains the trajectory of the tracked rigid bodies.

    """
    out = markers.copy(copy_data=False, copy_data_info=False)
    for cluster in definitions:
        out.data[cluster] = _track_cluster_frames(
            markers, definitions[cluster])
    return out


@unstable
@deprecated(since='master', until='January 2023', details="Use extend_cluster().")
def define_local_position(
        kinematics: TimeSeries,
        source_name: str,
        rigid_body_name: str) -> np.ndarray:
    """
    Define a point's local position based on a static or probing acquisition.

    Ideally, the acquisition should be a short static acquisition and every
    required marker must be visible at the same time at least once.

    Parameters
    ----------
    kinematics
        TimeSeries of the static or probing acquisition, that contains the
        required markers and/or rigid bodies.
    source_name
        Name of the marker or rigid body to express in local coordinates. This
        name must be a data key in the kinematics TimeSeries and should refer
        to a marker trajectory (Nx4 array) or a rigid body trajectory (Nx4x4
        array).
    rigid_body_name
        Name of the reference rigid body. This name must be a data key in the
        kinematics TimeSeries and should refer to a rigid body trajectory
        (Nx4x4 array).

    Returns
    -------
    np.ndarray
        The position of the marker or rigid body origin in the local coordinate
        system, as an Nx4 array.

    """
    marker_trajectory = kinematics.data[source_name]
    if marker_trajectory.shape[1:] == (4,):
        pass  # Marker trajectory
    elif marker_trajectory.shape[1:] == (4, 4):
        marker_trajectory = marker_trajectory[:, :, 3]  # frame trajectory

    rigid_body_trajectory = kinematics.data[rigid_body_name]

    local_points = geometry.get_local_coordinates(
        marker_trajectory, rigid_body_trajectory)
    to_keep = ~geometry.isnan(local_points)

    if np.all(to_keep is False):
        warnings.warn(
            "There are no frame where both the marker and body "
            "are visible at the same time.")

    local_points = local_points[to_keep]
    local_points = np.mean(local_points, axis=0)[np.newaxis]

    return local_points

# %% Footer


module_locals = locals()


def __dir__():
    return directory(module_locals)
