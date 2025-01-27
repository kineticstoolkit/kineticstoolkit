#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020-2025 Félix Chénier

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Provide functions related to kinematics analysis."""
__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2025 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import kineticstoolkit.geometry as geometry
from kineticstoolkit import TimeSeries
from kineticstoolkit.typing_ import check_param

import numpy as np
import warnings


def __dir__():
    return [
        "create_cluster",
        "extend_cluster",
        "track_cluster",
    ]


def create_cluster(
    markers: TimeSeries, /, names: list[str]
) -> dict[str, np.ndarray]:
    """
    Create a cluster definition based on a static acquisition.

    Parameters
    ----------
    markers
        Marker trajectories during a static acquisition.
    names
        The markers that define the cluster.

    Returns
    -------
    dict
        Dictionary where each entry represents the local position of a marker
        in an arbitrary coordinate system.

    Note
    -----
    0.10.0: Parameter `marker_names` was changed to `names`

    See Also
    --------
    ktk.kinematics.extend_cluster
    ktk.kinematics.track_cluster

    """
    check_param("markers", markers, TimeSeries)
    names = list(names)
    check_param("names", names, list, contents_type=str)

    n_samples = len(markers.time)
    n_markers = len(names)

    # Construct the global points array
    global_points = np.empty((n_samples, 4, n_markers))

    for i_marker, marker in enumerate(names):
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
    for i_marker, name in enumerate(names):
        output[name] = local_points[:, :, i_marker]

    return output


def extend_cluster(
    markers: TimeSeries, /, cluster: dict[str, np.ndarray], name: str
) -> dict[str, np.ndarray]:
    """
    Add a point to an existing cluster.

    Parameters
    ----------
    markers
        TimeSeries that includes the new point trajectory, along with point
        trajectories from the cluster definition.
    cluster
        The source cluster to add a new point to.
    name
        The name of the point to add (data key of the markers TimeSeries).

    Returns
    -------
    dict[str, np.ndarray]
        A copy of the initial cluster, with the added point.

    Note
    ----
    0.10.0: Parameter `new_point` was changed to `name`

    See Also
    --------
    ktk.kinematics.create_cluster
    ktk.kinematics.track_cluster

    """
    check_param("markers", markers, TimeSeries)
    check_param("cluster", cluster, dict, key_type=str)
    for key in cluster:
        cluster[key] = np.array(cluster[key])
    check_param("name", name, str)

    frames = _track_cluster_frames(markers, cluster)
    local_coordinates = geometry.get_local_coordinates(
        markers.data[name], frames
    )

    if np.all(geometry.isnan(local_coordinates)):
        warnings.warn(
            f"The point {name} was invisible during the entire TimeSeries."
        )
    else:
        cluster[name] = np.nanmean(local_coordinates, axis=0)[np.newaxis]

    return cluster


def track_cluster(
    markers: TimeSeries,
    /,
    cluster: dict[str, np.ndarray],
    *,
    include_lcs: bool = False,
    lcs_name: str = "LCS",
) -> TimeSeries:
    """
    Fit a cluster to a TimeSeries of point trajectories.

    This function fits a cluster to a TimeSeries and reconstructs a solidified
    version of all the points defined in this cluster.

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
        coordinate system. The default is "LCS".

    Returns
    -------
    TimeSeries
        A TimeSeries with the trajectories of all cluster points.

    See Also
    --------
    ktk.kinematics.create_cluster
    ktk.kinematics.track_cluster

    """
    check_param("markers", markers, TimeSeries)
    check_param("cluster", cluster, dict, key_type=str)
    for key in cluster:
        cluster[key] = np.array(cluster[key])
    check_param("include_lcs", include_lcs, bool)
    check_param("lcs_name", lcs_name, str)

    out = markers.copy(copy_data=False, copy_data_info=False)
    unit = _get_marker_unit(markers)

    # Track the cluster
    frames = _track_cluster_frames(markers, cluster)

    for marker in cluster:
        out.data[marker] = geometry.get_global_coordinates(
            cluster[marker], frames
        )
        if unit is not None:
            out.add_data_info(marker, "Unit", unit, in_place=True)

    if include_lcs:
        out.data[lcs_name] = frames

    return out


def _track_cluster_frames(
    markers: TimeSeries, cluster: dict[str, np.ndarray]
) -> np.ndarray:
    """Track a cluster and return its frame series."""
    # Set local and global points
    marker_names = cluster.keys()
    stacked_local_points = np.dstack(
        [np.array(cluster[_]) for _ in marker_names]
    )

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


def _get_marker_unit(markers: TimeSeries) -> None | str:
    """Get markers unit, raise ValueError if not all have the same unit."""
    unit = None
    for marker in markers.data:
        try:
            this_unit = markers.data_info[marker]["Unit"]
        except KeyError:
            this_unit = None

        if this_unit is not None:
            if unit is None:
                unit = this_unit
            else:
                if unit != this_unit:
                    raise ValueError(
                        "All markers must have the same unit. However, this "
                        f"TimeSeries has both {unit} and {this_unit}."
                    )
    return unit


def write_trc_file(markers: TimeSeries, /, filename: str) -> None:
    """
    Export a markers TimeSeries to OpenSim's TRC file format.

    Parameters
    ----------
    markers
        Marker trajectories.

    filename
        Name of the TRC file to create.

    Warning
    -------
    This function may eventually move either to the base namespace like
    write_c3d_file, or to an OpenSim extension that is currently being
    developed.

    """
    check_param("markers", markers, TimeSeries)
    check_param("filename", filename, str)

    markers = markers.copy()
    markers.fill_missing_samples(0)

    n_markers = len(markers.data)
    n_frames = markers.time.shape[0]
    data_rate = n_frames / (markers.time[1] - markers.time[0])
    camera_rate = data_rate
    units = "m"

    # Open file
    with open(filename, "w") as fid:
        fid.write(f"PathFileType\t4\t(X/Y/Z)\t{filename}\n")
        fid.write(
            "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t"
            "OrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n"
        )
        fid.write(
            f"{data_rate}\t{camera_rate}\t{n_frames}\t{n_markers}\t"
            f"{units}\t{data_rate}\t1\t{n_frames}\n"
        )

        # Write marker names
        fid.write("Frame#\tTime")
        for key in markers.data:
            fid.write(f"\t{key}\t\t")
        fid.write("\n")

        # Write coordinate names
        fid.write("\t")
        for i, key in enumerate(markers.data):
            fid.write(f"\tX{i+1}\tY{i+1}\tZ{i+1}")
        fid.write("\n\n")

        # Write trajectories
        for i_frame in range(n_frames):
            fid.write(f"{i_frame+1}\t" "{:.3f}".format(markers.time[i_frame]))
            for key in markers.data:
                fid.write(
                    "\t{:.5f}".format(markers.data[key][i_frame, 0])
                    + "\t{:.5f}".format(markers.data[key][i_frame, 1])
                    + "\t{:.5f}".format(markers.data[key][i_frame, 2])
                )
            fid.write("\n")
