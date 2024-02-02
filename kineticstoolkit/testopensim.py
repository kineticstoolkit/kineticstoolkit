#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2022 Laboratoire de recherche en mobilité et sport adapté

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
OpenSim wrappers for Kinetics Toolkit
"""

__author__ = "Karla Brottet, Félix Chénier"
__copyright__ = (
    "Copyright (C) 2022 Laboratoire de recherche en mobilité et sport adapté"
)
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"

import kineticstoolkit.lab as ktk
import pandas as pd


def read_sto(filename: str) -> ktk.TimeSeries:
    """Read a .sto opensim file."""
    return read_mot(filename)


def read_mot(filename: str):
    """Read a .mot opensim file."""
    # Initialisation
    header = 1
    nc = 0
    nr = 0

    # Read file line by line
    file_id = open(filename, "r")
    next_line = file_id.readline()

    # collection of the different parameters of the file
    while "endheader" not in next_line:
        if "datacolumns" in next_line:
            nc = int(next_line[next_line.index(" ") + 1 : len(next_line)])
        elif "datarows" in next_line:
            nr = int(next_line[next_line.index(" ") + 1 : len(next_line)])
        elif "nColumns" in next_line:
            nc = int(next_line[next_line.index("=") + 1 : len(next_line)])
        elif "nRows" in next_line:
            nr = int(next_line[next_line.index("=") + 1 : len(next_line)])

        next_line = file_id.readline()
        header = header + 1

    # Convert the motion file in Pandas DataFrame
    dt = pd.read_table(
        filename,
        header=header,
        nrows=nr,
        usecols=[i for i in range(nc)],
        skip_blank_lines=False,
        engine="python",
        index_col="time",
    )

    # Use kineticstoolkit to convert the dataframe in Timeseries
    ts = ktk.TimeSeries.from_dataframe(dt)

    return ts


def read_trc(filename: str) -> ktk.TimeSeries:
    """Read a .trc opensim file."""
    # Initialisation
    data_nc_read = 0
    data_nr_read = 0
    header = 0
    nc = 0
    nr = 0

    # Read file line by line
    file_id = open(filename, "r")
    next_line = file_id.readline()

    # collection of the different parameters
    while "Frame#" not in next_line:
        # get the location of the column number from the table
        if "NumMarkers" in next_line:
            lstnc = next_line.split()
            idnc = lstnc.index("NumMarkers")
            data_nc_read = True
        # get the location of the rows number from the table
        if "NumFrames" in next_line:
            lstnr = next_line.split()
            idnr = lstnr.index("NumFrames")
            data_nr_read = True

        next_line = file_id.readline()

        # writing the collected data
        if data_nc_read is True:
            lst = next_line.split()
            nc = int(lst[idnc])
            data_nc_read = False
        if data_nr_read is True:
            lst = next_line.split()
            nr = int(lst[idnr])
            data_nr_read = False

        header = header + 1

    # Convert the motion file in Pandas DataFrame
    df = pd.read_table(
        filename,
        header=(header),
        nrows=(nr + 1),
        usecols=[i for i in range(nc * 3 + 2)],
        skip_blank_lines=True,
        engine="python",
        index_col="Time",
    )

    # change the headers name to have an Nx3 array of the position signals
    colon_name = df.columns[0]
    for i in range(nc * 3 + 1):
        if "Unnamed" in df.columns[i]:
            old_name = df.columns[i]
            df = df.rename(columns={f"{old_name}": colon_name})
        else:
            colon_name = df.columns[i]

    # Use kineticstoolkit to convert the dataframe in Timeseries
    ts = ktk.TimeSeries.from_dataframe(df)

    return ts


def write_trc(filename: str, markers: ktk.TimeSeries):
    """
    Export a TimeSeries or marker positions to OpenSim's TRC file format.

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
    data_rate = markers.get_sample_rate()
    camera_rate = markers.get_sample_rate()
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
