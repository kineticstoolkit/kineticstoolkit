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
"""
Provide functions to load and save data.

The classes defined in this module are accessible directly from the
toplevel Kinetics Toolkit namespace (i.e. ktk.load, ktk.save).

"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2025 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"

from kineticstoolkit.timeseries import TimeSeries
import kineticstoolkit.dev.kinetics as kinetics
import kineticstoolkit.config
from kineticstoolkit.typing_ import check_param

import os
import numpy as np
import pandas as pd
import ezc3d
from typing import Any
import warnings
import shutil
import json
from datetime import datetime
import time
import getpass
import zipfile


def __dir__():  # pragma: no cover
    return ["save", "load"]


def _ezc3d_to_dict(c3d) -> dict[str, Any]:
    """Create a TimeSeries info dict based on an ezc3d class."""
    out = dict()  # type: dict[str, Any]

    try:
        for param_key in c3d["parameters"]:
            out[param_key] = dict()
            for prop_key in c3d["parameters"][param_key]:
                if prop_key.startswith("_"):
                    continue
                out[param_key][prop_key] = c3d["parameters"][param_key][
                    prop_key
                ]["value"]

    except KeyError:
        pass

    return out


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
    - pandas.DataFrame (basic DataFrames, e.g., without multi-indexing.)
    - pandas.Series
    - ktk.TimeSeries

    Parameters
    ----------
    filename:
        Name of the file to save to (e.g., "file.ktk.zip")
    variable:
        The variable to save.

    Returns
    -------
    None

    Caution
    -------
    Tuples are also supported but will be loaded back as lists, without
    warning. Complex Pandas Series (e.g., series or different types) may not be
    supported: only name, index and data are saved. Complex Pandas DataFrames
    (e.g., multiindex, columns of different types) may not be supported:
    only columns, index, and data are saved.

    See Also
    --------
    ktk.load

    """
    check_param("filename", filename, str)

    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return {"class__": "numpy.array", "value": obj.tolist()}

            elif isinstance(obj, TimeSeries):
                out = {}
                out["class__"] = "ktk.TimeSeries"
                out["time"] = obj.time.tolist()
                out["info"] = obj.info
                out["data"] = {}
                for key in obj.data:
                    out["data"][key] = obj.data[key].tolist()
                out["events"] = []
                for event in obj.events:
                    out["events"].append(
                        {
                            "time": event.time,
                            "name": event.name,
                        }
                    )
                return out

            elif isinstance(obj, pd.Series):
                return {
                    "class__": "pandas.Series",
                    "name": str(obj.name),
                    "index": obj.index.tolist(),
                    "data": obj.tolist(),
                }

            elif isinstance(obj, pd.DataFrame):
                return {
                    "class__": "pandas.DataFrame",
                    "columns": obj.columns.tolist(),
                    "index": obj.index.tolist(),
                    "data": obj.to_numpy().tolist(),
                }

            elif isinstance(obj, complex):
                return {
                    "class__": "complex",
                    "real": obj.real,
                    "imag": obj.imag,
                }

            else:
                return super().default(obj)

    now = datetime.now()
    if kineticstoolkit.config.is_pc:
        computer = "PC"
    elif kineticstoolkit.config.is_mac:
        computer = "Mac"
    elif kineticstoolkit.config.is_linux:
        computer = "Linux"
    else:
        computer = "Unknown"

    metadata = {
        "Software": "Kinetics Toolkit",
        "Version": kineticstoolkit.config.version,
        "Computer": computer,
        "FileFormat": 1.0,
        "SaveDate": now.strftime("%Y-%m-%d"),
        "SaveTime": now.strftime("%H:%M:%S"),
        "User": getpass.getuser(),
    }

    # Save
    temp_folder = (
        kineticstoolkit.config.temp_folder + "/save" + str(time.time())
    )

    try:
        shutil.rmtree(temp_folder)
    except:
        pass
    os.mkdir(temp_folder)

    with open(temp_folder + "/metadata.json", "w") as fid:
        json.dump(metadata, fid, indent="\t")

    with open(temp_folder + "/data.json", "w") as fid:
        json.dump(variable, fid, cls=CustomEncoder, indent="\t")

    shutil.make_archive(temp_folder, "zip", temp_folder)
    shutil.move(temp_folder + ".zip", filename)
    shutil.rmtree(temp_folder)


def _load_object_hook(obj):
    if "class__" in obj:
        to_class = obj["class__"]
        if to_class == "numpy.array":
            return np.array(obj["value"])

        elif to_class == "ktk.TimeSeries":
            out = TimeSeries()
            out.time = np.array(obj["time"])

            # Post-0.17
            if "info" in obj:
                out.info = obj["info"]
            # Pre-0.17
            if "time_info" in obj:
                out.time_info = obj["time_info"]
            if "data_info" in obj:
                out.data_info = obj["data_info"]

            for key in obj["data"]:
                out.data[key] = np.array(obj["data"][key])
            for event in obj["events"]:
                out = out.add_event(event["time"], event["name"])
            return out

        elif to_class == "pandas.DataFrame":
            return pd.DataFrame(
                obj["data"],
                columns=obj["columns"],
                index=obj["index"],
            )

        elif to_class == "pandas.Series":
            return pd.Series(
                obj["data"],
                name=obj["name"],
                index=obj["index"],
            )

        elif to_class == "complex":
            return obj["real"] + obj["imag"] * 1j

        else:
            warnings.warn(
                f'The "{to_class}" class is not supported by '
                "this version of Kinetics Toolkit. Please check "
                "that Kinetics Toolkit is up to date."
            )
            return obj

    else:
        return obj


def load(filename: str, *, include_metadata: bool = False) -> Any:
    """
    Load a ktk.zip file.

    Load a data file as saved using the ``ktk.save`` function.

    Usage::

        data = ktk.load(filename)
        data, metadata = ktk.load(filename, include_metadata=True)

    Parameters
    ----------
    filename
        The path of the zip file to load.
    include_metadata
        Optional. If True, the output is a tuple of this form:
        (data, metadata).

    Returns
    -------
    Any
        The loaded variable.

    See Also
    --------
    ktk.save

    """
    check_param("filename", filename, str)
    check_param("include_metadata", include_metadata, bool)

    archive = zipfile.ZipFile(filename, "r")

    data = json.loads(
        archive.read("data.json").decode(), object_hook=_load_object_hook
    )

    if include_metadata:
        metadata = json.loads(
            archive.read("metadata.json").decode(),
            object_hook=_load_object_hook,
        )
        return data, metadata

    else:
        return data


def read_c3d(
    filename: str,
    *,
    include_event_context: bool = False,
    convert_point_unit: bool | None = None,
    convert_forceplate_moment_unit: bool = True,
    convert_forceplate_position_unit: bool = True,
    **kwargs,
) -> dict[str, TimeSeries]:
    """
    Read point, analog and rotation data from a C3D file.

    If available, point positions are returned in `output["Points"]` as a
    TimeSeries, where each point corresponds to a data key. Each point position
    is expressed as an Nx4 point series::

        [
            [x0, y0, z0, 1.0],
            [x1, y1, z1, 1.0],
            [x2, y2, z2, 1.0],
            ...,
        ]

    If available, analog data is returned in `output["Analogs"]` as a
    TimeSeries, where each analog signal is expressed as a unidimensional
    array of length N.

    If available, force platform data is returned in `output["ForcePlatforms"]`
    as a TimeSeries containing the following keys where FPi means Force
    Platform i, i being the index of the platform (e.g., FP0, FP1, etc.):

        - FPi_Force: Ground reaction force components in global coordinates,
          as an Nx4 vector series [[Fx, Fy, Fz, 0.0], ...].
        - FPi_Moment: Ground reaction moment components in global
          coordinates, expressed at the origin of the force platform, as an
          Nx4 vector series [[Mx, My, Mz, 0.0], ...].
        - FPi_MomentAtCOP: Ground reaction moment components in global
          coordinates, expressed at the center of pressure, as an Nx4
          vector series.
        - FPi_COP: Centre of pressure in global coordinates as n Nx4 point
          series.
        - FPi_LCS: Local coordinate system of the force platform, expressed in
          global coordinates as an Nx4x4 transform series. The origin is at
          the middle point of the four corners, with z pointing down.
        - FPi_Corner1: Coordinates of the first corner (+x, +y) in global
          coordinates, as an Nx4 point series.
        - FPi_Corner2: Coordinates of the second corner (-x, +y) in global
          coordinates, as an Nx4 point series.
        - FPi_Corner3: Coordinates of the third corner (-x, -y) in global
          coordinates, as an Nx4 point series.
        - FPi_Corner4: Coordinates of the fourth corner (+x, -y) in global
          coordinates, as an Nx4 point series.

    If available, rigid body orientations are returned in `output["Rotations"]`
    as a TimeSeries, where each body orientation is expressed as an Nx4x4
    transforms series.

    The C3D parameters are returned in `output["C3DParameters"]`. Note that
    this content is dependent on each recording environment and software. It
    could also change between versions of the C3D parser (ezc3d).

    Some software stores calculated values such as angles, forces, moments,
    powers, etc. into the C3D file. Storing these data is software-specific
    and is not standardized in the C3D file format (https://www.c3d.org).
    This function reads these values as points regardless of their nature.

    Parameters
    ----------
    filename
        Path of the C3D file

    include_event_context
        Optional. True to include the event context, for C3D files that use
        this field. If False, the events in the output TimeSeries are named
        after the events names in the C3D files, e.g.: "Start", "Heel Strike",
        "Toe Off". If True, the events in the output TimeSeries are named using
        this scheme "context:name", e.g.,: "General:Start",
        "Right:Heel strike", "Left:Toe Off". Default is False.

    convert_point_unit
        Optional. True to convert the point units to meters, if they are
        expressed in other units such as mm in the C3D file. False to keep
        points as is. When unset, a warning is issued if points are stored in
        a different unit than meters. See caution note below.

    convert_forceplate_moment_unit
        Optional. True to convert forceplate moment unit to Nm. Default is
        True.

    convert_forceplate_position_unit
        Optional. True to convert forceplate position units to meters. Default
        is True.


    Returns
    -------
    dict[str, ktk.TimeSeries]
        A dict of TimeSeries, with keys being, if available: "Points",
        "Analogs", "ForcePlates" and/or "Rotations".

    Caution
    -------
    If, for a given C3D file, points are expressed in another unit than meters
    (e.g., mm), and that this file also contains calculated points such as
    angles, powers, etc., then you need to be cautious with the
    `convert_point_unit` parameter:

    - Setting `convert_point_unit` to False reads the file as is, but you
      then need to manually convert the points to meters. This can be done
      easily using ktk.geometry.scale.
    - Setting `convert_point_unit` to True scales all points to meters,
      but also scales every calculated angle, power, etc. as they are read
      as any other point.

    For these special cases, we recommend to set `convert_point_unit` to
    False, and then scale the points manually.

    See Also
    --------
    ktk.write_c3d, ktk.geometry.scale

    Note
    ----
    As for any instrument, please check that your data loads correctly on
    your first use (e.g., sampling frequency, position unit, location and
    orientation of the force platforms, etc.).

    """
    check_param("filename", filename, str)
    if convert_point_unit is not None:
        check_param("convert_point_unit", convert_point_unit, bool)
    check_param(
        "convert_forceplate_moment_unit", convert_forceplate_moment_unit, bool
    )
    check_param(
        "convert_forceplate_position_unit",
        convert_forceplate_position_unit,
        bool,
    )
    check_param("include_event_context", include_event_context, bool)
    if not filename.endswith(".c3d"):
        raise ValueError("The file name must end with '.c3d'.")

    if "return_ezc3d" in kwargs:
        return_ezc3d = kwargs["return_ezc3d"]
    else:
        return_ezc3d = False

    # Create the output
    output = {}

    # Create the reader
    if isinstance(filename, str) and os.path.exists(filename):
        try:
            reader = ezc3d.c3d(filename, extract_forceplat_data=True)
        except OSError:
            # Maybe there's an invalid character in filename.
            # Try to workaround
            # https://github.com/pyomeca/ezc3d/issues/252
            tempfile = kineticstoolkit.config.temp_folder + "/temp.c3d"
            shutil.copyfile(filename, tempfile)
            reader = ezc3d.c3d(tempfile, extract_forceplat_data=True)
            os.remove(tempfile)

    else:
        raise FileNotFoundError(f"File {filename} was not found.")

    if return_ezc3d:
        output["C3D"] = reader

    # ---------------------------------
    # List the events
    try:
        event_names = reader["parameters"]["EVENT"]["LABELS"]["value"]
        event_times = reader["parameters"]["EVENT"]["TIMES"]["value"].T
        event_times = event_times[:, 0] * 60 + event_times[:, 1]
    except KeyError:
        event_names = []
        event_times = []
    try:
        event_contexts = reader["parameters"]["EVENT"]["CONTEXTS"]["value"]
    except KeyError:
        event_contexts = ["" for _ in event_names]
    # Create a list of events to copy in the output TimeSeries
    temp_ts = TimeSeries()
    for i_event in range(len(event_names)):
        event_time = event_times[i_event]
        if include_event_context:
            event_name = event_contexts[i_event] + ":" + event_names[i_event]
        else:
            event_name = event_names[i_event]
        temp_ts.add_event(
            event_time,
            event_name,
            in_place=True,
        )
    events = temp_ts.events

    # -----------------
    # Points
    # -----------------
    # Get the marker label names and create a timeseries data entry for each
    # Get the labels
    point_rate = reader["parameters"]["POINT"]["RATE"]["value"][0]
    if reader["parameters"]["POINT"]["USED"]["value"][0] > 0:
        point_unit = reader["parameters"]["POINT"]["UNITS"]["value"][0]
    else:
        point_unit = "m"
    point_start = reader["header"]["points"]["first_frame"]
    if point_rate > 0:
        start_time = point_start / point_rate
    else:
        start_time = 0
    n_points = reader["parameters"]["POINT"]["USED"]["value"][0]

    labels = reader["parameters"]["POINT"]["LABELS"]["value"]
    # Check if labels2, labels3, labels4,.... exist.
    # https://www.c3d.org/HTML/default.htm?turl=Documents%2Fpointlabels2.htm
    i_additional_labels = 2
    while True:
        str_labels = f"LABELS{i_additional_labels}"
        try:
            additional_labels = reader["parameters"]["POINT"][str_labels][
                "value"
            ]
        except KeyError:
            break
        labels.extend(additional_labels)
        i_additional_labels += 1

    # Solve the point unit conversion mess (issue #147)
    scales = {"mm": 0.001, "cm": 0.01, "dm": 0.1, "m": 1.0}

    if convert_point_unit is None:
        if point_unit == "m":
            point_factor = 1.0
            point_unit = "m"
        elif point_unit in scales:
            warnings.warn(
                "In the specified file, points are expressed in "
                f"{point_unit}. They have been automatically converted to meters "
                f"(scaled by {scales[point_unit]}). Please note that "
                "if this file also contains calculated values such as "
                "angles, powers, etc., they have been also (wrongly) scaled by "
                f"{scales[point_unit]}. Consult "
                "https://kineticstoolkit.uqam.ca/doc/api/ktk.read_c3d.html "
                "for more information. You can mute this warning "
                "by explicitely setting `convert_point_unit` to either True "
                "or False."
            )
            point_factor = scales[point_unit]
            point_unit = "m"
        else:
            warnings.warn(
                "In the specified file, points are expressed in "
                f"`{point_unit}`, which is not recognized by ktk.read_c3d. They "
                "have been left as is, without attempting to convert to meters. "
                "You can mute this warning by setting `convert_point_unit` to "
                "False."
            )
            point_factor = 1.0
            # point_unit = Do not update

    elif convert_point_unit is True:
        try:
            point_factor = scales[point_unit]
            point_unit = "m"
        except KeyError:
            raise ValueError(
                "In the specified file, points are expressed in "
                f"`{point_unit}`, which is not recognized by ktk.read_c3d. "
                "Please set `convert_point_unit` to None of False."
            )

    else:
        point_factor = 1
        # point_unit = Do not update

    if n_points > 0:  # There are points
        points = TimeSeries()

        for i_label in range(n_points):
            # Make sure it's UTF8, and strip leading and ending spaces
            label = labels[i_label]
            key = label.encode("utf-8", "ignore").decode("utf-8").strip()

            # Ensure key is unique, in case of multiple series labelled
            # with the same name
            if (key == "") or (key in points.data):
                suffix_integer = 1
                while f"{key}_{suffix_integer}" in points.data:
                    suffix_integer += 1
                key = f"{key}_{suffix_integer}"

            points.data[key] = np.array(
                [point_factor, point_factor, point_factor, 1]
                * reader["data"]["points"][:, i_label, :].T
            )
            points.add_info(key, "Unit", point_unit, in_place=True)

        if n_points > 0:
            points.time = (
                np.arange(points.data[key].shape[0]) / point_rate + start_time
            )

        # Add events
        points.events = events.copy()

        output["Points"] = points

    # -----------------
    # Analogs
    # -----------------
    labels = reader["parameters"]["ANALOG"]["LABELS"]["value"]
    analog_rate = reader["parameters"]["ANALOG"]["RATE"]["value"][0]
    n_analogs = reader["parameters"]["ANALOG"]["USED"]["value"][0]

    try:
        units = reader["parameters"]["ANALOG"]["UNITS"]["value"]
        if len(units) == 0:
            raise KeyError("No unit")
    except KeyError:
        # No units in the file, create an empty unit for each label
        units = ["" for _ in labels]

    # Check if labels2, labels3, labels4,.... exist.
    # https://www.c3d.org/HTML/default.htm?turl=Documents%2Fpointlabels2.htm
    # In contrast to the points case, units is an array of strings.
    i_additional_labels = 2
    while True:
        str_labels = f"LABELS{i_additional_labels}"
        str_units = f"UNITS{i_additional_labels}"
        try:
            additional_labels = reader["parameters"]["ANALOG"][str_labels][
                "value"
            ]
            try:
                additional_units = reader["parameters"]["ANALOG"][str_units][
                    "value"
                ]
            except KeyError:
                # If there are no additional units, just fill with blank spaces
                additional_units = ["" for _ in additional_labels]

        except KeyError:
            break
        labels.extend(additional_labels)
        units.extend(additional_units)
        i_additional_labels += 1

    if len(labels) > 0:  # There are analogs
        analogs = TimeSeries()

        for i_label in range(n_analogs):
            # Strip leading and ending spaces
            label = labels[i_label]
            key = label.encode("utf-8", "ignore").decode("utf-8").strip()

            # Ensure key is unique, in case of multiple series labelled
            # with the same name
            if (key == "") or (key in analogs.data):
                suffix_integer = 1
                while f"{key}_{suffix_integer}" in analogs.data:
                    suffix_integer += 1
                key = f"{key}_{suffix_integer}"

            analogs.data[key] = reader["data"]["analogs"][0, i_label].T
            if units[i_label] != "":
                analogs.add_info(
                    key,
                    "Unit",
                    units[i_label].encode("utf-8", "ignore").decode("utf-8"),
                    in_place=True,
                )
        if n_analogs > 0:
            analogs.time = (
                np.arange(analogs.data[key].shape[0]) / analog_rate
                + start_time
            )

        # Add events
        analogs.events = events.copy()

        output["Analogs"] = analogs

    # -----------------
    # Rotations
    # -----------------
    # Some files do not have a ROTATION parameter, do nothing for those.
    if "ROTATION" in reader["parameters"]:
        rotations = TimeSeries()

        # Get the marker label names and create a timeseries data entry for each
        # Get the labels
        rotation_rate = reader["parameters"]["ROTATION"]["RATE"]["value"][0]
        rotation_start = reader["header"]["rotations"]["first_frame"]
        start_time = rotation_start / rotation_rate
        n_rotations = reader["parameters"]["ROTATION"]["USED"]["value"][0]

        labels = reader["parameters"]["ROTATION"]["LABELS"]["value"]

        if len(labels) > 0:  # There are rotations
            # no additional labels and scale conversion for rotation matrices
            # move to adding data to the TimeSeries
            for rotation_id in range(n_rotations):
                # Make sure it's UTF8, and strip leading and ending spaces
                label = labels[rotation_id]
                key = label.encode("utf-8", "ignore").decode("utf-8").strip()

                # Ensure key is unique, in case of multiple series labelled
                # with the same name
                if (key == "") or (key in rotations.data):
                    suffix_integer = 1
                    while f"{key}_{suffix_integer}" in rotations.data:
                        suffix_integer += 1
                    key = f"{key}_{suffix_integer}"

                rotations.data[key] = np.array(
                    np.transpose(
                        reader["data"]["rotations"][:, :, rotation_id, :],
                        (2, 0, 1),
                    )
                )

            if n_rotations > 0:
                rotations.time = (
                    np.arange(rotations.data[key].shape[0]) / rotation_rate
                    + start_time
                )

            # Matrices with nans should be complete nans. Some c3d may contain
            # nans in the data but [0, 0, 0, 1] on the 4th line.
            for data in rotations.data:
                rotations.data[data][rotations.isnan(data), :, :] = np.nan

            # Add events
            rotations.events = events.copy()

            # Add to output
            output["Rotations"] = rotations

    # -----------------
    # Platforms
    # -----------------
    if reader["data"]["platform"] != []:

        platforms = TimeSeries(time=analogs.time)  # type: ignore

        n_platforms = len(reader["data"]["platform"])
        for i_platform in range(n_platforms):

            # Define unit conversion factors
            forceplate_position_unit = reader["data"]["platform"][i_platform][
                "unit_position"
            ]
            if convert_forceplate_position_unit and (
                forceplate_position_unit == "mm"
            ):
                forceplate_position_factor = 0.001
                forceplate_position_unit = "m"
            elif convert_forceplate_position_unit and (
                forceplate_position_unit == "cm"
            ):
                forceplate_position_factor = 0.01
                forceplate_position_unit = "m"
            elif convert_forceplate_position_unit and (
                forceplate_position_unit == "dm"
            ):
                forceplate_position_factor = 0.1
                forceplate_position_unit = "m"
            else:
                forceplate_position_factor = 1

            forceplate_moment_unit = reader["data"]["platform"][i_platform][
                "unit_moment"
            ]
            if convert_forceplate_moment_unit and (
                forceplate_moment_unit == "Nmm"
            ):
                moment_factor = 0.001
                forceplate_moment_unit = "Nm"
            elif forceplate_moment_unit == "Nm":
                moment_factor = 1
            else:
                moment_factor = 1
                warnings.warn(
                    f"Moment unit is {forceplate_moment_unit} instead of Nm or Nmm."
                )

            # Add corners
            for i_corner in range(4):
                key = f"FP{i_platform}_Corner{i_corner+1}"
                platforms.data[key] = np.ones((len(platforms.time), 4))
                platforms.data[key][:, 0:3] = (
                    forceplate_position_factor
                    * reader["data"]["platform"][i_platform]["corners"][
                        0:3, i_corner
                    ]
                )
                platforms.add_info(
                    key, "Unit", forceplate_position_unit, in_place=True
                )

            # Add origin and the whole local coordinate system
            lcs = kinetics.create_forceplatform_lcs(
                platforms.data[f"FP{i_platform}_Corner1"],
                platforms.data[f"FP{i_platform}_Corner2"],
                platforms.data[f"FP{i_platform}_Corner3"],
                platforms.data[f"FP{i_platform}_Corner4"],
            )

            platforms.data[f"FP{i_platform}_LCS"] = lcs

            # Add ground reaction force
            force_unit = reader["data"]["platform"][i_platform]["unit_force"]
            if force_unit != "N":
                warnings.warn(
                    f"Force unit is {force_unit} instead of newtons."
                )

            key = f"FP{i_platform}_Force"
            force = np.zeros((len(platforms.time), 4))
            force[:, 0:3] = reader["data"]["platform"][i_platform]["force"].T
            platforms.data[key] = force
            platforms.add_info(key, "Unit", force_unit, in_place=True)

            # Add moment around origin
            key = f"FP{i_platform}_MomentAtCenter"
            moment = np.zeros((len(platforms.time), 4))
            moment[:, 0:3] = (
                moment_factor
                * reader["data"]["platform"][i_platform]["moment"].T
            )
            platforms.data[key] = moment
            platforms.add_info(
                key, "Unit", forceplate_moment_unit, in_place=True
            )

            # Add COP
            key = f"FP{i_platform}_COP"

            # # Calculated by KTK
            # local_force = geometry.get_local_coordinates(force, lcs)
            # local_moment = geometry.get_local_coordinates(moment, lcs)
            # local_cop = kinetics.calculate_cop(local_force, local_moment)
            # platforms.data[key] = geometry.get_global_coordinates(
            #     local_cop, lcs
            # )

            # Already calculated by ezc3d
            platforms.data[key] = np.ones((len(platforms.time), 4))
            platforms.data[key][:, 0:3] = (
                forceplate_position_factor
                * reader["data"]["platform"][i_platform][
                    "center_of_pressure"
                ].T
            )
            platforms.add_info(
                key, "Unit", forceplate_position_unit, in_place=True
            )

            # Add moments at COP
            key = f"FP{i_platform}_MomentAtCOP"
            platforms.data[key] = np.zeros((len(platforms.time), 4))
            platforms.data[key][:, 0:3] = (
                moment_factor * reader["data"]["platform"][i_platform]["Tz"].T
            )
            platforms.add_info(
                key, "Unit", forceplate_moment_unit, in_place=True
            )

        # Add events
        platforms.events = events.copy()

        output["ForcePlatforms"] = platforms

    # ---------------------------------
    # List the metadata (info)
    output["C3DParameters"] = _ezc3d_to_dict(reader)

    return output


def write_c3d(
    filename: str,
    points: TimeSeries | None = None,
    analogs: TimeSeries | None = None,
    rotations: TimeSeries | None = None,
) -> None:
    """
    Write points, analog, and rotations data to a C3D file.

    Parameters
    ----------
    filename
        Path of the C3D file

    points
        Optional. Points trajectories, where data key corresponds to a point,
        expressed as an Nx4 point series::

            [
                [x0, y0, z0, 1.0],
                [x1, y1, z1, 1.0],
                [x2, y2, z2, 1.0],
                ...,
            ]

        Events from this TimeSeries are also added to the c3d.

    analogs
        Optional. Analog signals, where each data key is one series. Series
        that are not unidimensional are converted to multiple unidimensional
        series. For instance, if the shape of analogs.data["Forces"] is
        1000x3, then three unidimensional series of length 1000 are created in
        the C3D: Forces[0], Forces[1] and Forces[2].

        If both `analogs` and `points` are specified, the sample rate of
        `analogs` must be an integer multiple of the `points`'s sample rate.
        Also, `analogs.time[0]` must be the same as `points.time[0]`.

    rotations
        Optional. Rotation matrices, where each data key corresponds to a
        rotation matrix, expressed as a Nx4x4 array.

        If both `rotations` and `points` are specified, the sample rate of
        `rotations` must be an integer multiple of the `points`'s sample rate.
        Also, `rotations.time[0]` must be the same as `rotations.time[0]`.

    See Also
    --------
    ktk.read_c3d

    Notes
    -----
    This function relies on `ezc3d`, which is installed by default using
    conda, but not using pip. Please install ezc3d before using
    write_c3d. https://github.com/pyomeca/ezc3d

    Example
    -------
    Create a simple c3d file with two markers sampled at 240 Hz and two
    sinusoidal analog signals sampled at 1200 Hz, during 10 seconds::

        import kineticstoolkit.lab as ktk
        import numpy as np

        points = ktk.TimeSeries()
        points.time = np.linspace(0, 10, 10*240, endpoint=False)
        points.data["Marker1"] = np.ones((2400, 4))
        points.data["Marker2"] = np.ones((2400, 4))

        analogs = ktk.TimeSeries()
        analogs.time = np.linspace(0, 10, 10*2400, endpoint=False)
        analogs.data["Signal1"] = np.sin(analogs.time)
        analogs.data["Signal2"] = np.cos(analogs.time)

        ktk.write_c3d("testfile.c3d", points=points, analogs=analogs)

    """
    check_param("filename", filename, str)
    check_param("points", points, (TimeSeries, type(None)))
    check_param("analogs", analogs, (TimeSeries, type(None)))
    check_param("rotations", rotations, (TimeSeries, type(None)))

    if not filename.endswith(".c3d"):
        raise ValueError("The file name must end with '.c3d'.")

    try:
        import ezc3d
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "The optional module ezc3d is not installed, but it is required "
            "to use this function. Please install it using: "
            "conda install -c conda-forge ezc3d"
        )

    if points is None:
        # Dummy point data must be created since analogs and rotations rate ratio
        # are based on point rate.
        points = TimeSeries()
        if rotations is not None:
            points = rotations.copy(copy_data=False, copy_info=False)
        elif analogs is not None:
            points = analogs.copy(copy_data=False, copy_info=False)
        else:
            raise ValueError(
                "At least one of points, analogs, or rotations must be "
                "provided. Writing empty C3D files is not supported."
            )

    # Create an empty c3d structure
    c3d = ezc3d.c3d()

    # Add the points, but first make some checks
    points._check_constant_sample_rate()
    point_rate = points.get_sample_rate()

    point_unit = None
    for key in points.data:
        # Check that this is a series of points
        if points.data[key].shape[1] != 4:
            raise ValueError(f"Point {key} is not a Nx4 series of points.")
        if not np.all(
            np.isclose(points.data[key][:, 3], 1.0),
            where=~points.isnan(key),
        ):
            raise ValueError(f"Point {key} is not a series of [x, y, z, 1.0]")

        # Check that units are all the same (or None)
        if key not in points.info:
            continue

        if "Unit" not in points.info[key]:
            continue

        this_unit = points.info[key]["Unit"]

        if this_unit is None:
            continue

        if point_unit is None:
            point_unit = this_unit
            continue

        if point_unit != this_unit:
            raise ValueError(
                "Found different point units in the TimeSeries: "
                f"{point_unit} and {this_unit}."
            )

    if point_unit is None:
        point_unit = "m"  # Default

    # Now format and add the points
    point_list = []
    point_data = np.zeros((4, len(points.data), len(points.time)))

    for i_point, point in enumerate(points.data):
        point_list.append(point)
        point_data[0, i_point, :] = points.data[point][:, 0]
        point_data[1, i_point, :] = points.data[point][:, 1]
        point_data[2, i_point, :] = points.data[point][:, 2]
        point_data[3, i_point, :] = points.data[point][:, 3]

    # Fill point data
    c3d["header"]["points"]["first_frame"] = round(points.time[0] * point_rate)
    c3d.add_parameter("POINT", "RATE", [point_rate])
    if len(point_list) > 0:
        c3d.add_parameter("POINT", "LABELS", [tuple(point_list)])
    c3d.add_parameter("POINT", "UNITS", point_unit)

    c3d["data"]["points"] = point_data

    # Fill analog data
    if analogs is not None:
        analogs._check_constant_sample_rate()
        analog_rate = analogs.get_sample_rate()

        rate_ratio = analog_rate / point_rate
        if ~np.isclose(rate_ratio, round(rate_ratio)):
            raise ValueError(
                "The sample rate of analogs must be an integer "
                "multiple of the points sample rate."
            )

        if ~np.isclose(analogs.time[0], points.time[0]):
            raise ValueError(
                "Points and analogs must share the same starting time. "
                f"However, points.time[0] = {points.time[0]} whereas "
                f"analogs.time[0] = {analogs.time[0]}."
            )

        # Since analogs are unidimensional, we will use the DataFrame exporter
        # to get one column per analog value. This way, forces would become
        # forces[0], forces[1], forces[2] and forces[3].
        df_analogs, analogs_info = analogs._to_dataframe_and_info()

        c3d.add_parameter("ANALOG", "LABELS", list(df_analogs.columns))
        c3d.add_parameter("ANALOG", "RATE", [analog_rate])
        c3d.add_parameter(
            "ANALOG",
            "UNITS",
            [_["Unit"] if "Unit" in _ else "" for _ in analogs_info],
        )
        c3d.add_parameter("ANALOG", "USED", [len(analogs_info)])
        c3d["header"]["analogs"]["first_frame"] = round(
            analogs.time[0] * analog_rate
        )
        c3d["data"]["analogs"] = (df_analogs.to_numpy().T)[np.newaxis]

    # fill rotation data
    if rotations is not None:
        rotations._check_constant_sample_rate()
        rotation_rate = rotations.get_sample_rate()

        rate_ratio = rotation_rate / point_rate
        if ~np.isclose(rate_ratio, round(rate_ratio)):
            raise ValueError(
                "The sample rate of rotations must be an integer "
                "multiple of the points sample rate."
            )

        if ~np.isclose(rotations.time[0], points.time[0]):
            raise ValueError(
                "Points and rotations must share the same starting time. "
                f"However, points.time[0] = {points.time[0]} whereas "
                f"rotations.time[0] = {rotations.time[0]}."
            )

        # Final data should be a 4x4xlen(rotations.data.keys())xlen(rotations.time)
        # np.ndarray
        c3d_rotations = np.zeros(
            (4, 4, len(rotations.data.keys()), len(rotations.time))
        )
        for i_rot, key in enumerate(rotations.data):
            # change from shape (len(rotations.time), 4, 4) to (4, 4, len(rotations.time))
            c3d_rotations[:, :, i_rot, :] = np.transpose(
                rotations.data[key], (1, 2, 0)
            )

        c3d.add_parameter("ROTATION", "LABELS", [*rotations.data.keys()])
        c3d.add_parameter("ROTATION", "RATE", [rotation_rate])
        c3d.add_parameter("ROTATION", "USED", [len(rotations.data.keys())])
        c3d["header"]["rotations"]["first_frame"] = round(
            rotations.time[0] * rotation_rate
        )
        c3d["data"]["rotations"] = c3d_rotations

    # Write the data
    c3d.write(filename)

    # ---------------------------------
    # Add the events
    c3d = ezc3d.c3d(filename)

    points = points.copy()
    if analogs is not None:
        points.events.extend(analogs.events)
    if rotations is not None:
        points.events.extend(rotations.events)
    points.remove_duplicate_events(in_place=True)

    for event in points.events:
        c3d.add_event(
            time=[event.time // 60, np.mod(event.time, 60)], label=event.name
        )

    # Write the data again
    # (workaround https://github.com/pyomeca/ezc3d/issues/263)
    c3d.write(filename)


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
