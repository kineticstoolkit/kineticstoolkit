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
from __future__ import annotations


__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"

from kineticstoolkit.timeseries import TimeSeries
from kineticstoolkit.exceptions import check_types
import kineticstoolkit.config

import os
import numpy as np
import pandas as pd
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
    - Tuples are also supported but will be loaded back as lists, without
    warning.
    - Complex Pandas Series (e.g., series or different types) may not be
    supported. Only the following attributes of the Series are saved:
    name, index, and the data itself.
    - Complex Pandas DataFrames (e.g., multiindex, columns of different types)
    may not be supported. Only the following attributes of the DataFrame are
    saved: columns, index, and the data itself.

    See also
    --------
    ktk.load

    """
    check_types(save, locals())

    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return {"class__": "numpy.array", "value": obj.tolist()}

            elif (
                str(type(obj))
                == "<class 'kineticstoolkit.timeseries.TimeSeries'>"
            ):
                out = {}
                out["class__"] = "ktk.TimeSeries"
                out["time"] = obj.time.tolist()
                out["time_info"] = obj.time_info
                out["data_info"] = obj.data_info
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
            out.time_info = obj["time_info"]
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

    See also
    --------
    ktk.save

    """
    check_types(load, locals())

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
    convert_point_unit: bool | None = None,
    include_event_context: bool = False,
    **kwargs,
) -> dict[str, TimeSeries]:
    """
    Read point and analog data from a C3D file.

    Point positions are returned in `output["Points"]` as a TimeSeries, where
    each point corresponds to a data key. Each point position is expressed as
    an Nx4 point series::

        [
            [x0, y0, z0, 1.0],
            [x1, y1, z1, 1.0],
            [x2, y2, z2, 1.0],
            ...,
        ]

    If available, analog data is returned in `output["Analogs"]` as a
    TimeSeries, where each analog signal is expressed as a unidimensional
    array of length N.

    Some applications store calculated values such as angles, forces, moments,
    powers, etc. into the C3D file. Storing these data is application-specific
    and is not standardized in the C3D file format (https://www.c3d.org).
    This function reads these values as points regardless of their nature.

    Parameters
    ----------
    filename
        Path of the C3D file

    convert_point_unit
        Optional. True to convert the point units to meters, if they are
        expressed in other units such as mm in the C3D file. False to keep
        points as is. When unset, if points are stored in a unit other than
        meters, then a warning is issued. See caution note below.

    include_event_context
        Optional. True to include the event context, for C3D files that use
        this field. If False, the events in the output TimeSeries are named
        after the events names in the C3D files, e.g.: "Start", "Heel Strike",
        "Toe Off". If True, the events in the output TimeSeries are named using
        this scheme "context:name", e.g.,: "General:Start",
        "Right:Heel strike", "Left:Toe Off". The default is False.

    Returns
    -------
    dict[ktk.TimeSeries]
        A dict of TimeSeries, with keys being "Points" and if available,
        "Analogs".

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

    Warning
    -------
    This function, which has been introduced in version 0.9, is still
    experimental and its behaviour or API may change slightly in the future.

    See also
    --------
    ktk.write_c3d, ktk.geometry.scale

    Notes
    -----
    - This function relies on `ezc3d`, which is installed by default using
      conda, but not using pip. Please install ezc3d before using
      read_c3d. https://github.com/pyomeca/ezc3d

    - As for any instrument, please check that your data loads correctly on
      your first use (e.g., sampling frequency, position unit). It is
      possible that read_c3d misses some corner cases.

    """
    """
    Experimental, unstable features in development:
        
    Parameters
    ----------
    read_force_plates
        True to read force plates.

    convert_moment_unit
        Optional. True to convert the moment units to Nm, even if they are
        expressed in Nmm in the C3D file.

    If the C3D file contains force plate data and read_force_plates is True,
    these data are returned in output["ForcePlates"] as a TimeSeries following
    this structure:

      - 'Forces0': Nx4 series of force vectors on platform 0.
      - 'Forces1': Nx4 series of force vectors on platform 1.
      - 'Forces2': Nx4 series of force vectors on platform 2.
      - ...
      - 'Moments0': Nx4 series of moment vectors on platform 0.
      - 'Moments1': Nx4 series of moment vectors on platform 2.
      - 'Moments2': Nx4 series of moment vectors on platform 3.
      - ...
        
    """
    check_types(read_c3d, locals())

    try:
        import ezc3d
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "The optional module ezc3d is not installed, but it is required "
            "to use this function. Please install it using: "
            "conda install -c conda-forge ezc3d"
        )

    # Additional arguments (in development)
    if "extract_force_plates" in kwargs:
        extract_force_plates = kwargs["extract_force_plates"]
    else:
        extract_force_plates = False

    if "convert_moment_unit" in kwargs:
        convert_moment_unit = kwargs["convert_moment_unit"]
    else:
        convert_moment_unit = True

    # Create the output
    output = {}

    # Create the reader
    if isinstance(filename, str) and os.path.exists(filename):
        try:
            reader = ezc3d.c3d(
                filename, extract_forceplat_data=extract_force_plates
            )
        except OSError:
            # Maybe there's an invalid character in filename.
            # Try to workaround
            # https://github.com/pyomeca/ezc3d/issues/252
            tempfile = kineticstoolkit.config.temp_folder + "/temp.c3d"
            shutil.copyfile(filename, tempfile)
            reader = ezc3d.c3d(
                tempfile, extract_forceplat_data=extract_force_plates
            )
            os.remove(tempfile)

    else:
        raise FileNotFoundError(f"File {filename} was not found.")

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

    # ---------------------------------
    # Create the points TimeSeries
    points = TimeSeries()

    # Get the marker label names and create a timeseries data entry for each
    # Get the labels
    point_rate = reader["parameters"]["POINT"]["RATE"]["value"][0]
    point_unit = reader["parameters"]["POINT"]["UNITS"]["value"][0]
    point_start = reader["header"]["points"]["first_frame"]
    start_time = point_start / point_rate
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
                f"{point_unit}. They were automatically converted to meters "
                f"by scaling them by {scales[point_unit]}. Please note that "
                "if this file also contains calculated values such as "
                "angles, powers, etc., they were also (wrongly) scaled by "
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
                "were left as is, without attempting to convert to meters. "
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

    for i_label in range(n_points):
        # Make sure it's UTF8, and strip leading and ending spaces
        label = labels[i_label]
        key = label.encode("utf-8", "ignore").decode("utf-8").strip()
        if label != "":
            points.data[key] = np.array(
                [point_factor, point_factor, point_factor, 1]
                * reader["data"]["points"][:, i_label, :].T
            )
            points = points.add_data_info(key, "Unit", point_unit)

    if n_points > 0:
        points.time = (
            np.arange(points.data[key].shape[0]) / point_rate + start_time
        )

    # Add events
    for i_event in range(len(event_names)):
        event_time = event_times[i_event]
        if include_event_context:
            event_name = event_contexts[i_event] + ":" + event_names[i_event]
        else:
            event_name = event_names[i_event]
        points.add_event(
            event_time,
            event_name,
            in_place=True,
        )
    points.sort_events(in_place=True)

    # Add to output
    output["Points"] = points

    # Analogs
    labels = reader["parameters"]["ANALOG"]["LABELS"]["value"]
    analog_rate = reader["parameters"]["ANALOG"]["RATE"]["value"][0]
    units = reader["parameters"]["ANALOG"]["UNITS"]["value"]
    n_analogs = reader["parameters"]["ANALOG"]["USED"]["value"][0]

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
            additional_units = reader["parameters"]["ANALOG"][str_units][
                "value"
            ]
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
            analogs.data[key] = reader["data"]["analogs"][0, i_label].T
            if units[i_label] != "":
                analogs.add_data_info(
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
        for i_event, event_name in enumerate(event_names):
            analogs.add_event(event_times[i_event], event_name, in_place=True)
        analogs.sort_events(in_place=True)

        output["Analogs"] = analogs

    # ---------------------------------
    # Create the platforms TimeSeries
    if extract_force_plates and reader["data"]["platform"] != []:
        platforms = TimeSeries(time=analogs.time)  # type: ignore

        n_platforms = len(reader["data"]["platform"])
        for i_platform in range(n_platforms):
            force_unit = reader["data"]["platform"][0]["unit_force"]

            if force_unit != "N":
                warnings.warn(
                    f"Force unit is {force_unit} instead of newtons."
                )

            key = f"Forces{i_platform}"
            platforms.data[key] = np.zeros((len(platforms.time), 4))
            platforms.data[key][:, 0:3] = reader["data"]["platform"][
                i_platform
            ]["force"].T
            platforms.add_data_info(key, "Unit", force_unit, in_place=True)

            moment_unit = reader["data"]["platform"][0]["unit_moment"]

            if convert_moment_unit and (moment_unit == "Nmm"):
                moment_factor = 0.001
                moment_unit = "Nm"
            elif moment_unit == "Nm":
                moment_factor = 1
            else:
                moment_factor = 1
                warnings.warn(f"Moment unit is {moment_unit} instead of Nm.")

            key = f"Moments{i_platform}"
            platforms.data[key] = np.zeros((len(platforms.time), 4))
            platforms.data[key][:, 0:3] = (
                moment_factor
                * reader["data"]["platform"][i_platform]["moment"].T
            )
            platforms.add_data_info(key, "Unit", moment_unit, in_place=True)

        # Add events
        for i_event, event_name in enumerate(event_names):
            platforms.add_event(
                event_times[i_event], event_name, in_place=True
            )
        platforms.sort_events(in_place=True)

        output["ForcePlates"] = platforms

    return output


def write_c3d(
    filename: str, points: TimeSeries, analogs: TimeSeries | None = None
) -> None:
    """
    Write points and analog data to a C3D file.

    Parameters
    ----------
    filename
        Path of the C3D file

    points
        Points trajectories, where data key corresponds to a point, expressed
        as an Nx4 point series::

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
        series. For instance, if the shape of analogs.data['Forces'] is
        1000x3, then three unidimensional series of length 1000 are created in
        the C3D: Forces[0], Forces[1] and Forces[2].

        The sample rate of `analogs` must be an integer multiple of the
        `points`'s sample rate. Also, `analogs.time[0]` must be the same as
        `points.time[0]`.

    See also
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
    try:
        import ezc3d
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "The optional module ezc3d is not installed, but it is required "
            "to use this function. Please install it using: "
            "conda install -c conda-forge ezc3d"
        )

    # Basic type check
    check_types(write_c3d, locals())

    # Create an empty c3d structure
    c3d = ezc3d.c3d()

    # Add the points, but first make some checks
    points._check_not_empty_data()
    points._check_constant_sample_rate()

    point_rate = points.get_sample_rate()

    point_unit = None
    for key in points.data:
        # Check that this is a series of points
        if points.data[key].shape[1] != 4:
            raise ValueError(f"Point {key} is not a Nx4 series of points.")
        if not np.all(
            np.isclose(points.data[key][:, 3], 1.0), where=~points.isnan(key)
        ):
            raise ValueError(f"Point {key} is not a series of [x, y, z, 1.0]")

        # Check that units are all the same (or None)
        if key not in points.data_info:
            continue

        if "Unit" not in points.data_info[key]:
            continue

        this_unit = points.data_info[key]["Unit"]

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
    c3d.add_parameter("POINT", "LABELS", [tuple(point_list)])
    c3d.add_parameter("POINT", "UNITS", point_unit)

    c3d["data"]["points"] = point_data

    # Fill analog data
    if analogs is not None:
        analogs._check_not_empty_data()
        analogs._check_constant_sample_rate()
        analog_rate = analogs.get_sample_rate()

        rate_ratio = analog_rate / point_rate
        if ~np.isclose(rate_ratio, int(rate_ratio)):
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
        df_analogs, analogs_data_info = analogs._to_dataframe_and_info()

        c3d.add_parameter("ANALOG", "LABELS", list(df_analogs.columns))
        c3d.add_parameter("ANALOG", "RATE", [analog_rate])
        c3d.add_parameter(
            "ANALOG",
            "UNITS",
            [_["Unit"] if "Unit" in _ else "" for _ in analogs_data_info],
        )
        c3d.add_parameter("ANALOG", "USED", [len(analogs_data_info)])
        c3d["data"]["analogs"] = (df_analogs.to_numpy().T)[np.newaxis]

    # Write the data
    c3d.write(filename)

    # ---------------------------------
    # Add the events
    c3d = ezc3d.c3d(filename)
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
