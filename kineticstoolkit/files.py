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

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"

from kineticstoolkit.timeseries import TimeSeries
import kineticstoolkit.config

import os
import numpy as np
import pandas as pd
import warnings
import shutil
import json
from datetime import datetime
import time
import getpass
import zipfile

from typing import Any, Dict


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
    - pandas.DataFrame
    - pandas.Series
    - ktk.TimeSeries

    Tuples are also supported but will be loaded back as lists, without
    warning.
    """

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
                    "dtype": str(obj.dtype),
                    "index": obj.index.tolist(),
                    "data": obj.tolist(),
                }

            elif isinstance(obj, pd.DataFrame):
                return {
                    "class__": "pandas.DataFrame",
                    "columns": obj.columns.tolist(),
                    "dtypes": [str(dtype) for dtype in obj.dtypes],
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
                dtype=obj["dtypes"][0],
                columns=obj["columns"],
                index=obj["index"],
            )

        elif to_class == "pandas.Series":
            return pd.Series(
                obj["data"],
                dtype=obj["dtype"],
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

    Load a data file as saved using the ktk.save function.

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
    """
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
    filename: str, *, convert_point_unit: bool = True
) -> Dict[str, TimeSeries]:
    """
    Read point and analog data from a C3D file.

    Point positions are returned in a TimeSeries in output['Points'], where
    each marker corresponds to a data key. Each marker position is expressed
    in this form:

    array([[x0, y0, z0, 1.], [x1, y1, z1, 1.], [x2, y2, z2, 1.], ...])

    If the C3D file contains analog data, it is returned as a TimeSeries in
    output['Analogs'].

    Parameters
    ----------
    filename
        Path of the C3D file
    convert_point_unit
        Optional. True to ensure that the unit for points is in meters, and
        not in millimeters.

    Notes
    -----
    - This function relies on `ezc3d`, which is available on
      conda-forge and on git-hub. Please install ezc3d before using
      read_c3d_file. https://github.com/pyomeca/ezc3d

    - The point unit must be either mm or m. In both cases, the final unit
      returned in the output TimeSeries is m.

    - As for any instrument, please check that your data loads correctly on
      your first use (e.g., sampling frequency, position unit). It is
      possible that read_c3d misses some corner cases.

    """
    try:
        import ezc3d
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "The optional module ezc3d is not installed, but it is required "
            "to use this function. Please install it using: "
            "conda install -c conda-forge ezc3d"
        )

    # Create the reader
    if isinstance(filename, str) and os.path.exists(filename):
        reader = ezc3d.c3d(filename)
    else:
        raise FileNotFoundError(f"File {filename} was not found.")

    # Create the output
    output = {}

    # ---------------------------------
    # List the events
    if "EVENT" in reader["parameters"]:
        event_names = reader["parameters"]["EVENT"]["LABELS"]["value"]
        event_times = reader["parameters"]["EVENT"]["TIMES"]["value"].T
        event_times = event_times[:, 0] * 60 + event_times[:, 1]
    else:
        event_names = []
        event_times = []

    # ---------------------------------
    # Create the points TimeSeries
    points = TimeSeries()

    # Get the marker label names and create a timeseries data entry for each
    # Get the labels
    point_rate = reader["parameters"]["POINT"]["RATE"]["value"][0]
    point_unit = reader["parameters"]["POINT"]["UNITS"]["value"][0]
    point_start = reader["header"]["points"]["first_frame"]
    start_time = point_start / point_rate
    labels = reader["parameters"]["POINT"]["LABELS"]["value"]
    n_points = reader["parameters"]["POINT"]["USED"]["value"][0]

    if convert_point_unit and (point_unit == "mm"):
        point_factor = 0.001
        point_unit = "m"
    elif point_unit == "m":
        point_factor = 1
    else:
        raise (ValueError("Point unit must be 'm' or 'mm'."))

    for i_label in range(n_points):
        # Strip leading and ending spaces
        label = labels[i_label]
        key = label.strip()
        if label != "":
            points.data[key] = np.array(
                [point_factor, point_factor, point_factor, 1]
                * reader["data"]["points"][:, i_label, :].T
            )
            points = points.add_data_info(key, "Unit", point_unit)

    points.time = (
        np.arange(points.data[key].shape[0]) / point_rate + start_time
    )

    for i_event, event_name in enumerate(event_names):
        points.add_event(event_times[i_event], event_name, in_place=True)
    points.sort_events(in_place=True)

    output["Points"] = points

    # ---------------------------------
    # Create the analogs TimeSeries

    labels = reader["parameters"]["ANALOG"]["LABELS"]["value"]
    analog_rate = reader["parameters"]["ANALOG"]["RATE"]["value"][0]
    units = reader["parameters"]["ANALOG"]["UNITS"]["value"]
    n_analogs = reader["parameters"]["ANALOG"]["USED"]["value"][0]

    if len(labels) == 0:  # There are no analogs
        analogs = None
    else:
        analogs = TimeSeries()

        for i_label in range(n_analogs):
            # Strip leading and ending spaces
            label = labels[i_label]
            key = label.strip()
            analogs.data[key] = reader["data"]["analogs"][0, i_label].T
            if units[i_label] != "":
                analogs.add_data_info(
                    key, "Unit", units[i_label], in_place=True
                )

        analogs.time = (
            np.arange(analogs.data[key].shape[0]) / analog_rate + start_time
        )

        for i_event, event_name in enumerate(event_names):
            analogs.add_event(event_times[i_event], event_name, in_place=True)
        analogs.sort_events(in_place=True)

        output["Analogs"] = analogs

    return output


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
