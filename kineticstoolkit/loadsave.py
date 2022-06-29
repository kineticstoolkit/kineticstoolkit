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
from kineticstoolkit.decorators import directory
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

from typing import Any


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


module_locals = locals()


def __dir__():  # pragma: no cover
    return directory(module_locals)


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
