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
Provide some tools to analyze EMG.

Warning
-------
This module is in very early development (almost empty) and will change in the
future.

"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


from kineticstoolkit import TimeSeries
from kineticstoolkit.decorators import unstable, directory
import pandas as pd
from typing import Dict, List


@unstable
def read_delsys_csv(filename: str) -> Dict[str, Dict[str, TimeSeries]]:
    """
    Read a CSV file exported from Delsys Trigno converter.

    Parameters
    ----------
    filename
        Name of the CSV file to read.

    Returns
    -------
    Dict[str, Dict[str, TimeSeries]]
        A dict with the following keys: 'emg', 'acc', gyro', 'mag', each
        one containing another dict whose keys correspond to a different
        sensor.

    """
    # Check the number of rows to skip
    n_rows = 0
    with open(filename, 'r') as fid:
        while True:
            s = fid.readline()
            if s.startswith('X[s]'):
                break
            else:
                n_rows += 1

    # Open the CSV
    df = pd.read_csv(filename, skiprows=n_rows)

    # Create a TimeSeries for each signal since they all have different time
    # vectors
    n_signals = int(len(df.columns) / 2)

    emg = {}
    acc = {}
    gyro = {}
    mag = {}

    for i_signal in range(n_signals):
        time = df.iloc[:, i_signal * 2].to_numpy()
        name = df.columns[i_signal * 2 + 1]
        data = df.iloc[:, i_signal * 2 + 1].to_numpy()

        if ': Acc' in name:
            short_name = name
            ts = TimeSeries(time=time, data={short_name: data})
            acc[short_name] = ts
        elif ': Mag' in name:
            short_name = name
            ts = TimeSeries(time=time, data={short_name: data})
            mag[short_name] = ts
        elif ': Gyro' in name:
            short_name = name
            ts = TimeSeries(time=time, data={short_name: data})
            gyro[short_name] = ts
        elif ': EMG' in name:
            short_name = name.split(':')[0]
            ts = TimeSeries(time=time, data={short_name: data})
            emg[short_name] = ts

    return {'emg': emg, 'acc': acc, 'gyro': gyro, 'mag': mag}


module_locals = locals()


def __dir__():
    return directory(module_locals)
