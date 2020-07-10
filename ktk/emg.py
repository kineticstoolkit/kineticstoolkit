#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020 Félix Chénier
#
# This file is not for redistribution.
"""
EMG analysis.
"""

import ktk
import pandas as pd

def read_delsys_csv(filename):
    """Read a CSV file exported from Delsys Trigno converter."""
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
            ts = ktk.TimeSeries(time=time, data={short_name: data})
            acc[short_name] = ts
        elif ': Mag' in name:
            short_name = name
            ts = ktk.TimeSeries(time=time, data={short_name: data})
            mag[short_name] = ts
        elif ': Gyro' in name:
            short_name = name
            ts = ktk.TimeSeries(time=time, data={short_name: data})
            gyro[short_name] = ts
        elif ': EMG' in name:
            short_name = name.split(':')[0]
            ts = ktk.TimeSeries(time=time, data={short_name: data})
            emg[short_name] = ts

    return {'emg': emg, 'acc': acc, 'gyro': gyro, 'mag': mag}
