#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 11:16:40 2019

@author: felix
"""

import numpy as np
from .timeseries import TimeSeries

def readfile(filename):
    filedata = np.genfromtxt(filename, delimiter = ',')

    channels = filedata[:,1:7]
    time = filedata[:,0]

    ts = TimeSeries(time=time, data={'channels': channels},
                    time_unit='s', data_unit={'channels': 'raw'})

    return(ts)

def calculateforcesandmoments(kinetics):
    gains = np.array(
            [[-0.0314, -0.0300, 0.0576, 0.0037, 0.0019, -0.0019]])
    offsets = np.array(
            [[-111.3874, -63.3298, -8.6596, 1.8089, 1.5761, -0.8869]])

#    # Calculate the forces and moments
    forces_moments = gains * kinetics.data['channels'] + offsets

    kinetics.data['forces'] = forces_moments[:,0:3]
    kinetics.data_unit['forces'] = 'N'
    kinetics.data['moments'] = forces_moments[:,3:6]
    kinetics.data_unit['moments'] = 'Nm'

    return(kinetics)
