#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 11:16:40 2019

@author: felix
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ktk import TimeSeries, TimeSeriesEvent
from ktk import filters


def readfile(filename):

    dataframe = pd.read_csv(filename, header=None)

    data = dataframe.to_numpy()

    index = data[:, 1]
    time = np.arange(0, len(index)) / 240
    channels = data[:, 6:12]
    forces = data[:, 18:21]
    moments = data[:, 21:24]
    angle_deg = data[:, 3]
    angle_rad = np.unwrap(np.deg2rad(angle_deg))

    ts = TimeSeries(time=time)

    ts.data['index'] = index
    ts.data['channels'] = channels
    ts.data['forces'] = np.block([[forces, np.zeros((len(index), 1))]])
    ts.data['moments'] = np.block([[moments, np.zeros((len(index), 1))]])
    ts.data['angle'] = angle_rad

    ts.add_data_info('channels', 'unit', 'raw')
    ts.add_data_info('forces', 'unit', 'N')
    ts.add_data_info('moments', 'unit', 'Nm')
    ts.add_data_info('angle', 'unit', 'rad')

    return ts


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


def detect_pushes(tsin, push_trigger=5, recovery_trigger=2,
                  minimum_push_time=0.1, minimum_recovery_time=0.2):
    """
    Detect pushes and recoveries automatically.


    Parameters
    ----------
    tsin : TimeSeries
        Input TimeSeries that must contain a 'forces' key in its data dict.
    push_trigger : float, optional
        The total force over which a push phase is triggered, in newton.
        The default is 5.
    recovery_trigger : float, optional
        The total force under which a recovery phase is triggered, in newton.
    minimum_push_time : float, optional
        The minimum time required for a push time, in seconds. Detected pushes
        that last less than this minimum time are removed from the push
        analysis. The default is 0.1.
    minimum_recovery_time : float, optional
        The minimum time required for a recovery time, in seconds. Detected
        recoveies that last less than this minimum time are removed from the
        push analysis. The default is 0.2.

    Returns
    -------
    tsout : TimeSeries
        The output timeseries, which is identical to tsin but with the
        following added events:
            - 'push_start'
            - 'push_end'
            - 'cycle_end'

    """
    # Calculate the total force
    f_tot = np.sqrt(np.sum(tsin.data['forces']**2, axis=1))
    ts_force = TimeSeries(time=tsin.time, data={'Ftot': f_tot})

    # Smooth the total force to avoid detecting pushes on glitches
    ts_force = filters.smooth(ts_force, 3)

    # Remove the median if it existed
    ts_force.data['Ftot'] = \
            ts_force.data['Ftot'] - np.median(ts_force.data['Ftot'])

    # Find the pushes
    time = ts_force.time
    data = ts_force.data['Ftot']

    push_state = True   # We start on Push state to wait for a first release, which
                        # allows to ensure the first push will be complete.
    is_first_push = True

    events = []

    for i in range(0, len(data)):

        if ((push_state is False) and (data[i] > push_trigger) and
            (is_first_push is True or
             time[i] - events[-1].time >= minimum_recovery_time)):

            push_state = True

            if is_first_push is False:
                # It's not only the first push, it's also the end of a cycle.
                events.append(TimeSeriesEvent(time[i]-1E-10, 'cycle_end'))

            events.append(TimeSeriesEvent(time[i], 'push_start'))

            is_first_push = False

        elif ((push_state is True) and (data[i] < recovery_trigger)):

            push_state = False

            # Is the push long enough to be considered as a push?
            if (len(events) == 0 or  # Not grab yet
                    time[i] - events[-1].time >= minimum_push_time):
                # Yes.
                events.append(TimeSeriesEvent(time[i], 'push_end'))
            else:
                # No. Remove the last push start.
                events = events[:-2]

    # The first event in list was only to initiate the list. We must remove it.
    # The second event in list is a release. We must remove it.
    events = events[1:]

    # If we stopped during a push, remove the last push_start to ensure that
    # we only have complete pushes.
    if push_state is True:
        events = events[:-2]

    # Form the output timeseries
    tsout = tsin.copy()
    tsout.events = events

    ts_force.events = events


    ts_force.plot()

