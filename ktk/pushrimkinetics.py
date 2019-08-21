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


def read_file(filename):

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

    ts.data['Index'] = index
    ts.data['Channels'] = channels
    ts.data['Forces'] = np.block([[forces, np.zeros((len(index), 1))]])
    ts.data['Moments'] = np.block([[moments, np.zeros((len(index), 1))]])
    ts.data['Angle'] = angle_rad

    ts.add_data_info('Channels', 'Unit', 'raw')
    ts.add_data_info('Forces', 'Unit', 'N')
    ts.add_data_info('Moments', 'Unit', 'Nm')
    ts.add_data_info('Angle', 'Unit', 'rad')

    return ts


def find_recovery_indices(Mz):
    """
    Find recovery indices based on a vector of propulsion moments.

    This function analyzes the Mz moments to find which data correspond to
    pushes and which data correspond to recoveries. The method is very
    conservative on what could be considered as a recovery, so that every
    index returned by this function is almost certain to correspond to a
    recovery. This function is used by pushrimkinetics.remove_sinusoids
    to identify the instants with no hand contact. It should not be used to
    isolate the push and recovery phases (use pushrimkinetics.detectpushes
    instead).

    Parameters
    ----------
    Mz : array
        Array that contains the propulsion moments in Nm.

    Returns
    -------
    index : array
        Array of bools where each True represents recovery.

    """
    Mz = Mz.copy()

    threshold = 2.24  # (Nm): max tolerance for the remaining values.

    while np.nanmax(Mz) - np.nanmin(Mz) > threshold:

        # Remove 1% of data that are the farthest to the median:

        # Sort data
        index_to_remove = np.argsort(np.abs(Mz - np.nanmedian(Mz)))
        sorted_Mz = Mz[index_to_remove]
        index_to_remove = index_to_remove[~np.isnan(sorted_Mz)]

        # Remove the 1% upper.
        index_to_remove = index_to_remove[
                int(0.99*len(index_to_remove))-1:]

        # Assign nan to these data
        Mz[index_to_remove] = np.nan

    index = ~np.isnan(Mz)

    return index


def remove_sinusoids(kinetics, baseline_kinetics=None):
    """
    Remove sinusoids in forces and moments.

    Reference: F. Chénier, R. Aissaoui, C. Gauthier, and D. H. Gagnon,
    "Wheelchair pushrim kinetics measurement: A method to cancel
    inaccuracies due to pushrim weight and wheel camber," Medical
    Engineering and Physics, vol. 40, pp. 75--86, 2017.

    Parameters
    ----------
    kinetics : TimeSeries
        TimeSeries that contains at least Forces, Moments and Angle data.
    baseline_kinetics : TimeSeries, optional
        TimeSeries that contains at least Forces and Moments data. This
        TimeSeries contains a baseline trial, where the wheelchair must be
        pushed by an operator and where no external force must be applied on
        the pushrims. If no baseline is provided, the baseline is calculated
        based on a detection of recoveries in the supplied kinetics
        TimeSeries.

    Returns
    -------
    kinetics : TimeSeries
        A copy of the input TimeSeries, where sinusoids are removed from
        Forces and Moments data.

    """
    kinetics = kinetics.copy()

    if baseline_kinetics is None:
        # Create baseline kinetics.
        recovery_index = find_recovery_indices(kinetics.data['Moments'][:, 2])
        f_ofs = np.hstack((kinetics.data['Forces'][recovery_index, 0:3],
                           kinetics.data['Moments'][recovery_index, 0:3]))
        theta_baseline = kinetics.data['Angle'][recovery_index]

    else:
        # Use baseline kinetics.
        f_ofs = np.hstack((baseline_kinetics.data['Forces'][:, 0:3],
                           baseline_kinetics.data['Moments'][:, 0:3]))
        theta_baseline = baseline_kinetics.data['Angle'][:]

    # Do the regression
    theta_baseline = theta_baseline[:, np.newaxis]
    q = np.hstack((
            np.sin(theta_baseline),
            np.cos(theta_baseline),
            np.ones((len(theta_baseline), 1))
            ))
    A = np.linalg.lstsq(q, f_ofs, rcond=None)
    A = A[0]

    # Apply the regression to forces and moments
    theta = kinetics.data['Angle']
    theta = theta[:, np.newaxis]

    f = np.hstack((kinetics.data['Forces'][:, 0:3],
                   kinetics.data['Moments'][:, 0:3]))

    q = np.hstack((
            np.sin(theta),
            np.cos(theta),
            np.ones((len(theta), 1))
            ))

    f = f - q @ A

    # Make the output timeseries
    kinetics.data['Forces'][:, 0:3] = f[:, 0:3]
    kinetics.data['Moments'][:, 0:3] = f[:, 3:6]

    return kinetics


def calculateforcesandmoments(kinetics):
    gains = np.array(
            [[-0.0314, -0.0300, 0.0576, 0.0037, 0.0019, -0.0019]])
    offsets = np.array(
            [[-111.3874, -63.3298, -8.6596, 1.8089, 1.5761, -0.8869]])

#    # Calculate the forces and moments
    forces_moments = gains * kinetics.data['channels'] + offsets

    kinetics.data['Forces'] = forces_moments[:,0:3]
    kinetics.add_data_info('Forces', 'Unit', 'N')
    kinetics.data['Moments'] = forces_moments[:,3:6]
    kinetics.add_data_info('Moments', 'Unit', 'Nm')

    return(kinetics)


def detect_pushes(tsin, push_trigger=5, recovery_trigger=2,
                  minimum_push_time=0.1, minimum_recovery_time=0.2):
    """
    Detect pushes and recoveries automatically.


    Parameters
    ----------
    tsin : TimeSeries
        Input TimeSeries that must contain a 'Forces' key in its data dict.
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
            - 'pushstart'
            - 'pushend'
            - 'cycleend'

    """
    # Calculate the total force
    f_tot = np.sqrt(np.sum(tsin.data['Forces']**2, axis=1))
    ts_force = TimeSeries(time=tsin.time, data={'Ftot': f_tot})

    # Smooth the total force to avoid detecting pushes on glitches
    ts_force = filters.smooth(ts_force, 11)

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
                events.append(TimeSeriesEvent(time[i]-1E-6, 'cycleend'))

            events.append(TimeSeriesEvent(time[i], 'pushstart'))

            is_first_push = False

        elif ((push_state is True) and (data[i] < recovery_trigger)):

            push_state = False

            # Is the push long enough to be considered as a push?
            if (len(events) == 0 or  # Not grab yet
                    time[i] - events[-1].time >= minimum_push_time):
                # Yes.
                events.append(TimeSeriesEvent(time[i], 'pushend'))
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

    return tsout

    ts_force.events = events


    ts_force.plot()

