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
Provide functions to calculate inverse dynamics.

Warning
-------
This module is currently experimental and its API and behaviour could be
modified in the future.

"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import kineticstoolkit.filters
from kineticstoolkit.decorators import unstable, directory

import numpy as np
from typing import Dict, List
from kineticstoolkit import TimeSeries

import warnings

default_filter_fc = 10  # Hz
default_filter_order = 2


@unstable
def get_anthropometrics(segment_name: str,
                        total_mass: float) -> Dict[str, float]:
    """
    Get anthropometric values for a given segment name.

    For the moment, only this table is available:

    D. A. Winter, Biomechanics and Motor Control of Human Movement,
    4th ed. University of Waterloo, Waterloo, Ontario, Canada,
    John Wiley & Sons, 2009.

    Parameters
    ----------
    segment_name
        The name of the segment, either:

        - 'Hand' (wrist axis to knuckle II of middle finger)
        - 'Forearm' (elbow axis to ulnar styloid)
        - 'UpperArm' (glenohumeral axis to elbow axis)
        - 'ForearmHand' (elbow axis to ulnar styloid)
        - 'TotalArm' (glenohumeral joint to ulnar styloid)
        - 'Foot' (lateral malleolus to head metatarsal II)
        - 'Leg' (femoral condyles to medial malleolus)
        - 'Thigh' (greater trochanter to femoral condyles)
        - 'FootLeg' (fomeral condyles to medial malleolus)
        - 'TotalLeg' (greater trochanter to medial malleolus)
        - 'TrunkHeadNeck' (greater trochanter to glenohumeral joint)
        - 'HeadArmsTrunk' (greater trochanter to glenohumeral joint)

    total_mass
        The total mass of the person, in kg.

    Returns
    -------
    Dict[str, float]
        A dict with the following keys:

        - 'Mass' : Mass of the segment, in kg.
        - 'COMProximalRatio' : Distance between the segment's center of
          mass and the proximal joint, as a ratio of the distance between
          both joints.
        - 'COMDistalRatio' : Distance between the segment's center of mass
          and the distal joint, as a ratio of the distance between
          both joints.
        - 'GyrationCOMRatio': Radius of gyration around the segment's center
          of mass, as a ratio of the distance between both joints.
        - 'GyrationProximalRatio': Radius of gyration around the segment's
           proximal joint, as a ratio of the distance between both joints.
        - 'GyrationDistalRatio': Radius of gyration around the segment's
           distal joint, as a ratio of the distance between both joints.

    """
    table = dict()
    table['Hand'] = [0.006, 0.506, 0.494, 0.297, 0.587, 0.577]
    table['Forearm'] = [0.016, 0.430, 0.570, 0.303, 0.526, 0.647]
    table['UpperArm'] = [0.028, 0.436, 0.564, 0.322, 0.542, 0.645]
    table['ForearmHand'] = [0.022, 0.682, 0.318, 0.468, 0.827, 0.565]
    table['TotalArm'] = [0.050, 0.530, 0.470, 0.368, 0.645, 0.596]
    table['Foot'] = [0.0145, 0.50, 0.50, 0.475, 0.690, 0.690]
    table['Leg'] = [0.0465, 0.433, 0.567, 0.302, 0.528, 0.643]
    table['Thigh'] = [0.100, 0.433, 0.567, 0.323, 0.540, 0.653]
    table['FootLeg'] = [0.061, 0.606, 0.394, 0.416, 0.735, 0.572]
    table['TotalLeg'] = [0.161, 0.447, 0.553, 0.326, 0.560, 0.650]
    table['TrunkHeadNeck'] = [0.578, 0.66, 0.34, 0.503, 0.830, 0.607]
    table['HeadArmsTrunk'] = [0.678, 0.626, 0.374, 0.496, 0.798, 0.621]

    out = dict()
    try:
        out['Mass'] = table[segment_name][0] * total_mass
        out['COMProximalRatio'] = table[segment_name][1]
        out['COMDistalRatio'] = table[segment_name][2]
        out['GyrationCOMRatio'] = table[segment_name][3]
        out['GyrationProximalRatio'] = table[segment_name][4]
        out['GyrationDistalRatio'] = table[segment_name][5]
        return out
    except KeyError:
        raise ValueError(f'The segment "{segment_name}" is not available.')


@unstable
def calculate_com_position(
        ts: TimeSeries, inertial_constants: Dict[str, float]) -> TimeSeries:
    """
    Calculate the position of the segment's center of mass.

    Adds the data key 'COMPosition' to the TimeSeries based on the
    'COMProximalRatio' key of the dict inertial_constants.

    Parameters
    ----------
    ts
        A TimeSeries with the following data keys:

        - ProximalJointPosition (Nx4)
        - DistalJointPosition (Nx4)

    inertial_constants
        A dict with at least a 'COMProximalRatio' key, which is the distance
        between the segment's center of mass and the proximal joint, as a
        ratio of the distance between both joints.

    Returns
    -------
    TimeSeries
        A copy of the input timeseries plus the 'COMPosition' data
        key.

    """
    ts = ts.copy()

    ts.data['COMPosition'] = (
        inertial_constants['COMProximalRatio'] *
        (ts.data['DistalJointPosition'] - ts.data['ProximalJointPosition']) +
        ts.data['ProximalJointPosition'])
    return ts


@unstable
def calculate_com_acceleration(
        ts: TimeSeries, filter_func: str, **kwargs) -> TimeSeries:
    """
    Calculate the acceleration of the segment's center of mass.

    Adds the data key 'COMAcceleration' to the TimeSeries based on the
    specified filter function.

    Parameters
    ----------
    ts
        A TimeSeries with the at least a 'COMPosition' data key.

    filter_func
        'savgol': calculate the acceleration using the 2nd order coefficient
        of a moving polynomial.
        'butter': no-lag 2nd order filter followed by a centered derivate.

    window_length
        Only for Savistky-Golay filters (savgol).
        The length of the filter window. window_length must be a positive
        odd integer less or equal than the length of the TimeSeries.

    poly_order
        Only for Savistky-Golay filters (savgol).
        Optional. The order of the polynomial used to fit the samples.
        polyorder must be less than window_length. The default is 2.

    fc
        Only for Butterworth filters (butter).
        Cut-off frequency in Hz.
    order
        Only for Butterworth filters (butter).
        Optional. Order of the filter. The default is 2.

    Returns
    -------
    TimeSeries
        A copy of the input timeseries plus the 'COMAcceleration' data
        key.

    """
    ts = ts.copy()

    if filter_func == 'savgol':
        if 'window_length' not in kwargs:
            raise ValueError(
                "window_length must be specified for Savitzky-Golay filters")
        if 'poly_order' not in kwargs:
            kwargs['poly_order'] = 2

        ts_com = ts.get_subset('COMPosition')
        ts_acc = kineticstoolkit.filters.savgol(
            ts_com, window_length=kwargs['window_length'],
            poly_order=kwargs['poly_order'], deriv=2)

        return ts

    elif filter_func == 'butter':
        if 'fc' not in kwargs:
            raise ValueError(
                "fc must be specified for Butterworth filters")
        if 'order' not in kwargs:
            kwargs['order'] = 2

        ts_com = ts.get_subset('COMPosition')
        ts_acc = kineticstoolkit.filters.butter(
            ts_com, fc=kwargs['fc'],
            order=kwargs['order'])

        ts_acc = kineticstoolkit.filters.deriv(ts_acc, n=2)
        ts_acc = ts_acc.rename_data('COMPosition', 'COMAcceleration')

        ts = ts.merge(ts_acc, resample=True)

        return ts

    else:
        raise ValueError('Unknown filter type')


@unstable
def calculate_segment_angles(ts: TimeSeries) -> TimeSeries:
    """
    Calculate the segment's projection angles in the three axes.

    Parameters
    ----------
    ts
        A TimeSeries with the following data keys:

        - ProximalJointPosition (Nx4)
        - DistalJointPosition (Nx4)

    Returns
    -------
    TimeSeries
        A copy of the input timeseries plus the 'SegmentAngles' data
        key, represented as an Nx3 numpy array.

    """
    ts = ts.copy()

    proximal_to_distal_joint_distance = (
        ts.data['DistalJointPosition'] - ts.data['ProximalJointPosition'])

    segment_angle_x = np.arctan2(
        proximal_to_distal_joint_distance[:, 2],
        proximal_to_distal_joint_distance[:, 1])
    segment_angle_y = np.arctan2(
        proximal_to_distal_joint_distance[:, 2],
        proximal_to_distal_joint_distance[:, 0])
    segment_angle_z = np.arctan2(
        proximal_to_distal_joint_distance[:, 1],
        proximal_to_distal_joint_distance[:, 0])
    ts.data['SegmentAngles'] = np.concatenate((
        segment_angle_x[:, np.newaxis],
        segment_angle_y[:, np.newaxis],
        segment_angle_z[:, np.newaxis]), axis=1)

    return ts


@unstable
def calculate_segment_rotation_rates(
        ts: TimeSeries, filter_func: str, **kwargs) -> TimeSeries:
    """
    Calculate the segments' projected angular velocities and accelerations.

    Parameters
    ----------
    ts
        A TimeSeries with at least the 'SegmentAngles' key (Nx3).

    filter_func
        'savgol': calculate the velocity and acceleration using the 1st and
        2nd order coefficients of a moving polynomial.
        'butter': no-lag 2nd order filter followed by centered derivates.

    window_length
        Only for Savitzky-Golay filters (savgol).
        The length of the filter window. window_length must be a positive
        odd integer less or equal than the length of the TimeSeries.

    poly_order
        Only for Savitzky-Golay filters (savgol).
        Optional. The order of the polynomial used to fit the samples.
        polyorder must be less than window_length. The default is 2.

    fc
        Only for Butterworth filters (butter).
        Cut-off frequency in Hz.

    order
        Only for Butterworth filters (butter).
        Optional. Order of the filter. The default is 2.

    Returns
    -------
    TimeSeries
        A copy of the input timeseries plus the 'AngularVelocity' and
        'AngularAcceleration' data keys (each Nx3).

    """
    ts = ts.copy()

    if filter_func == 'savgol':
        if 'window_length' not in kwargs:
            raise ValueError(
                "window_length must be specified for Savitzky-Golay filters")
        if 'poly_order' not in kwargs:
            kwargs['poly_order'] = 2

        ts_angle = ts.get_subset('SegmentAngles')
        ts_angvel = kineticstoolkit.filters.savgol(
            ts_angle,
            window_length=kwargs['window_length'],
            poly_order=kwargs['poly_order'], deriv=1)
        ts_angacc = kineticstoolkit.filters.savgol(
            ts_angle,
            window_length=kwargs['window_length'],
            poly_order=kwargs['poly_order'], deriv=2)
        ts.data['AngularVelocity'] = ts_angvel.data['SegmentAngles']
        ts.data['AngularAcceleration'] = ts_angacc.data['SegmentAngles']

        return ts

    elif filter_func == 'butter':
        if 'fc' not in kwargs:
            raise ValueError(
                "fc must be specified for Butterworth filters")
        if 'order' not in kwargs:
            kwargs['order'] = 2

        ts_angle = ts.get_subset('SegmentAngles')
        ts_filt = kineticstoolkit.filters.butter(
            ts_angle, fc=kwargs['fc'],
            order=kwargs['order'])

        ts_vel = kineticstoolkit.filters.deriv(ts_filt, n=1)
        ts_vel = ts_vel.rename_data('SegmentAngles', 'AngularVelocity')

        ts_acc = kineticstoolkit.filters.deriv(ts_filt, n=2)
        ts_acc = ts_acc.rename_data('SegmentAngles', 'AngularAcceleration')

        ts = ts.merge(ts_vel, resample=True)
        ts = ts.merge(ts_acc, resample=True)

        return ts

    else:
        raise ValueError('Unknown filter type')


@unstable
def calculate_proximal_wrench(
        ts: TimeSeries, inertial_constants: Dict[str, float]) -> TimeSeries:
    """
    Calculate the proximal wrench based on a TimeSeries.

    This function is based on R. Dumas, R. Aissaoui, and J. A. De Guise,
    "A 3D generic inverse dynamic method using wrench notation and quaternion
    algebra,” Comput Meth Biomech Biomed Eng, vol. 7, no. 3, pp. 159–166, 2004.

    Parameters
    ----------
    ts
        A TimeSeries with the following data keys:

        - ProximalJointPosition (Nx4)
        - DistalJointPosition (Nx4)
        - ForceApplicationPosition (Nx4)
        - DistalForces (Nx4)
        - DistalMoments (Nx4)

        and these ones (although they can be reconstructed automatically by
        the function, in which case warnings are issued):

        - COMPosition (Nx4)
        - COMAcceleration (Nx4)
        - SegmentAngles (Nx3)

    inertial_constants
        A dict that contains the following keys:

        - 'Mass': Mass of the segment, in kg.
        - 'GyrationCOMRatio': Radius of gyration around the segment's
          center of mass, as a ratio of the distance between
          both joints.

        This dict may be generated using the get_anthropometrics function.

    Returns
    -------
    TimeSeries
        A copy of the input timeseries plus these extra data keys:

        - 'ProximalForces' (Nx4)
        - 'ProximalMoments' (Nx4)
    """
    ts = ts.copy()

    n_frames = ts.time.shape[0]

    proximal_to_distal_joint_distance = (
        ts.data['DistalJointPosition'] -
        ts.data['ProximalJointPosition'])

    ts.data['RadiusOfGyration'] = np.zeros((n_frames, 3))
    ts.data['RadiusOfGyration'][:, 0] = \
        inertial_constants['GyrationCOMRatio'] * np.sqrt(
        proximal_to_distal_joint_distance[:, 1] ** 2 +
        proximal_to_distal_joint_distance[:, 2] ** 2)
    ts.data['RadiusOfGyration'][:, 1] =  \
        inertial_constants['GyrationCOMRatio'] * np.sqrt(
        proximal_to_distal_joint_distance[:, 0] ** 2 +
        proximal_to_distal_joint_distance[:, 2] ** 2)
    ts.data['RadiusOfGyration'][:, 2] =  \
        inertial_constants['GyrationCOMRatio'] * np.sqrt(
        proximal_to_distal_joint_distance[:, 0] ** 2 +
        proximal_to_distal_joint_distance[:, 1] ** 2)

    # Center of mass position and acceleration
    if 'COMPosition' not in ts.data:
        ts = calculate_com_position(ts, inertial_constants)
        warnings.warn(
            "COMPosition not found in input TimeSeries. I calculated it.")

    if 'COMAcceleration' not in ts.data:
        ts = calculate_com_acceleration(
            ts, filter_func='butter',
            fc=default_filter_fc, order=default_filter_order)
        warnings.warn(
            f"COMAcceleration not found in input TimeSeries. I calculated it "
            f"using a low-pass ButterWorth filter of order "
            f"{default_filter_order} at {default_filter_fc} Hz.")

    # Rotation angle, velocity and acceleration

    if (('AngularVelocity' not in ts.data) or
            ('AngularAcceleration' not in ts.data)):

        # Start by calculating the angles
        if 'SegmentAngles' not in ts.data:
            ts = calculate_segment_angles(ts)
            warnings.warn(
                "SegmentAngles not found in input TimeSeries. I calculated it.")

        ts = calculate_segment_rotation_rates(
            ts, filter_func='butter',
            fc=default_filter_fc, order=default_filter_order)
        warnings.warn(
            f"COMAcceleration not found in input TimeSeries. I calculated it "
            f"using a low-pass ButterWorth filter of order "
            f"{default_filter_order} at {default_filter_fc} Hz.")

    # Forces line of the wrench equation (16)
    a_i = ts.data['COMAcceleration'][:, 0:3]
    g = np.array([0, -9.81, 0])
    F_i_minus_1 = ts.data['DistalForces'][:, 0:3]

    # Moments line of the wrench equation (16)
    c_i = (ts.data['COMPosition'][:, 0:3] -
           ts.data['ProximalJointPosition'][:, 0:3])
    d_i = (ts.data['ForceApplicationPosition'] -
           ts.data['ProximalJointPosition'])[:, 0:3]

    segment_mass = inertial_constants['Mass']
    I_i_temp = segment_mass * ts.data['RadiusOfGyration'][:, 0:3] ** 2
    # Diagonalize I_i:
    I_i = np.zeros((n_frames, 3, 3))
    I_i[:, 0, 0] = I_i_temp[:, 0]
    I_i[:, 1, 1] = I_i_temp[:, 1]
    I_i[:, 2, 2] = I_i_temp[:, 2]

    alpha_i = ts.data['AngularAcceleration']
    omega_i = ts.data['AngularVelocity']

    M_i_minus_1 = ts.data['DistalMoments'][:, 0:3]

    # Calculation of the proximal wrench
    proximal_wrench = np.zeros((n_frames, 6, 1))
    for i_frame in range(n_frames):

        skew_symmetric_c_i = np.array([
            [0, -c_i[i_frame, 2], c_i[i_frame, 1]],
            [c_i[i_frame, 2], 0, -c_i[i_frame, 0]],
            [-c_i[i_frame, 1], c_i[i_frame, 0], 0]])

        skew_symmetric_d_i = np.array([
            [0, -d_i[i_frame, 2], d_i[i_frame, 1]],
            [d_i[i_frame, 2], 0, -d_i[i_frame, 0]],
            [-d_i[i_frame, 1], d_i[i_frame, 0], 0]])

        matrix_1 = np.block(
            [[segment_mass * np.eye(3), np.zeros((3, 3))],
             [segment_mass * skew_symmetric_c_i, I_i[i_frame]]])

        matrix_2 = np.block([a_i[i_frame] - g, alpha_i[i_frame]])
        matrix_2 = matrix_2[:, np.newaxis]  # Convert 1d to column vector

        matrix_3 = np.hstack([
            np.zeros(3),
            np.cross(omega_i[i_frame], I_i[i_frame] @ omega_i[i_frame])])
        matrix_3 = matrix_3[:, np.newaxis]  # Convert 1d to column vector

        matrix_4 = np.block([
            [np.eye(3), np.zeros((3, 3))],
            [skew_symmetric_d_i, np.eye(3)]])

        matrix_5 = np.block([F_i_minus_1[i_frame], M_i_minus_1[i_frame]])
        matrix_5 = matrix_5[:, np.newaxis]  # Convert 1d to column vector

        proximal_wrench[i_frame] = (
            matrix_1 @ matrix_2 + matrix_3 + matrix_4 @ matrix_5)

    # Initialize to a series of vectors of length 4
    ts.data['ProximalForces'] = np.zeros((n_frames, 4))
    ts.data['ProximalMoments'] = np.zeros((n_frames, 4))
    # Assign the 3 first components of the vectors
    ts.data['ProximalForces'][:, 0:3] = proximal_wrench[:, 0:3, 0]
    ts.data['ProximalMoments'][:, 0:3] = proximal_wrench[:, 3:6, 0]

    return ts


module_locals = locals()


def __dir__():
    return directory(module_locals)
