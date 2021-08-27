#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020-2021 Félix Chénier

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
Provide tools associated to anthropometric measurements and estimates.

Warning
-------
This module is in very early development and everything can still change.
Please don't use this module in production code.

Nomenclature
------------

Skin markers (either real or reconstructed):

- HeadVertex
- Sellion
- C7
- Suprasternale
- PubicSymphysis
- AnteriorSuperiorIliacSpineR/L
- PosteriorSuperiorIliacSpineR/L
- AcromionR/L
- LateralHumeralEpicondyleR/L
- MedialHumeralEpicondyleR/L
- UlnarStyloidR/L
- RadialStyloidR/L
- CarpalMetaHead2R/L
- CarpalMetaHead5R/L
- HipJointCenterR/L
- LateralFemoralEpicondyleR/L
- MedialFemoralEpicondyleR/L
- LateralMalleolusR/L
- MedialMalleolusR/L
- CalcaneusR/L
- NavicularR/L
- TarsalMetaHead1R/L
- TarsalMetaHead5R/L

Inferred joint centers:

- L5S1JointCenter
- C7T1JointCenter
- GlenohumeralJointCenterR/L
- ElbowJointCenterR/L
- WristJointCenterR/L
- KneeJointCenterR/L
- AnkleJointCenterR/L

"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2021 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


from kineticstoolkit.decorators import unstable, directory
from kineticstoolkit import TimeSeries, geometry, config, kinematics, Player
from typing import Dict
import numpy as np
import pandas as pd
from warnings import warn


@unstable
def infer_pelvis_joint_centers(
        markers: TimeSeries,
        sex: str = 'M') -> TimeSeries:
    """
    Infer L5S1 and hip joint centers based on anthropometric data.

    Creates L5S1JointCenter, HipJointCenterL and HipJointCenterR based on
    AnteriorSuperiorIliacSpineR/L, PosteriorSuperiorIliacSpineR/L and
    PubicSymphysis, following the method presented in Reed et al. (1999)[1]_.

    Parameters
    ----------
    markers
        TimeSeries that contain the trajectory of
        AnteriorSuperiorIliacSpineR/L, PosteriorSuperiorIliacSpineR/L and
        PubicSymphysis as Nx4 series.
    sex
        Optional. Either 'M' or 'F'. The default is 'M'.

    Returns
    -------
    TimeSeries
        A TimeSeries with L5S1JointCenter, HipJointCenterL and
        HipJointCenterR as Nx4 series.

    .. [1] Reed, M., Manary, M.A., Schneider, L.W., 1999.
       Methods for Measuring and Representing Automobile Occupant Posture.
       Presented at the International Congress &  Exposition, pp. 1999-01–0959.
       https://doi.org/10.4271/1999-01-0959

    """
    # Get the required markers
    try:
        rasis = markers.data['AnteriorSuperiorIliacSpineR']
        lasis = markers.data['AnteriorSuperiorIliacSpineL']
        rpsis = markers.data['PosteriorSuperiorIliacSpineR']
        lpsis = markers.data['PosteriorSuperiorIliacSpineL']
        sym = markers.data['PubicSymphysis']
    except KeyError:
        raise ValueError(
            "Not enough markers to reconstruct L5S1 and hip joint centers.")

    # Create a local coordinate system at the anterior superior iliac spines
    # midpoint, according to Reed et al.
    masis = 0.5 * (rasis + lasis)
    lcs = geometry.create_frames(
        origin=masis,
        y=lasis - rasis,
        yz=masis - sym
    )

    # Calculate the markers in the local coordinate system
    local_rasis = geometry.get_local_coordinates(rasis, lcs)
    local_lasis = geometry.get_local_coordinates(lasis, lcs)
    local_rpsis = geometry.get_local_coordinates(rpsis, lcs)
    local_lpsis = geometry.get_local_coordinates(lpsis, lcs)
    local_sym = geometry.get_local_coordinates(sym, lcs)

    # Create a rigid body definition using these locations
    pelvis_definition = {
        'AnteriorSuperiorIliacSpineR':
            np.nanmean(local_rasis, axis=0)[np.newaxis],
        'AnteriorSuperiorIliacSpineL':
            np.nanmean(local_lasis, axis=0)[np.newaxis],
        'PosteriorSuperiorIliacSpineR':
            np.nanmean(local_rpsis, axis=0)[np.newaxis],
        'PosteriorSuperiorIliacSpineL':
            np.nanmean(local_lpsis, axis=0)[np.newaxis],
        'PubicSymphysis':
            np.nanmean(local_sym, axis=0)[np.newaxis],
    }

    # Track the pelvis using this definition
    tracked_pelvis = kinematics.track_rigid_body(
        markers, pelvis_definition, 'Pelvis'
    )

    # Pelvis width
    pw = np.abs(np.nanmean(
        local_rasis[:, 1] - local_lasis[:, 1]
    ))

    output = markers.copy(copy_data=False, copy_data_info=False)

    # Create L5S1JointCenter
    if sex == 'F':
        local_position = np.array(
            [[-0.289 * pw, 0.0, 0.172 * pw, 1.0]]
        )
    elif sex == 'M':
        local_position = np.array(
            [[-0.264 * pw, 0.0, 0.126 * pw, 1.0]]
        )
    else:
        raise ValueError("sex must be either 'M' or 'F'")

    output.data['L5S1JointCenter'] = geometry.get_global_coordinates(
        local_position, tracked_pelvis.data['Pelvis']
    )

    # Create HipJointCenterR if not already included in markers
    if sex == 'F':
        local_position = np.array(
            [[-0.197 * pw, -0.372 * pw, -0.270 * pw, 1.0]]
        )
    else:  # M
        local_position = np.array(
            [[-0.208 * pw, -0.361 * pw, -0.278 * pw, 1.0]]
        )
    output.data['HipJointCenterR'] = geometry.get_global_coordinates(
        local_position, tracked_pelvis.data['Pelvis']
    )

    # Create HipJointCenterL if not already included in markers
    if sex == 'F':
        local_position = np.array(
            [[-0.197 * pw, 0.372 * pw, -0.270 * pw, 1.0]]
        )
    else:  # M
        local_position = np.array(
            [[-0.208 * pw, 0.361 * pw, -0.278 * pw, 1.0]]
        )
    output.data['HipJointCenterL'] = geometry.get_global_coordinates(
        local_position, tracked_pelvis.data['Pelvis']
    )

    return output


@unstable
def infer_thorax_joint_centers(
        markers: TimeSeries,
        sex: str = 'M') -> TimeSeries:
    """
    Infer C7T1 and glenohumeral joint centers based on anthropometric data.

    Creates C7T1JointCenter, GlenohumeralJointCenterL and
    GlenohumeralJointCenterR based on C7, L5S1JointCenter, Suprasternal,
    AcromionR and AcromionL, following the method presented in Reed et al.
    (1999)[1]_. The scapulo-thoracic joint must be in a neutral
    position.

    Warning
    -------
    Only male data is implemented at the moment.

    Parameters
    ----------
    markers
        TimeSeries that contain the trajectory of C7, L5S1JointCenter,
        Suprasternal, AcromionR and AcromionL as Nx4 series.
    sex
        Optional. Either 'M' or 'F'. The default is 'M'.

    Returns
    -------
    TimeSeries
        A TimeSeries with C7T1JointCenter, GlenohumeralJointCenterL and
        GlenohumeralJointCenterR as Nx4 series.

    .. [1] Reed, M., Manary, M.A., Schneider, L.W., 1999.
       Methods for Measuring and Representing Automobile Occupant Posture.
       Presented at the International Congress &  Exposition, pp. 1999-01–0959.
       https://doi.org/10.4271/1999-01-0959

    """
    # Get the required markers
    try:
        c7 = markers.data['C7']
        l5s1 = markers.data['L5S1JointCenter']
        sup = markers.data['Suprasternale']
        rac = markers.data['AcromionR']
        lac = markers.data['AcromionL']
    except KeyError:
        raise ValueError(
            "Not enough markers to reconstruct C7T1 and GH joint centers.")
        return

    if sex == 'M':
        c7t1_angle = 8  # deg
        c7t1_ratio = 0.55
        gh_angle = -67  # deg
        gh_ratio = 0.42
    # elif sex == 'F':
    #     c7t1_angle = 14  # deg
    #     c7t1_ratio = 0.53
    #     gh_angle = -5  # deg
    #     gh_ratio = 0.53
    else:
        raise ValueError("sex must be 'M' for now.")

    # Create reference frames with x: X7-SUP, y: L5S1-C7, z: right
    c7sup = sup - c7
    c7_lcs = geometry.create_frames(
        origin=c7,
        x=(c7sup),
        xy=(c7 - l5s1)
    )
    rac_lcs = geometry.create_frames(
        origin=rac,
        x=(c7sup),
        xy=(c7 - l5s1)
    )
    lac_lcs = geometry.create_frames(
        origin=lac,
        x=(c7sup),
        xy=(c7 - l5s1)
    )

    # Thorax width (tw)
    tw = np.sqrt(np.sum(
        np.nanmean(c7sup, axis=0) ** 2))

    # Local positions
    local_c7t1 = np.array([[
        c7t1_ratio * tw * np.cos(np.deg2rad(c7t1_angle)),
        c7t1_ratio * tw * np.sin(np.deg2rad(c7t1_angle)),
        0.0,
        1.0,
    ]])

    local_rgh = np.array([[
        gh_ratio * tw * np.cos(np.deg2rad(gh_angle)),
        gh_ratio * tw * np.sin(np.deg2rad(gh_angle)),
        0.0,
        1.0,
    ]])

    local_lgh = np.array([[
        gh_ratio * tw * np.cos(np.deg2rad(gh_angle)),
        gh_ratio * tw * np.sin(np.deg2rad(gh_angle)),
        0.0,
        1.0,
    ]])

    # Return global positions
    output = markers.copy(copy_data=False, copy_data_info=False)
    output.data['C7T1JointCenter'] = geometry.get_global_coordinates(
        local_c7t1, c7_lcs)
    output.data['GlenohumeralJointCenterR'] = \
        geometry.get_global_coordinates(
            local_rgh, rac_lcs)
    output.data['GlenohumeralJointCenterL'] = \
        geometry.get_global_coordinates(
            local_lgh, lac_lcs)

    return output


@unstable
def infer_extremity_joint_centers(
        markers: TimeSeries) -> TimeSeries:
    """
    Infer extremity joint centers based on medial and lateral markers.

    Creates:
    - ElbowJointCenterR/L as the midpoint between
      LateralHumeralEpicondyleR/L and MedialHumeralEpicondyleR;
    - WristJointCenterR as the midpoint between
      UlnarStyloidR/L and RadialStyloidR/L;
    - KneeJointCenterR/L as the midpoint between
      LateralFemoralEpicondyleR/L and MedialFemoralEpicondyleR/L;
    - AnkleJointCenterR/L as the midpoint between
      LateralMalleolusR/L and MedialMalleolusR/L.

    Parameters
    ----------
    markers
        TimeSeries the contains the required trajectories as Nx4 series.

    Returns
    -------
    TimeSeries
        A TimeSeries with the calculated joint center trajectories as Nx4
        series. Trajectories that could not be calculated due to missing
        markers are ignored.

    """
    output = markers.copy(copy_data=False, copy_data_info=False)
    try:
        output.data['ElbowJointCenterR'] = 0.5 * (
            markers.data['LateralHumeralEpicondyleR']
            + markers.data['MedialHumeralEpicondyleR'])
    except KeyError:
        pass

    try:
        output.data['ElbowJointCenterL'] = 0.5 * (
            markers.data['LateralHumeralEpicondyleL']
            + markers.data['MedialHumeralEpicondyleL'])
    except KeyError:
        pass

    try:
        output.data['KneeJointCenterR'] = 0.5 * (
            markers.data['LateralFemoralEpicondyleR']
            + markers.data['MedialFemoralEpicondyleR'])
    except KeyError:
        pass

    try:
        output.data['KneeJointCenterL'] = 0.5 * (
            markers.data['LateralFemoralEpicondyleL']
            + markers.data['MedialFemoralEpicondyleL'])
    except KeyError:
        pass

    try:
        output.data['WristJointCenterR'] = 0.5 * (
            markers.data['RadialStyloidR']
            + markers.data['UlnarStyloidR'])
    except KeyError:
        pass

    try:
        output.data['WristJointCenterL'] = 0.5 * (
            markers.data['RadialStyloidL']
            + markers.data['UlnarStyloidL'])
    except KeyError:
        pass

    try:
        output.data['AnkleJointCenterR'] = 0.5 * (
            markers.data['LateralMalleolusR']
            + markers.data['MedialMalleolusR'])
    except KeyError:
        pass

    try:
        output.data['AnkleJointCenterL'] = 0.5 * (
            markers.data['LateralMalleolusL']
            + markers.data['MedialMalleolusL'])
    except KeyError:
        pass

    return output


def define_local_coordinate_system(
        markers: TimeSeries,
        segment: str
) -> Dict[str, np.ndarray]:
    """
    Create local coordinate system definitions based on static markers.

    Parameters
    ----------
    markers
        TimeSeries that contains marker trajectories as Nx4
        series, ideally recorded during a short static acquisition.

    segment
        Name of the segment. Can be 'Pelvis', 'Thorax', 'HeadNeck',
        'ArmR', 'ArmL', 'ForearmR', 'ForearmL', 'HandR', 'HandL', 'ThighR',
        'ThighL', 'LegR', 'LegL', 'FootR', 'FootL', a sequence (e.g., list) of
        many segments, or 'all'.

    Returns
    -------
    Dict[str, np.ndarray]
        A dict that contains the local position of the segment's markers as
        1x4 arrays.

    Notes
    -----
    The pelvis local coordinate system is located at LS51, with X anterior,
    Y up and Z right. The segment definition is based on
    Dumas et al. (2007) [1]_.

    The required markers are:
    - AnteriorSuperiorIliacSpineR
    - AnteriorSuperiorIliacSpineL
    - PosteriorSuperiorIliacSpineR
    - PosteriorSuperiorIliacSpineL
    - L5S1JointCenter

    Additionally, if the following markers are also included in the
    TimeSeries, they will also be expressed in local coordinates in the
    returned dictionary:

    - PubicSymphysis
    - HipJointCenterR
    - HipJointCenterL

    The thorax local coordinate system is located at C7T1JointCenter,
    with X anterior, Y up and Z right. The segment definition is based on
    Dumas et al. (2007) [1]_.

    This required markers are:
    - Suprasternale
    - L5S1JointCenter
    - C7T1JointCenter




    .. [1] Dumas, R., Chèze, L., Verriest, J.-P., 2007.
       Adjustments to McConville et al. and Young et al. body segment inertial
       parameters. Journal of Biomechanics 40, 543–553.
       https://doi.org/10.1016/j.jbiomech.2006.02.013

    """
    if segment == 'Pelvis':
        return _define_pelvis_coordinate_system(markers)
    elif segment == 'Thorax':
        return _define_thorax_coordinate_system(markers)
    else:
        raise ValueError("This segment is not recognized.")


def _define_pelvis_coordinate_system(
        markers: TimeSeries) -> Dict[str, np.ndarray]:
    """Create the Pelvis definition based on static markers."""
    # Get the required markers
    try:
        rasis = markers.data['AnteriorSuperiorIliacSpineR']
        lasis = markers.data['AnteriorSuperiorIliacSpineL']
        rpsis = markers.data['PosteriorSuperiorIliacSpineR']
        lpsis = markers.data['PosteriorSuperiorIliacSpineL']
        l5s1 = markers.data['L5S1JointCenter']
    except KeyError:
        raise ValueError(
            "Not enough markers to create the pelvis coordinate system."
        )

    # Create the local coordinate system according to Dumas et al. (2007)
    mpsis = 0.5 * (rpsis + lpsis)
    lcs = geometry.create_frames(
        origin=l5s1, z=(rasis - lasis),
        xz=(0.5 * (rasis + lasis) - mpsis),
    )

    # Create the pelvis definition
    pelvis_definition = {}
    for marker in [
        'AnteriorSuperiorIliacSpineR',
        'AnteriorSuperiorIliacSpineL',
        'PosteriorSuperiorIliacSpineR',
        'PosteriorSuperiorIliacSpineL',
        'PubicSymphysis',
        'L5S1',
        'HipJointCenterR',
        'HipJointCenterL',
    ]:
        if marker in markers.data:
            pelvis_definition[marker] = np.nanmean(
                geometry.get_local_coordinates(
                    markers.data[marker],
                    lcs
                ), axis=0)[np.newaxis, :]

    return pelvis_definition


def _define_thorax_coordinate_system(
        markers: TimeSeries) -> Dict[str, np.ndarray]:
    """Create the Thorax definition based on static markers."""
    # Get the required markers
    try:
        c7t1 = markers.data['C7T1JointCenter']
        sup = markers.data['Suprasternale']
        l5s1 = markers.data['L5S1JointCenter']
    except KeyError:
        raise ValueError(
            "Not enough markers to create the thorax coordinate system."
        )

    # Create the local coordinate system
    lcs = geometry.create_frames(
        origin=c7t1, y=(c7t1 - l5s1),
        xy=(sup - l5s1),
    )

    # Create the thorax definition
    thorax_definition = {}
    for marker in [
        'C7T1JointCenter',
        'Suprasternale',
        'L5S1JointCenter',
    ]:
        thorax_definition[marker] = np.nanmean(
            geometry.get_local_coordinates(
                markers.data[marker],
                lcs
            ), axis=0)[np.newaxis, :]

    return thorax_definition


def define_head_neck_coordinate_system(
        markers: TimeSeries) -> Dict[str, np.ndarray]:
    """
    Create the Head+Neck definition based on static markers.

    The head+neck local coordinate system is located at C7T1JointCenter,
    with X anterior, Y up and Z right. The segment definition is based on
    Dumas et al. (2007) [1]_.

    Parameters
    ----------
    markers
        TimeSeries that contains minimally the following markers as Nx4
        series, ideally recorded during a short static acquisition:

        - C7T1JointCenter
        - Sellion
        - HeadVertex

    Returns
    -------
    Dict[str, np.ndarray]
        A dict that contains the local position of the head+neck markers as
        1x4 arrays.

    .. [1] Dumas, R., Chèze, L., Verriest, J.-P., 2007.
       Adjustments to McConville et al. and Young et al. body segment inertial
       parameters. Journal of Biomechanics 40, 543–553.
       https://doi.org/10.1016/j.jbiomech.2006.02.013

    """
    # Get the required markers
    try:
        c7t1 = markers.data['C7T1JointCenter']
        sel = markers.data['Sellion']
        hv = markers.data['HeadVertex']
    except KeyError:
        raise ValueError(
            "Not enough markers to create the thorax coordinate system."
        )

    # Create the local coordinate system
    lcs = geometry.create_frames(
        origin=c7t1, y=(hv - c7t1),
        xy=(sel - c7t1),
    )

    # Create the thorax definition
    head_neck_definition = {}
    for marker in [
        'C7T1JointCenter',
        'Sellion',
        'HeadVertex',
    ]:
        head_neck_definition[marker] = np.nanmean(
            geometry.get_local_coordinates(
                markers.data[marker],
                lcs
            ), axis=0)[np.newaxis, :]

    return head_neck_definition


def define_arm_coordinate_system(
        markers: TimeSeries, side: str = 'R') -> Dict[str, np.ndarray]:
    """
    Create the ArmR or ArmL definition based on static markers.

    The arm local coordinate system is located at GlenohumeralJointCenterR/L,
    with X anterior, Y up and Z right. The segment definition is based on
    the ISB, using the 1st definition with both humeral epicondyles [1]_.

    Parameters
    ----------
    markers
        TimeSeries that contains minimally the following markers as Nx4
        series, ideally recorded during a short static acquisition:

        - GlenohumeralJointCenterR or GlenohumeralJointCenterL
        - ElbowJointCenterR or ElbowJointCenterL
        - LateralHumeralEpicondyleR or LateralHumeralEpicondyleL
        - MedialHumeralEpicondyleR or MedialHumeralEpicondyleR

    side
        Optional. Either 'R' or 'L'. The default is 'R'.

    Returns
    -------
    Dict[str, np.ndarray]
        A dict that contains the local position of the arm markers as
        1x4 arrays.

    .. [1] Wu, G. et al., 2005.
       ISB recommendation on definitions of joint coordinate systems of
       various joints for the reporting of human joint motion - Part II:
       shoulder, elbow, wrist and hand. Journal of Biomechanics 38, 981–992.
       https://doi.org/10.1016/j.jbiomech.2004.05.042

    """
    # Get the required markers
    try:
        elbow_center = markers.data[f'ElbowJointCenter{side}']
        gh = markers.data[f'GlenohumeralJointCenter{side}']
        if side == 'R':
            r_ep = markers.data['LateralHumeralEpicondyleR']
            l_ep = markers.data['MedialHumeralEpicondyleR']
        elif side == 'L':
            r_ep = markers.data['MedialHumeralEpicondyleL']
            l_ep = markers.data['LateralHumeralEpicondyleL']
        else:
            raise ValueError(
                "side must be either 'R' or 'L'"
            )

    except KeyError:
        raise ValueError(
            "Not enough markers to create the arm coordinate system."
        )

    # Create the local coordinate system
    lcs = geometry.create_frames(
        origin=gh, y=(gh - elbow_center),
        yz=(r_ep - l_ep),
    )

    # Create the arm definition
    arm_definition = {}
    for marker in [
        f'GlenohumeralJointCenter{side}',
        f'ElbowJointCenter{side}',
        f'LateralHumeralEpicondyle{side}',
        f'LateralHumeralEpicondyle{side}',
    ]:
        arm_definition[marker] = np.nanmean(
            geometry.get_local_coordinates(
                markers.data[marker],
                lcs
            ), axis=0)[np.newaxis, :]

    return arm_definition


def define_forearm_coordinate_system(
        markers: TimeSeries, side: str = 'R') -> Dict[str, np.ndarray]:
    """
    Create the ForearmR or ForearmL definition based on static markers.

    The forearm local coordinate system is located at ElbowJointCenterR/L,
    with X anterior, Y up and Z right. The segment definition is based on
    the ISB [1]_.

    Parameters
    ----------
    markers
        TimeSeries that contains minimally the following markers as Nx4
        series, ideally recorded during a short static acquisition:

        - ElbowJointCenterR or ElbowJointCenterL
        - WristJointCenterR or WristJointCenterL
        - UlnarStyloidR or UlnarStyloidL
        - RadialStyloidR or RadialStyloidL

    side
        Optional. Either 'R' or 'L'. The default is 'R'.

    Returns
    -------
    Dict[str, np.ndarray]
        A dict that contains the local position of the forearm markers as
        1x4 arrays.

    .. [1] Wu, G. et al., 2005.
       ISB recommendation on definitions of joint coordinate systems of
       various joints for the reporting of human joint motion - Part II:
       shoulder, elbow, wrist and hand. Journal of Biomechanics 38, 981–992.
       https://doi.org/10.1016/j.jbiomech.2004.05.042

    """
    # Get the required markers
    try:
        elbow_center = markers.data[f'ElbowJointCenter{side}']
        wrist_center = markers.data[f'WristJointCenter{side}']
        if side == 'R':
            r_st = markers.data['RadialStyloidR']
            l_st = markers.data['UlnarStyloidR']
        elif side == 'L':
            r_st = markers.data['UlnarStyloidL']
            l_st = markers.data['RadialStyloidL']
        else:
            raise ValueError(
                "side must be either 'R' or 'L'"
            )
    except KeyError:
        raise ValueError(
            "Not enough markers to create the forearm coordinate system."
        )

    # Create the local coordinate system
    lcs = geometry.create_frames(
        origin=elbow_center, y=(elbow_center - wrist_center),
        yz=(r_st - l_st),
    )

    # Create the forearm definition
    forearm_definition = {}
    for marker in [
        f'ElbowJointCenter{side}',
        f'WristJointCenter{side}',
        f'UlnarStyloid{side}',
        f'RadialStyloid{side}',
    ]:
        forearm_definition[marker] = np.nanmean(
            geometry.get_local_coordinates(
                markers.data[marker],
                lcs
            ), axis=0)[np.newaxis, :]

    return forearm_definition


def define_hand_coordinate_system(
        markers: TimeSeries, side: str = 'R') -> Dict[str, np.ndarray]:
    """
    Create the HandR or HandL definition based on static markers.

    The forearm local coordinate system is located at WristJointCenterR/L,
    with X anterior, Y up and Z right. The segment definition is based on
    the ISB [1]_.

    Parameters
    ----------
    markers
        TimeSeries that contains minimally the following markers as Nx4
        series, ideally recorded during a short static acquisition:

        - WristJointCenterR or WristJointCenterL
        - CarpalMetaHead2R or CarpalMetaHead2L
        - CarpalMetaHead5R or CarpalMetaHead5L

    side
        Optional. Either 'R' or 'L'. The default is 'R'.

    Returns
    -------
    Dict[str, np.ndarray]
        A dict that contains the local position of the hand markers as
        1x4 arrays.

    .. [1] Wu, G. et al., 2005.
       ISB recommendation on definitions of joint coordinate systems of
       various joints for the reporting of human joint motion - Part II:
       shoulder, elbow, wrist and hand. Journal of Biomechanics 38, 981–992.
       https://doi.org/10.1016/j.jbiomech.2004.05.042

    """
    # Get the required markers
    try:
        wrist_center = markers.data[f'WristJointCenter{side}']
        if side == 'R':
            r_meta = markers.data['CarpalMetaHead2R']
            l_meta = markers.data['CarpalMetaHead5R']
        elif side == 'L':
            r_meta = markers.data['CarpalMetaHead5L']
            l_meta = markers.data['CarpalMetaHead2L']
        else:
            raise ValueError(
                "side must be either 'R' or 'L'"
            )
    except KeyError:
        raise ValueError(
            "Not enough markers to create the hand coordinate system."
        )

    # Create the local coordinate system
    meta_center = 0.5 * (r_meta + l_meta)
    lcs = geometry.create_frames(
        origin=wrist_center, y=(wrist_center - meta_center),
        yz=(r_meta - l_meta),
    )

    # Create the hand definition
    hand_definition = {}
    for marker in [
        f'ElbowJointCenter{side}',
        f'WristJointCenter{side}',
        f'UlnarStyloid{side}',
        f'RadialStyloid{side}',
    ]:
        hand_definition[marker] = np.nanmean(
            geometry.get_local_coordinates(
                markers.data[marker],
                lcs
            ), axis=0)[np.newaxis, :]

    return hand_definition


def define_thigh_coordinate_system(
        markers: TimeSeries, side: str = 'R') -> Dict[str, np.ndarray]:
    """
    Create the ThighR or ThighL definition based on static markers.

    The thigh local coordinate system is located at HipJointCenterR/L,
    with X anterior, Y up and Z right. The segment definition is based on
    the ISB [2]_.

    Parameters
    ----------
    markers
        TimeSeries that contains minimally the following markers as Nx4
        series, ideally recorded during a short static acquisition:

        - HipJointCenterR or HipJointCenterL
        - KneeJointCenterR or KneeJointCenterL
        - LateralFemoralEpicondyleR or LateralFemoralEpicondyleL
        - MedialFemoralEpicondyleR or MedialFemoralEpicondyleL

    side
        Optional. Either 'R' or 'L'. The default is 'R'.

    Returns
    -------
    Dict[str, np.ndarray]
        A dict that contains the local position of the thigh markers as
        1x4 arrays.

    .. [2] Wu, G. et al., 2002.
       ISB recommendation on definitions of joint coordinate system of various
       joints for the reporting of human joint motion—part I: ankle, hip, and
       spine. Journal of Biomechanics 35, 543–548.
       https://doi.org/10.1016/S0021-9290(01)00222-6

    """
    # Get the required markers
    try:
        hip_center = markers.data[f'HipJointCenter{side}']
        knee_center = markers.data[f'KneeJointCenter{side}']
        if side == 'R':
            r_ep = markers.data['LateralFemoralEpicondyleR']
            l_ep = markers.data['MedialFemoralEpicondyleR']
        elif side == 'L':
            r_ep = markers.data['MedialFemoralEpicondyleL']
            l_ep = markers.data['LateralFemoralEpicondyleL']
        else:
            raise ValueError(
                "side must be either 'R' or 'L'"
            )
    except KeyError:
        raise ValueError(
            "Not enough markers to create the thigh coordinate system."
        )

    # Create the local coordinate system
    lcs = geometry.create_frames(
        origin=hip_center, y=(hip_center - knee_center),
        yz=(r_ep - l_ep),
    )

    # Create the thigh definition
    thigh_definition = {}
    for marker in [
        f'HipJointCenter{side}',
        f'KneeJointCenter{side}',
        f'LateralFemoralEpicondyle{side}',
        f'MedialFemoralEpicondyle{side}',
    ]:
        thigh_definition[marker] = np.nanmean(
            geometry.get_local_coordinates(
                markers.data[marker],
                lcs
            ), axis=0)[np.newaxis, :]

    return thigh_definition


def define_leg_coordinate_system(
        markers: TimeSeries, side: str = 'R') -> Dict[str, np.ndarray]:
    """
    Create the LegR or LegL definition based on static markers.

    The leg local coordinate system is located at KneeJointCenterR/L,
    with X anterior, Y up and Z right. The segment definition is based on
    the ISB [2]_.

    Parameters
    ----------
    markers
        TimeSeries that contains minimally the following markers as Nx4
        series, ideally recorded during a short static acquisition:

        - AnkleJointCenterR or AnkleJointCenterL
        - KneeJointCenterR or KneeJointCenterL
        - LateralMalleolusR or LateralMalleolusL
        - MedialMalleolusR or MedialMalleolusL

    side
        Optional. Either 'R' or 'L'. The default is 'R'.

    Returns
    -------
    Dict[str, np.ndarray]
        A dict that contains the local position of the leg markers as
        1x4 arrays.

    .. [2] Wu, G. et al., 2002.
       ISB recommendation on definitions of joint coordinate system of various
       joints for the reporting of human joint motion—part I: ankle, hip, and
       spine. Journal of Biomechanics 35, 543–548.
       https://doi.org/10.1016/S0021-9290(01)00222-6

    """
    # Get the required markers
    try:
        knee_center = markers.data[f'KneeJointCenter{side}']
        ankle_center = markers.data[f'AnkleJointCenter{side}']
        if side == 'R':
            r_mal = markers.data['LateralMalleolusR']
            l_mal = markers.data['MedialMalleolusR']
        elif side == 'L':
            r_mal = markers.data['MedialMalleolusL']
            l_mal = markers.data['LateralMalleolusL']
        else:
            raise ValueError(
                "side must be either 'R' or 'L'"
            )
    except KeyError:
        raise ValueError(
            "Not enough markers to create the leg coordinate system."
        )

    # Create the local coordinate system
    lcs = geometry.create_frames(
        origin=knee_center, y=(knee_center - ankle_center),
        yz=(r_mal - l_mal),
    )

    # Create the leg definition
    leg_definition = {}
    for marker in [
        f'KneeJointCenter{side}',
        f'AnkleJointCenter{side}',
        f'LateralMalleolus{side}',
        f'MedialMalleolus{side}',
    ]:
        leg_definition[marker] = np.nanmean(
            geometry.get_local_coordinates(
                markers.data[marker],
                lcs
            ), axis=0)[np.newaxis, :]

    return leg_definition


def define_foot_coordinate_system(
        markers: TimeSeries, side: str = 'R') -> Dict[str, np.ndarray]:
    """
    Create the FootR or FootL definition based on static markers.

    The foot local coordinate system is located at AnkleJointCenterR/L,
    with X anterior, Y up and Z right. The segment definition is based on
    the ISB [2]_.

    Parameters
    ----------
    markers
        TimeSeries that contains minimally the following markers as Nx4
        series, ideally recorded during a short static acquisition:

        - AnkleJointCenterR or AnkleJointCenterL
        - CalcaneousR or CalcaneousL
        - TarsalMetaHead1R or TarsalMetaHead1L
        - TarsalMetaHead5R or TarsalMetaHead5L

    side
        Optional. Either 'R' or 'L'. The default is 'R'.

    Returns
    -------
    Dict[str, np.ndarray]
        A dict that contains the local position of the foot markers as
        1x4 arrays.

    .. [2] Wu, G. et al., 2002.
       ISB recommendation on definitions of joint coordinate system of various
       joints for the reporting of human joint motion—part I: ankle, hip, and
       spine. Journal of Biomechanics 35, 543–548.
       https://doi.org/10.1016/S0021-9290(01)00222-6

    """
    # Get the required markers
    try:
        ankle_center = markers.data[f'AnkleJointCenter{side}']
        calc = markers.data[f'Calcaneus{side}']
        if side == 'R':
            r_meta = markers.data['TarsalMetaHead5R']
            l_meta = markers.data['TarsalMetaHead1R']
        elif side == 'L':
            r_meta = markers.data['TarsalMetaHead1L']
            l_meta = markers.data['TarsalMetaHead5L']
        else:
            raise ValueError(
                "side must be either 'R' or 'L'"
            )
    except KeyError:
        raise ValueError(
            "Not enough markers to create the foot coordinate system."
        )

    # Create the local coordinate system
    meta_center = 0.5 * (r_meta + l_meta)
    lcs = geometry.create_frames(
        origin=ankle_center, x=(meta_center - calc),
        xz=(r_meta - l_meta),
    )

    # Create the foot definition
    foot_definition = {}
    for marker in [
        f'AnkleJointCenter{side}',
        f'TarsalMetaHead1{side}',
        f'TarsalMetaHead5{side}',
        f'Calcaneus{side}',
    ]:
        foot_definition[marker] = np.nanmean(
            geometry.get_local_coordinates(
                markers.data[marker],
                lcs
            ), axis=0)[np.newaxis, :]

    return foot_definition


def _define_coordinate_systems(
    markers: TimeSeries, /, *, sex: str = 'M'
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Create local coordinate systems for the body segments.

    References
    ----------

    .. [2] Reed, M., Manary, M.A., Schneider, L.W., 1999.
       Methods for Measuring and Representing Automobile Occupant Posture.
       Presented at the International Congress &  Exposition, pp. 1999-01–0959.
       https://doi.org/10.4271/1999-01-0959

    .. [3] Reynolds, H.M., Snow, C.C., Young, J.W., 1982.
       Spatial Geometry of the Human Pelvis.
       FEDERAL AVIATION ADMINISTRATION OKLAHOMA CITY OK CIVIL AEROMEDICAL INST.

    """
    definitions = {}  # type: Dict[str, Dict[str, np.ndarray]]

    # Create center of mass definitions, from Dumas 2007, Table 2.
    length = np.sqrt(
        (definitions['Pelvis']['HipJointCenterR'][0, 0]  # x
         - definitions['Pelvis']['L5S1'][0, 0]) ** 2
        +
        (definitions['Pelvis']['HipJointCenterR'][0, 1]  # y
         - definitions['Pelvis']['L5S1'][0, 1]) ** 2
    )

    if sex == 'F':
        definitions['Pelvis']['PelvisCOM'] = (
            0.01 * np.array([[-0.9, -23.2, 0.2, 0.0]]) * length
        )
    else:  # 'M'
        definitions['Pelvis']['PelvisCOM'] = (
            0.01 * np.array([[2.8, -28.0, -0.6, 0.0]]) * length
        )
    definitions['Pelvis']['PelvisCOM'][0, 3] = 1.0

    # Visual test
    bodies = kinematics.track_rigid_body(
        markers, definitions['Pelvis'], 'Pelvis')

    Player(markers, bodies, segments=LINKS)

    return definitions


# %% Constants

#: A link model to help kinematics visualization
LINKS = {
    'Pelvis': {
        'Links': [
            ['AnteriorSuperiorIliacSpineR', 'AnteriorSuperiorIliacSpineL'],
            ['AnteriorSuperiorIliacSpineL', 'PosteriorSuperiorIliacSpineL'],
            ['PosteriorSuperiorIliacSpineL', 'PosteriorSuperiorIliacSpineR'],
            ['PosteriorSuperiorIliacSpineR', 'AnteriorSuperiorIliacSpineR'],

            ['AnteriorSuperiorIliacSpineR', 'HipJointCenterR'],
            ['PosteriorSuperiorIliacSpineR', 'HipJointCenterR'],
            ['AnteriorSuperiorIliacSpineL', 'HipJointCenterL'],
            ['PosteriorSuperiorIliacSpineL', 'HipJointCenterL'],

            ['AnteriorSuperiorIliacSpineR', 'PubicSymphysis'],
            ['AnteriorSuperiorIliacSpineL', 'PubicSymphysis'],
            ['HipJointCenterR', 'PubicSymphysis'],
            ['HipJointCenterL', 'PubicSymphysis'],
            ['HipJointCenterL', 'HipJointCenterR'],
        ],
        'Color': [0.25, 0.5, 0.25],
    },
    'Trunk': {
        'Links': [
            ['L5S1JointCenter', 'C7T1JointCenter'],

            ['C7', 'GlenohumeralJointCenterR'],
            ['C7', 'GlenohumeralJointCenterL'],
            ['Suprasternale', 'GlenohumeralJointCenterR'],
            ['Suprasternale', 'GlenohumeralJointCenterL'],

            ['C7', 'AcromionR'],
            ['C7', 'AcromionL'],
            ['Suprasternale', 'AcromionR'],
            ['Suprasternale', 'AcromionL'],

            ['GlenohumeralJointCenterR', 'AcromionR'],
            ['GlenohumeralJointCenterL', 'AcromionL'],

        ],
        'Color': [0.5, 0.5, 0],
    },
    'HeadNeck': {
        'Links': [
            ['Sellion', 'C7T1JointCenter'],
            ['HeadVertex', 'C7T1JointCenter'],
            ['Sellion', 'HeadVertex'],
        ],
        'Color': [0.5, 0.5, 0.25],
    },
    'UpperArms': {
        'Links': [
            ['GlenohumeralJointCenterR', 'ElbowJointCenterR'],
            ['GlenohumeralJointCenterR', 'LateralHumeralEpicondyleR'],
            ['GlenohumeralJointCenterR', 'MedialHumeralEpicondyleR'],
            ['LateralHumeralEpicondyleR', 'MedialHumeralEpicondyleR'],

            ['GlenohumeralJointCenterL', 'ElbowJointCenterL'],
            ['GlenohumeralJointCenterL', 'LateralHumeralEpicondyleL'],
            ['GlenohumeralJointCenterL', 'MedialHumeralEpicondyleL'],
            ['LateralHumeralEpicondyleL', 'MedialHumeralEpicondyleL'],
        ],
        'Color': [0.5, 0.25, 0],
    },
    'Forearms': {
        'Links': [
            ['ElbowJointCenterR', 'WristJointCenterR'],
            ['RadialStyloidR', 'LateralHumeralEpicondyleR'],
            ['UlnarStyloidR', 'MedialHumeralEpicondyleR'],
            ['RadialStyloidR', 'UlnarStyloidR'],

            ['ElbowJointCenterL', 'WristJointCenterL'],
            ['RadialStyloidL', 'LateralHumeralEpicondyleL'],
            ['UlnarStyloidL', 'MedialHumeralEpicondyleL'],
            ['RadialStyloidL', 'UlnarStyloidL'],
        ],
        'Color': [0.5, 0, 0],
    },
    'Hands': {
        'Links': [
            ['RadialStyloidR', 'CarpalMetaHead2R'],
            ['UlnarStyloidR', 'CarpalMetaHead5R'],
            ['CarpalMetaHead2R', 'CarpalMetaHead5R'],

            ['RadialStyloidL', 'CarpalMetaHead2L'],
            ['UlnarStyloidL', 'CarpalMetaHead5L'],
            ['CarpalMetaHead2L', 'CarpalMetaHead5L'],
        ],
        'Color': [0.5, 0, 0],
    },
    'Tighs': {
        'Links': [
            ['HipJointCenterR', 'KneeJointCenterR'],
            ['HipJointCenterR', 'LateralFemoralEpicondyleR'],
            ['HipJointCenterR', 'MedialFemoralEpicondyleR'],
            ['LateralFemoralEpicondyleR', 'MedialFemoralEpicondyleR'],

            ['HipJointCenterL', 'KneeJointCenterL'],
            ['HipJointCenterL', 'LateralFemoralEpicondyleL'],
            ['HipJointCenterL', 'MedialFemoralEpicondyleL'],
            ['LateralFemoralEpicondyleL', 'MedialFemoralEpicondyleL'],
        ],
        'Color': [0, 0.5, 0.5],
    },
    'Legs': {
        'Links': [
            ['AnkleJointCenterR', 'KneeJointCenterR'],
            ['LateralMalleolusR', 'LateralFemoralEpicondyleR'],
            ['MedialMalleolusR', 'MedialFemoralEpicondyleR'],
            ['LateralMalleolusR', 'MedialMalleolusR'],

            ['AnkleJointCenterL', 'KneeJointCenterL'],
            ['LateralMalleolusL', 'LateralFemoralEpicondyleL'],
            ['MedialMalleolusL', 'MedialFemoralEpicondyleL'],
            ['LateralMalleolusL', 'MedialMalleolusL'],
        ],
        'Color': [0, 0.25, 0.5],
    },

}


# Load Inertial Values
INERTIAL_VALUES = {}

# Dumas2007
_ = pd.read_csv(config.root_folder + '/data/anthropometrics/dumas_2007.csv')
_[['RelIXY', 'RelIXZ', 'RelIYZ']] = (
    _[['RelIXY', 'RelIXZ', 'RelIYZ']].applymap(lambda s: complex(s))
)
INERTIAL_VALUES['Dumas2007'] = _


module_locals = locals()


def __dir__():
    return directory(module_locals)


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
