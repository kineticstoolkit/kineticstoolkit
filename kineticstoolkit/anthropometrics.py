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
- AnteriorSuperiorIliacSpineL/R
- PosteriorSuperiorIliacSpineL/R
- PubicSymphysis
- L5S1JointCenter
- C7
- C7T1JointCenter
- Suprasternale
- AcromionL/R
- GlenohumeralJointCenterL/R
- LateralHumeralEpicondyleL/R
- MedialHumeralEpicondyleL/R
- ElbowJointCenterL/R
- UlnarStyloidL/R
- RadialStyloidL/R
- WristJointCenterL/R
- CarpalMetaHead1-5
- FingerTip1-5L/R
- HipJointCenterL/R
- LateralFemoralEpicondyleL/R
- MedialFemoralEpicondyleL/R
- KneeJointCenterL/R
- LateralMalleolusL/R
- MedialMalleolusL/R
- AnkleJointCenterL/R
- CalcaneusL/R
- NavicularL/R
- TarsalMetaHead1-5L/R
- ToeTip1-5L/R

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
def infer_hip_l5s1_joint_centers(
        markers: TimeSeries,
        sex: str = 'M') -> TimeSeries:
    """
    Infer L5S1 and hip joint centers based on anthropometric data.

    Creates L5S1JointCenter, HipJointCenterL and HipJointCenterR based on
    AnteriorSuperiorIliacSpineL/R, PosteriorSuperiorIliacSpineL/R and
    PubicSymphysis, following the method presented in Reed et al. (1999)[1]_.

    Parameters
    ----------
    markers
        TimeSeries that contain the trajectory of
        AnteriorSuperiorIliacSpineL/R, PosteriorSuperiorIliacSpineL/R and
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
        warn("Not enough markers to reconstruct L5S1 and hip joint centers.")
        return

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

    output.data['L5S1'] = geometry.get_global_coordinates(
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
def infer_glenohumeral_c7t1_joint_centers(
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
        l5s1 = markers.data['L5S1']
        sup = markers.data['Suprasternale']
        rac = markers.data['AcromionR']
        lac = markers.data['AcromionL']
    except KeyError:
        warn("Not enough markers to reconstruct C7T1 and GH joint centers.")
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
    output.data['C7T1'] = geometry.get_global_coordinates(
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
    - ElbowJointCenterL/R as the midpoint between
      LateralHumeralEpicondyleL/R and MedialHumeralEpicondyleR;
    - WristJointCenterR as the midpoint between
      UlnarStyloidL/R and RadialStyloidL/R;
    - KneeJointCenterL/R as the midpoint between
      LateralFemoralEpicondyleL/R and MedialFemoralEpicondyleL/R;
    - AnkleJointCenterL/R as the midpoint between
      LateralMalleolusL/R and MedialMalleolusL/R.

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


def _define_pelvis(
        markers: TimeSeries,
        sex: str = 'M') -> Dict[str, np.ndarray]:
    """
    Create the Pelvis definition based on static markers.

    The pelvis local coordinate system is located at LS51, with X anterior,
    Y up and Z right. The segment definition is based on
    Dumas et al. (2007) [1]_, which is based on Reed et al. (1999) [2]_ and
    Reynolds et al. (1982) [3]_.

    Parameters
    ----------
    markers
        TimeSeries that contains the following markers as Nx4 series, ideally
        recorded during a short static acquisition:

        - AnteriorSuperiorIliacSpineR
        - AnteriorSuperiorIliacSpineL
        - PosteriorSuperiorIliacSpineR
        - PosteriorSuperiorIliacSpineL
        - PubicSymphysis

    sex
        Optional. 'F' for female, 'M' for male. The default is 'M'.

    Returns
    -------
    Dict[str, np.ndarray]
        A dict that contains the following data as positions in the
        pelvis local coordinate system as 1x4 arrays:

        - AnteriorSuperiorIliacSpineR
        - AnteriorSuperiorIliacSpineL
        - HipJointCenterR
        - HipJointCenterL
        - L5S1
        - PosteriorSuperiorIliacSpineR
        - PosteriorSuperiorIliacSpineL
        - PubicSymphysis

    The markers TimeSeries is also modified: markers L5S1, HipJointCenterR
    and HipJointCenterL are created if they didn't already exist.

    """
    markers.merge(
        infer_hip_l5s1_joint_centers(markers, sex),
        in_place=True,
    )
    markers.merge(
        infer_glenohumeral_c7t1_joint_centers(markers, sex),
        in_place=True,
    )
    markers.merge(
        infer_extremity_joint_centers(markers),
        in_place=True,
    )

    # Get the required markers
    try:
        rasis = markers.data['AnteriorSuperiorIliacSpineR']
        lasis = markers.data['AnteriorSuperiorIliacSpineL']
        rpsis = markers.data['PosteriorSuperiorIliacSpineR']
        lpsis = markers.data['PosteriorSuperiorIliacSpineL']
        l5s1 = markers.data['L5S1']
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
        pelvis_definition[marker] = np.nanmean(
            geometry.get_local_coordinates(
                markers.data[marker],
                lcs
            ), axis=0)[np.newaxis, :]

    return pelvis_definition


def define_coordinate_systems(
    markers: TimeSeries, /, *, sex: str = 'M'
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Create local coordinate systems for the body segments.

    Docstring to update.



    References
    ----------
    .. [1] Dumas, R., Chèze, L., Verriest, J.-P., 2007.
       Adjustments to McConville et al. and Young et al. body segment inertial
       parameters. Journal of Biomechanics 40, 543–553.
       https://doi.org/10.1016/j.jbiomech.2006.02.013

    .. [2] Reed, M., Manary, M.A., Schneider, L.W., 1999.
       Methods for Measuring and Representing Automobile Occupant Posture.
       Presented at the International Congress &  Exposition, pp. 1999-01–0959.
       https://doi.org/10.4271/1999-01-0959

    .. [3] Reynolds, H.M., Snow, C.C., Young, J.W., 1982.
       Spatial Geometry of the Human Pelvis.
       FEDERAL AVIATION ADMINISTRATION OKLAHOMA CITY OK CIVIL AEROMEDICAL INST.

    """
    definitions = {}

    # --- Pelvis segment ---
    definitions['Pelvis'] = _define_pelvis(markers, sex)

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
    segments = {
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
            'Color': [1, 0.5, 0.5],
        },
        'Trunk': {
            'Links': [
                ['L5S1', 'C7T1'],

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
            'Color': [1, 0.5, 0.5],
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
            'Color': [1, 0, 0.5],
        },
        'Forearms': {
            'Links': [
                ['WristJointCenterR', 'ElbowJointCenterR'],
                ['RadialStyloidR', 'LateralHumeralEpicondyleR'],
                ['UlnarStyloidR', 'MedialHumeralEpicondyleR'],
                ['RadialStyloidR', 'UlnarStyloidR'],

                ['WristJointCenterL', 'ElbowJointCenterL'],
                ['RadialStyloidL', 'LateralHumeralEpicondyleL'],
                ['UlnarStyloidL', 'MedialHumeralEpicondyleL'],
                ['RadialStyloidL', 'UlnarStyloidL'],
            ],
            'Color': [1, 0.5, 0],
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
            'Color': [0.5, 0, 1],
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
            'Color': [0, 0.5, 1],
        },

    }
    bodies = kinematics.track_rigid_body(
        markers, definitions['Pelvis'], 'Pelvis')

    Player(markers, bodies, segments=segments)

    return definitions


# Load Inertial Values
INERTIAL_VALUES = {}

# Dumas2007
_ = pd.read_csv(config.root_folder + '/data/anthropometrics/dumas_2007.csv')
_[['RelIXY', 'RelIXZ', 'RelIYZ']] = (
    _[['RelIXY', 'RelIXZ', 'RelIYZ']].applymap(lambda s: np.complex(s))
)
INERTIAL_VALUES['Dumas2007'] = _


module_locals = locals()


def __dir__():
    return directory(module_locals)


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
