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
from typing import Dict, Union, Sequence
import numpy as np
import pandas as pd
from warnings import warn


@unstable
def infer_joint_centers(
        markers: TimeSeries,
        /,
        segment: str,
        *,
        sex: str = 'M',
) -> TimeSeries:
    """
    Infer joint centers based on anthropometric data.

    This function is aimed to be used on static acquisitions in quasi-neutral
    position.

    For the pelvis, creates L5S1JointCenter, HipJointCenterL and
    HipJointCenterR based on AnteriorSuperiorIliacSpineR/L,
    PosteriorSuperiorIliacSpineR/L and PubicSymphysis, following the method
    presented in Reed et al. (1999).

    For the thorax, creates C7T1JointCenter, GlenohumeralJointCenterL and
    GlenohumeralJointCenterR based on C7, L5S1JointCenter, Suprasternal,
    AcromionR and AcromionL, following the method presented in Reed et al.
    (1999). The scapulo-thoracic joint must be in a neutral
    position.

    For the extremities, creates:

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
        TimeSeries that contain the trajectory of the required markers as
        Nx4 series.
    segment
        Can be either 'Pelvis', 'Thorax', or 'Extremities'. An error is raised
        in case of missing markers.
    sex
        Optional. Can be either 'M' or 'F'. The default is 'M'.

    Returns
    -------
    TimeSeries
        A TimeSeries that contains the inferred joint center trajectories.

    References
    ----------
    - Reed, M., Manary, M.A., Schneider, L.W., 1999. Methods for Measuring and
      Representing Automobile Occupant Posture. Presented at the International
      Congress &  Exposition, pp. 1999-01–0959.
      https://doi.org/10.4271/1999-01-0959

    """
    if segment == 'Pelvis':
        return _infer_pelvis_joint_centers(markers, sex=sex)
    elif segment == 'Thorax':
        return _infer_thorax_joint_centers(markers, sex=sex)
    elif segment == 'Extremities':
        return _infer_extremity_joint_centers(markers)
    elif segment == '':
        out = markers.copy()
        for segment in ['Pelvis', 'Thorax', 'Extremities']:
            try:
                out.merge(
                    infer_joint_centers(
                        out,
                        segment=segment,
                        sex=sex,
                    ),
                    in_place=True,
                )
            except ValueError:
                pass
        return out
    else:
        raise ValueError("Unrecognized segment name.")


def _infer_pelvis_joint_centers(
        markers: TimeSeries,
        sex: str = 'M') -> TimeSeries:
    """Infer L5S1 and hip joint centers based on anthropometric data."""
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

    # Create a cluster using these locations
    cluster = {
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
    tracked_pelvis = kinematics.track_cluster(
        markers, cluster, include_lcs=True, lcs_name='Pelvis')

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


def _infer_thorax_joint_centers(
        markers: TimeSeries,
        sex: str = 'M') -> TimeSeries:
    """Infer C7T1 and glenohumeral joint centers based on anthropom. data."""
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


def _infer_extremity_joint_centers(
        markers: TimeSeries) -> TimeSeries:
    """Infer extremity joint centers based on medial and lateral markers."""
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


@unstable
def track_local_coordinate_systems(
        markers: TimeSeries,
        /,
        segments: Union[str, Sequence[str]],
        *,
        method: int = 1) -> TimeSeries:
    """
    Create local coordinate system definitions based on static markers.

    Parameters
    ----------
    markers
        TimeSeries that contains marker trajectories as Nx4 during an action.

    segments
        Name of the segment. Can be 'Pelvis', 'Thorax', 'HeadNeck',
        'ArmR', 'ArmL', 'ForearmR', 'ForearmL', 'HandR', 'HandL', 'ThighR',
        'ThighL', 'LegR', 'LegL', 'FootR', or 'FootL'. An empty string or
        an empty list will try to process every segment.

    method
        Optional. Select one ot both methods to reconstruct the axial
        orientation of the arm as proposed by the ISB: 1 to use both humeral
        epicondyles, 2 to use the elbow center and the ulnar styloid. The
        default is 1. This parameter is ignored for segments other than
        ArmR and ArmL.

    Returns
    -------
    TimeSeries
        A TimeSeries that contains the segments' local position and orientation
        as series of Nx4x4 frames.

    """
    output = markers.copy(copy_data=False, copy_data_info=False)

    # If we have no segment or an empty list, go with every segment
    if len(segments) == 0:
        segments = [
            'Pelvis', 'Thorax', 'HeadNeck', 'ArmR', 'ArmL', 'ForearmR',
            'ForearmL', 'HandR', 'HandL', 'ThighR', 'ThighL', 'LegR', 'LegL',
            'FootR', 'FootL',
        ]

    # If we have a list of segments
    if not isinstance(segments, str):
        for segment in segments:
            output.merge(
                track_local_coordinate_systems(
                    markers,
                    segment,
                    method=method
                ),
                in_place=True,
            )
        return output

    # If we have a single segment
    segment = segments

    if segment == 'Pelvis':
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

        mpsis = 0.5 * (rpsis + lpsis)
        output.data[segment] = geometry.create_frames(
            origin=l5s1, z=(rasis - lasis),
            xz=(0.5 * (rasis + lasis) - mpsis),
        )

    elif segment == 'Thorax':
        try:
            c7t1 = markers.data['C7T1JointCenter']
            sup = markers.data['Suprasternale']
            l5s1 = markers.data['L5S1JointCenter']
        except KeyError:
            raise ValueError(
                "Not enough markers to create the thorax coordinate system."
            )

        output.data[segment] = geometry.create_frames(
            origin=c7t1, y=(c7t1 - l5s1),
            xy=(sup - l5s1),
        )

    elif segment == 'HeadNeck':
        try:
            c7t1 = markers.data['C7T1JointCenter']
            sel = markers.data['Sellion']
            hv = markers.data['HeadVertex']
        except KeyError:
            raise ValueError(
                "Not enough markers to create the thorax coordinate system."
            )

        output.data[segment] = geometry.create_frames(
            origin=c7t1, y=(hv - c7t1),
            xy=(sel - c7t1),
        )

    elif segment in ['ArmR', 'ArmL']:
        side = segment[-1]
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

        output.data[segment] = geometry.create_frames(
            origin=gh, y=(gh - elbow_center),
            yz=(r_ep - l_ep),
        )

    elif segment in ['ForearmR', 'ForearmL']:
        side = segment[-1]
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

        output.data[segment] = geometry.create_frames(
            origin=elbow_center, y=(elbow_center - wrist_center),
            yz=(r_st - l_st),
        )

    elif segment in ['HandR', 'HandL']:
        side = segment[-1]
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

        meta_center = 0.5 * (r_meta + l_meta)
        output.data[segment] = geometry.create_frames(
            origin=wrist_center, y=(wrist_center - meta_center),
            yz=(r_meta - l_meta),
        )

    elif segment in ['ThighR', 'ThighL']:
        side = segment[-1]
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

        output.data[segment] = geometry.create_frames(
            origin=hip_center, y=(hip_center - knee_center),
            yz=(r_ep - l_ep),
        )

    elif segment in ['LegR', 'LegL']:
        side = segment[-1]
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

        output.data[segment] = geometry.create_frames(
            origin=knee_center, y=(knee_center - ankle_center),
            yz=(r_mal - l_mal),
        )

    elif segment in ['FootR', 'FootL']:
        side = segment[-1]
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
        output.data[segment] = geometry.create_frames(
            origin=ankle_center, x=(meta_center - calc),
            xz=(r_meta - l_meta),
        )

    return output


@unstable
def estimate_center_of_mass(
        markers: TimeSeries,
        /,
        segments: Union[str, Sequence[str]],
        *,
        sex: str = 'M') -> TimeSeries:
    """
    Estimate the segments' center of mass based on anthropometric data.

    Parameters
    ----------
    markers
        TimeSeries that contains marker trajectories as Nx4 during an action.

    segments
        Name of the segment. Can be 'Pelvis', 'Thorax', 'HeadNeck',
        'ArmR', 'ArmL', 'ForearmR', 'ForearmL', 'HandR', 'HandL', 'ThighR',
        'ThighL', 'LegR', 'LegL', 'FootR', or 'FootL'. An empty string or
        an empty list will try to process every segment.

    sex
        Optional. Either 'M' or 'F'. The default is 'M'.

    Returns
    -------
    TimeSeries
        A TimeSeries with the trajectory of the segments' centers of mass,
        named {segment}CenterOfMass.

    """
    markers = markers.copy()  # We will add markers to it so copy it first.
    output = markers.copy(copy_data=False, copy_data_info=False)

    # If we have no segment or an empty list, go with every segment
    if len(segments) == 0:
        segments = [
            'Pelvis', 'Thorax', 'HeadNeck', 'ArmR', 'ArmL', 'ForearmR',
            'ForearmL', 'HandR', 'HandL', 'ThighR', 'ThighL', 'LegR', 'LegL',
            'FootR', 'FootL',
        ]

    # If we have a list of segments
    if not isinstance(segments, str):
        for segment in segments:
            output.merge(
                estimate_center_of_mass(markers, segment),
                in_place=True,
            )
        return output

    # From here we have a single segment
    segment = segments

    # Decompose segment name and side
    if segment not in ['Pelvis', 'Thorax', 'HeadNeck']:
        side = segment[-1]
        segment = segment[0:-1]
    else:
        side = ''

    # Calculate the local coordinate system for this segment
    lcs = track_local_coordinate_systems(markers, segments=(segment + side))

    df = INERTIAL_VALUES['Dumas2007']
    # Search the inertial value tables for the given segment and sex
    _ = df.loc[(df['Segment'] == segment) & (df['Gender'] == sex)]
    constants = _.to_dict('records')[0]

    # Add possible missing markers
    if 'ProjectedHipJointCenter' in [
        constants['LengthPoint1'],
        constants['LengthPoint2'],
    ]:
        # Calculate ProjectedHipJointCenter
        local_rhip = geometry.get_local_coordinates(
            markers.data['HipJointCenterR'],
            lcs.data['Pelvis'])
        local_rhip[:, 2] = 0  # Projection in sagittal plane
        local_lhip = geometry.get_local_coordinates(
            markers.data['HipJointCenterL'],
            lcs.data['Pelvis'])
        local_lhip[:, 2] = 0  # Projection in sagittal plane
        local_hips = 0.5 * (local_rhip + local_lhip)

        markers.data['ProjectedHipJointCenter'] = \
            geometry.get_global_coordinates(local_hips, lcs.data['Pelvis'])

    if 'CarpalMetaHeadM25' in [
        constants['LengthPoint1'],
        constants['LengthPoint2'],
    ]:
        # Calculate midpoint of carpal meat head 2 and 5
        markers.data[f'CarpalMetaHeadM25{side}'] = 0.5 * (
            markers.data[f'CarpalMetaHead2{side}']
            + markers.data[f'CarpalMetaHead5{side}']
        )

    if 'TarsalMetaHeadM15' in [
        constants['LengthPoint1'],
        constants['LengthPoint2'],
    ]:
        # Calculate midpoint of carpal meat head 2 and 5
        markers.data[f'TarsalMetaHeadM15{side}'] = 0.5 * (
            markers.data[f'TarsalMetaHead1{side}']
            + markers.data[f'TarsalMetaHead5{side}']
        )

    # Calculate the segment length
    segment_length = np.sqrt(
        np.sum(
            np.nanmean(
                markers.data[constants['LengthPoint2'] + side]
                - markers.data[constants['LengthPoint1'] + side],
                axis=0,
            ) ** 2
        )
    )

    # Add COM output
    output.data[f'{segment}{side}CenterOfMass'] = \
        geometry.get_global_coordinates(
            np.array([[
                segment_length * constants['RelComX'],
                segment_length * constants['RelComY'],
                segment_length * constants['RelComZ'],
                1.0,
            ]]),
            lcs.data[segment + side],
    )

    return output


@unstable
def estimate_global_center_of_mass(
        coms: TimeSeries,
        /,
        sex: str = 'M'
) -> TimeSeries:
    """
    Estimate the global center of mass.

    Parameters
    ----------
    coms
        A TimeSeries with the trajectory of every segment's center of mass.
        The segments must be in this list: 'PelvisCenterOfMass',
        'ThoraxCenterOfMass', 'HeadNeckCenterOfMass',
        'ArmRCenterOfMass', 'ArmLCenterOfMass',
        'ForearmRCenterOfMass', 'ForearmLCenterOfMass',
        'HandRCenterOfMass', 'HandLCenterOfMass',
        'ThighRCenterOfMass', 'ThighLCenterOfMass',
        'LegRCenterOfMass', 'LegLCenterOfMass',
        'FootRCenterOfMass', 'FootLCenterOfMass'.
    sex
        Optional. Either 'M' or 'F'. The default is 'M'.

    Returns
    -------
    ktk.TimeSeries
        A TimeSeries with a single element named 'GlobalCenterOfMass'.

    """
    inertial_data = INERTIAL_VALUES['Dumas2007']

    out = coms.copy(copy_data=False, copy_data_info=False)
    out.data['GlobalCenterOfMass'] = np.zeros([out.time.shape[0], 4])

    cumulative_mass = 0.0

    for data in coms.data:
        segment_name = data.replace('CenterOfMass', '')
        if segment_name not in ['Thorax', 'HeadNeck', 'Pelvis']:
            # The segment name is terminated by L or R. Remove it.
            segment_name = segment_name[0:-1]

        segment_rel_mass = inertial_data.loc[
            (inertial_data['Segment'] == segment_name)
            & (inertial_data['Gender'] == sex),
            'RelMass'
        ]

        if len(segment_rel_mass) == 1:

            out.data['GlobalCenterOfMass'] += (
                float(segment_rel_mass) * coms.data[data]
            )

            cumulative_mass += float(segment_rel_mass)

    out.data['GlobalCenterOfMass'] /= cumulative_mass
    out.data['GlobalCenterOfMass'][:, 3] = 1

    return out


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
        'Color': [0.5, 0, 0.25],
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
    'Feets': {
        'Links': [
            ['CalcaneusR', 'TarsalMetaHead1R'],
            ['CalcaneusR', 'TarsalMetaHead5R'],
            ['MedialMalleolusR', 'TarsalMetaHead1R'],
            ['LateralMalleolusR', 'TarsalMetaHead5R'],
            ['CalcaneusR', 'MedialMalleolusR'],
            ['CalcaneusR', 'LateralMalleolusR'],
            ['TarsalMetaHead1R', 'TarsalMetaHead5R'],

            ['CalcaneusL', 'TarsalMetaHead1L'],
            ['CalcaneusL', 'TarsalMetaHead5L'],
            ['MedialMalleolusL', 'TarsalMetaHead1L'],
            ['LateralMalleolusL', 'TarsalMetaHead5L'],
            ['CalcaneusL', 'MedialMalleolusL'],
            ['CalcaneusL', 'LateralMalleolusL'],
            ['TarsalMetaHead1L', 'TarsalMetaHead5L'],
        ],
        'Color': [0.25, 0.0, 0.75],
    },
}


# Load Inertial Values
INERTIAL_VALUES = {}

# Dumas2007
_ = pd.read_csv(config.root_folder + '/data/anthropometrics_dumas_2007.csv')
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
