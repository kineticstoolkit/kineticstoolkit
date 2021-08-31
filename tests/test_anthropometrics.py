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

"""Unit tests for Kinetics Toolkit's antropometrics modules."""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2021 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import kineticstoolkit as ktk


def test_define_coordinate_systems():
    """Test the dumas2007 function."""
    # Load a sample file
    kinematics = ktk.load(
        ktk.config.root_folder
        + '/data/anthropometrics/static_all_markers.ktk.zip'
    )
    markers = kinematics['kinematics']['Markers'].copy()

    rename_dict = {
        'AnteriorIliacCrestR': 'AnteriorSuperiorIliacSpineR',
        'AnteriorIliacCrestL': 'AnteriorSuperiorIliacSpineL',
        'Symphysis': 'PubicSymphysis',
        'Body_C7': 'C7',
        'Body_JugularNotch': 'Suprasternale',
        'Body_AcromionR': 'AcromionR',
        'Body_AcromionL': 'AcromionL',
        'Body_LateralEpicondyleR': 'LateralHumeralEpicondyleR',
        'Body_LateralEpicondyleL': 'LateralHumeralEpicondyleL',
        'MedialEpicondyleR': 'MedialHumeralEpicondyleR',
        'MedialEpicondyleL': 'MedialHumeralEpicondyleL',
        'Body_RadialStyloidR': 'RadialStyloidR',
        'Body_RadialStyloidL': 'RadialStyloidL',
        'Body_UlnarStyloidR': 'UlnarStyloidR',
        'Body_UlnarStyloidL': 'UlnarStyloidL',
        'Body_FemoralLateralEpicondyleR': 'LateralFemoralEpicondyleR',
        'Body_FemoralLateralEpicondyleL': 'LateralFemoralEpicondyleL',
        'Body_FemoralMedialEpicondyleR': 'MedialFemoralEpicondyleR',
        'Body_FemoralMedialEpicondyleL': 'MedialFemoralEpicondyleL',
        'Body_LateralMalleolusR': 'LateralMalleolusR',
        'Body_LateralMalleolusL': 'LateralMalleolusL',
        'Body_MedialMalleolusR': 'MedialMalleolusR',
        'Body_MedialMalleolusL': 'MedialMalleolusL',
        'Body_Meta2R': 'CarpalMetaHead2R',
        'Body_Meta5R': 'CarpalMetaHead5R',
        'Body_Meta2L': 'CarpalMetaHead2L',
        'Body_Meta5L': 'CarpalMetaHead5L',
        'Body_HeadVertex': 'HeadVertex',
        'Body_InterEyes': 'Sellion',
    }
    for old_name in rename_dict:
        markers.rename_data(old_name, rename_dict[old_name], in_place=True)

    # Create fake posterior iliac spine markers
    markers.data['PosteriorSuperiorIliacSpineR'] = markers.data[
        'AnteriorSuperiorIliacSpineR'
    ] + [[0.1, -0.03, 0, 0]]
    markers.data['PosteriorSuperiorIliacSpineL'] = markers.data[
        'AnteriorSuperiorIliacSpineL'
    ] + [[0.1, -0.03, 0, 0]]

    # Infer missing makers
    for segment in ['Pelvis', 'Thorax', 'Extremities']:
        markers.merge(
            ktk.anthropometrics.infer_joint_centers(
                markers, segment=segment, sex='M'
            ),
            in_place=True,
        )

    # Generate segment clusters
    clusters = {}
    clusters['Pelvis'] = ktk.kinematics.create_cluster(
        markers, [
            'MWC_Marker1',
            'MWC_Marker2',
            'MWC_Marker3',
            'MWC_Marker4',
            'AnteriorSuperiorIliacSpineR',
            'AnteriorSuperiorIliacSpineL',
            'PosteriorSuperiorIliacSpineR',
            'PosteriorSuperiorIliacSpineL',
            'PubicSymphysis',
            'L5S1JointCenter',
            'HipJointCenterR',
            'HipJointCenterL',
        ]
    )

    clusters['Trunk'] = ktk.kinematics.create_cluster(
        markers, [
            'C7',
            'Body_T8',
            'Xiphoid',
            'Suprasternale',
            'C7T1JointCenter',
        ]
    )

    clusters['HeadNeck'] = ktk.kinematics.create_cluster(
        markers, [
            'Sellion',
            'HeadVertex',
            'Body_Chin',
        ]
    )

    for side in ['R', 'L']:

        clusters[f'Arm{side}'] = ktk.kinematics.create_cluster(
            markers, [
                f'Arm{side}_Marker1',
                f'Arm{side}_Marker2',
                f'Arm{side}_Marker3',
                f'Arm{side}_Marker4',
                f'GlenohumeralJointCenter{side}',
                f'LateralHumeralEpicondyle{side}',
                f'MedialHumeralEpicondyle{side}',
                f'ElbowJointCenter{side}',
            ]
        )

        clusters[f'Forearm{side}'] = ktk.kinematics.create_cluster(
            markers, [
                f'ElbowJointCenter{side}',
                f'UlnarStyloid{side}',
                f'RadialStyloid{side}',
                f'WristJointCenter{side}',
            ]
        )

        clusters[f'Hand{side}'] = ktk.kinematics.create_cluster(
            markers, [
                f'WristJointCenter{side}',
                f'CarpalMetaHead2{side}',
                f'CarpalMetaHead5{side}',
            ]
        )

        clusters[f'Thigh{side}'] = ktk.kinematics.create_cluster(
            markers, [
                f'HipJointCenter{side}',  # Normally I would remove this one.
                f'LateralFemoralEpicondyle{side}',
                f'MedialFemoralEpicondyle{side}',
                f'KneeJointCenter{side}',
            ]
        )

        clusters[f'Leg{side}'] = ktk.kinematics.create_cluster(
            markers, [
                f'Leg{side}_Marker1',
                f'Leg{side}_Marker2',
                f'Leg{side}_Marker3',
                f'Leg{side}_Marker4',
                f'LateralMalleolus{side}',
                f'MedialMalleolus{side}',
                f'KneeJointCenter{side}',
                f'AnkleJointCenter{side}',
            ]
        )

    # Test if everything can be built again using these clusters.
    test_markers = kinematics['kinematics']['Markers'].copy()
    for old_name in rename_dict:
        test_markers.rename_data(
            old_name, rename_dict[old_name], in_place=True)

    for segment in clusters:
        test_markers.merge(
            ktk.kinematics.track_cluster(
                test_markers,
                clusters[segment]
            ),
            in_place=True,
        )

    # Create segment trajectories
    bodies = markers.copy(copy_data=False, copy_data_info=False)
    for segment in [
        'Pelvis',
        'Thorax',
        'HeadNeck',
        'ArmR',
        'ArmL',
        'ForearmR',
        'ForearmL',
        'HandR',
        'HandL',
        'ThighR',
        'ThighL',
        'LegR',
        'LegL'
    ]:
        bodies.merge(
            ktk.anthropometrics.track_local_coordinate_systems(
                markers, segment),
            in_place=True,
        )

    ktk.Player(test_markers, bodies, segments=ktk.anthropometrics.LINKS)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
