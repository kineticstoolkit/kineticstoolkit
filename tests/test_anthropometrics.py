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
    markers = kinematics['kinematics']['Markers']

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
    markers.merge(
        ktk.anthropometrics.infer_pelvis_joint_centers(markers, 'M'),
        in_place=True,
    )
    markers.merge(
        ktk.anthropometrics.infer_thorax_joint_centers(markers, 'M'),
        in_place=True,
    )
    markers.merge(
        ktk.anthropometrics.infer_extremity_joint_centers(markers),
        in_place=True,
    )

    # Generate the local coordinate systems
    rigid_body_definitions = {}
    rigid_body_definitions['Pelvis'] = \
        ktk.anthropometrics.define_pelvis_coordinate_system(markers)
    rigid_body_definitions['Thorax'] = \
        ktk.anthropometrics.define_thorax_coordinate_system(markers)
    rigid_body_definitions['HeadNeck'] = \
        ktk.anthropometrics.define_head_neck_coordinate_system(markers)
    rigid_body_definitions['ArmR'] = \
        ktk.anthropometrics.define_arm_coordinate_system(markers, side='R')
    rigid_body_definitions['ArmL'] = \
        ktk.anthropometrics.define_arm_coordinate_system(markers, side='L')
    rigid_body_definitions['ForearmR'] = \
        ktk.anthropometrics.define_forearm_coordinate_system(markers, side='R')
    rigid_body_definitions['ForearmL'] = \
        ktk.anthropometrics.define_forearm_coordinate_system(markers, side='L')
    rigid_body_definitions['HandR'] = \
        ktk.anthropometrics.define_hand_coordinate_system(markers, side='R')
    rigid_body_definitions['HandL'] = \
        ktk.anthropometrics.define_hand_coordinate_system(markers, side='L')
    rigid_body_definitions['ThighR'] = \
        ktk.anthropometrics.define_thigh_coordinate_system(markers, side='R')
    rigid_body_definitions['ThighL'] = \
        ktk.anthropometrics.define_thigh_coordinate_system(markers, side='L')
    rigid_body_definitions['LegR'] = \
        ktk.anthropometrics.define_leg_coordinate_system(markers, side='R')
    rigid_body_definitions['LegL'] = \
        ktk.anthropometrics.define_leg_coordinate_system(markers, side='L')

    # Track these rigid bodies
    rigid_bodies = ktk.kinematics.track_rigid_bodies(
        markers, rigid_body_definitions)

    # Print the results
    ktk.Player(markers, rigid_bodies, segments=ktk.anthropometrics.SEGMENTS)

if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
