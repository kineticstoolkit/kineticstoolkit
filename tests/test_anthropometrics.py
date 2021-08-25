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


def define_coordinate_systems():
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

    # Generate the pelvis coordinate system
    pelvis_definition = ktk.anthropometrics.define_coordinate_systems(markers)

if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
