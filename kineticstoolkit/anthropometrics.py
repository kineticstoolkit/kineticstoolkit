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

"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2021 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


from kineticstoolkit.decorators import unstable, directory
from kineticstoolkit import TimeSeries, geometry, config
from typing import Dict
import numpy as np
import pandas as pd


"""
NOTES WHILE I'M BUILDING IT.

Steps
1. Get xyz pelvis dimensions by creating a temporary local corodinate system.
2.

Should I just reconstruct everything with the mothos, since it's documented
as one article, without deconstructing into many functions?

"""


def define_pelvis(
        markers: TimeSeries, /, *, sex: str = 'M') -> Dict[str, np.ndarray]:
    """
    Create a local coordinate system for the Pelvis.

    The pelvis local coordinate system is located at LS51, with X anterior,
    Y up and Z right. The segment definition is based on
    Dumas et al. (2007) [1]_, which is based on Reed et al. (1999) [2]_.

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
            "Not enough markers to reconstruct the pelvis segment."
        )

    # Create a temporary, well-aligned local coordinate system
    mpsis = 0.5 * (rpsis + lpsis)
    temp_LCS = geometry.create_frames(
        origin=mpsis, z=(rasis - lasis),
        xz=(0.5 * (rasis - lasis) - mpsis),
    )

    # Calculate the markers in the local coordinate system
    local_rasis = geometry.get_local_coordinates(rasis, temp_LCS)
    local_lasis = geometry.get_local_coordinates(lasis, temp_LCS)
    local_rpsis = geometry.get_local_coordinates(rpsis, temp_LCS)
    local_lpsis = geometry.get_local_coordinates(lpsis, temp_LCS)
    local_sym = geometry.get_local_coordinates(sym, temp_LCS)

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

    # Pelvis width
    pw = np.nanmean(
        local_rasis[:, 2] - local_lasis[:, 2]
    )

    # Create L5S1 and hip joint definitions
    if sex == 'F':
        pelvis_definition['L5S1'] = np.array(
            [[0.289 * pw, 0.172 * pw, 0.0, 1.0]]
        )
        pelvis_definition['HipJointCenterR'] = np.array(
            [[0.197 * pw, 0.270 * pw, 0.372 * pw, 1.0]]
        )
        pelvis_definition['HipJointCenterL'] = np.array(
            [[0.197 * pw, 0.270 * pw, -0.372 * pw, 1.0]]
        )
    elif sex == 'M':
        pelvis_definition['L5S1'] = np.array(
            [[0.264 * pw, 0.126 * pw, 0.0, 1.0]]
        )
        pelvis_definition['HipJointCenterR'] = np.array(
            [[0.208 * pw, 0.278 * pw, 0.361 * pw, 1.0]]
        )
        pelvis_definition['HipJointCenterL'] = np.array(
            [[0.208 * pw, 0.278 * pw, -0.361 * pw, 1.0]]
        )
    else:
        raise ValueError("sex must be either 'F' or 'M'")

    # Move the origin to L5S1
    l5s1_offset = pelvis_definition['L5S1'].copy()
    l5s1_offset[:, 3] = 0
    for point in pelvis_definition:
        pelvis_definition[point] -= l5s1_offset

    # Create COM, from Dumas 2007, Table 2.
    length = np.sqrt(
        (pelvis_definition['HipJointCenterR'][0, 0]  # x
         - pelvis_definition['L5S1'][0, 0]) ** 2
        +
        (pelvis_definition['HipJointCenterR'][0, 1]  # y
         - pelvis_definition['L5S1'][0, 1]) ** 2
    )

    if sex == 'F':
        pelvis_definition['PelvisCOM'] = (
            0.01 * np.array([[-0.9, -23.2, 0.2, 0.0]]) * length
        )
    else:  # 'M'
        pelvis_definition['PelvisCOM'] = (
            0.01 * np.array([[2.8, -28.0, -0.6, 0.0]]) * length
        )
    pelvis_definition['PelvisCOM'][0, 3] = 1.0

    return pelvis_definition



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
