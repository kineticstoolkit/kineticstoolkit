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

This module is in very early development and everything can still change.
Please don't use this module in production code.

"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2021 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


from kineticstoolkit.decorators import unstable, directory
from kineticstoolkit import TimeSeries, geometry
from typing import Dict
import numpy as np


"""
NOTES WHILE I'M BUILDING IT.

Steps
1. Get xyz pelvis dimensions by creating a temporary local corodinate system.
2.

Should I just reconstruct everything with the mothos, since it's documented
as one article, without deconstructing into many functions?

"""


def define_pelvis_coordinate_system(
    markers: TimeSeries,
    /,
    *,
    sex: str = 'M',
    flesh_adjustment: bool = True,
) -> TimeSeries:
    """
    Adjust markers for flesh and estimate L5S1, hips and COM and local CS.

    The pelvis local coordinate system is located at LS51, with X anterior,
    Y up and Z right.

    Parameters
    ----------
    markers :
        TimeSeries that contains the following markers as Nx4 series:
        - AnteriorSuperiorIliacSpineR
        - AnteriorSuperiorIliacSpineL
        - PosteriorSuperiorIliacSpineR
        - PosteriorSuperiorIliacSpineL
        - PublicSymphysis
    sex : Optional.
        F' for female, 'M' for male. The default is 'M'.
    flesh_adjusment : Optional.
        True to adjust the skin markers to anatomical landmarks, False if the
        markers are already landmarks. The default is
        True.

    Returns
    -------
    TimeSeries
        Contains the following data:
        - AnteriorSuperiorIliacSpineR: Nx4 series.
        - AnteriorSuperiorIliacSpineL: Nx4 series.
        - PosteriorSuperiorIliacSpineR: Nx4 series.
        - PosteriorSuperiorIliacSpineL: Nx4 series.
        - PublicSymphysis: Nx4 series.
        - PelvisCOM: Center of mass of the pelvis, as an Nx4 series
        - PelvisLCS: Local coordinate system of the pelvis, as an Nx4x4 series.

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
        origin=mpsis, z=(rasis - lasis), xz=(0.5 * (rasis - lasis) - mpsis)
    )

    # Calculate the markers in the local coordinate system
    local_rasis = geometry.get_local_coordinates(rasis, temp_LCS)
    local_lasis = geometry.get_local_coordinates(lasis, temp_LCS)
    local_rpsis = geometry.get_local_coordinates(rpsis, temp_LCS)
    local_lpsis = geometry.get_local_coordinates(lpsis, temp_LCS)
    local_sym = geometry.get_local_coordinates(sym, temp_LCS)

    # Create a rigid body definition using these locations
    pelvis_definition = {
        'AnteriorSuperiorIliacSpineR': np.nanmean(local_rasis, axis=0)[
            np.newaxis
        ],
        'AnteriorSuperiorIliacSpineL': np.nanmean(local_lasis, axis=0)[
            np.newaxis
        ],
        'PosteriorSuperiorIliacSpineR': np.nanmean(local_rpsis, axis=0)[
            np.newaxis
        ],
        'PosteriorSuperiorIliacSpineL': np.nanmean(local_lpsis, axis=0)[
            np.newaxis
        ],
        'PubicSymphysis': np.nanmean(local_sym, axis=0)[np.newaxis],
    }

    # Flesh margin adjustment to calculate pelvis width
    if flesh_adjustment:
        adjusted_local_rasis = local_rasis + np.array([[-10, 0, 0, 0]]) / 1000
        adjusted_local_lasis = local_lasis + np.array([[-10, 0, 0, 0]]) / 1000
        adjusted_local_sym = (
            local_sym + np.array([[-17.7, -17.7, 0, 0]]) / 1000
        )
    else:
        adjusted_local_rasis = local_rasis
        adjusted_local_lasis = local_lasis
        adjusted_local_sym = local_sym

    # Lengths calculation
    pw = np.nanmean(
        adjusted_local_rasis[:, 2] - adjusted_local_lasis[:, 2]
    )  # Pelvis width
    ph = np.nanmean(
        0.5
        * (adjusted_local_rasis[:, 1] + adjusted_local_lasis[:, 1])
        / adjusted_local_sym[:, 1]
    )  # Pelvis height
    pd = np.nanmean(
        0.5 * (adjusted_local_rasis[:, 0] + adjusted_local_lasis[:, 0])
        - 0.5 * (local_rpsis[:, 0] + local_lpsis[:, 0])
    )  # Pelvis depth

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

    # Move the origin to L5S1
    l5s1_offset = pelvis_definition['L5S1'].copy()
    l5s1_offset[:, 3] = 0
    for point in pelvis_definition:
        pelvis_definition[point] -= l5s1_offset

    return pelvis_definition


@unstable
def create_frames(markers: TimeSeries, *, method='dumas2007') -> TimeSeries:
    """
    Create body trajectories based on markers, using the specified method.

    At the moment, this function follows the method described in
    Dumas, R., Chèze, L., Verriest, J.-P., 2007. Adjustments to McConville
    et al. and Young et al. body segment inertial parameters. Journal of
    Biomechanics 40, 543–553. https://doi.org/10.1016/j.jbiomech.2006.02.013

    Parameters
    ----------
    markers: A TimeSeries that contains a mix of the following markers:
        Toe3EndL/R, LateralMalleolusL/R, MedialMalleolusL/R,
        LateralFemoralEpicondyleL/R, MedialFemoralEpicondyleL/R,
        AnteroSuperiorIliacSpineL/R, PosteroSuperiorIliacSpineL/R,
        PubicSymphysis, GlenohumeralJointL/R,
        LateralHumeralEpicondyleL/R, MedialHumeralEpicondyleL/R,
        RadialStyloidL/R, UlnarStyloidL/R,
        HandMeta2L/R, HandMeta5L/R,
        C7, JugularNotch, HeadVertex, Sellion.
    methods: Optional. Currently only 'dumas2007' is implemented.

    Returns
    -------
    TimeSeries
        A TimeSeries that contains these data as Nx4x4 series (if the
        corresponding markers are available):
        Pelvis, Torso, HeadNeck, ArmL, ArmR,
        ForearmL, ForearmR, HandL, HandR,
        ThighL, ThighR, LegL, LegR, FootL, FootR

    """
    pass


module_locals = locals()


def __dir__():
    return directory(module_locals)


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
