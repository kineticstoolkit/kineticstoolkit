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


@unstable
def dumas2007(kinematics: TimeSeries) -> TimeSeries:
    """
    Create COM points and frames using Dumas (2007).

    At the moment, this function follows the method described in
    Dumas, R., Chèze, L., Verriest, J.-P., 2007. Adjustments to McConville
    et al. and Young et al. body segment inertial parameters. Journal of
    Biomechanics 40, 543–553. https://doi.org/10.1016/j.jbiomech.2006.02.013

    Parameters
    ----------
    kinematics:
        A TimeSeries that contains the following markers:
            Toe3EndL/R, LateralMalleolusL/R, MedialMalleolusL/R,
            LateralFemoralEpicondyleL/R, MedialFemoralEpicondyleL/R,
            AnteriorIliacSpineL/R, PosteriorIliacSpineL/R, PubicSymphysis,
            GlenohumeralJointL/R,
            LateralHumeralEpicondyleL/R, MedialHumeralEpicondyleL/R,
            RadialStyloidL/R, UlnarStyloidL/R,
            HandMeta2L/R, HandMeta5L/R,
            C7, JugularNotch, HeadVertex, Sellion.

    Returns
    -------
    A copy of the input TimeSeries, with additional points and frames:
        Centres of mass (Nx4 series):
            ComPelvis, ComTorso, ComHeadNeck, ComArmL, ComArmR,
            ComForearmL, ComForearmR, ComHandL, ComHandR,
            ComThighL, ComThighR, ComLegL, ComLegR, ComFootL, ComFootR
        Frames (Nx4x4 series):
            Pelvis, Torso, HeadNeck, ArmL, ArmR,
            ForearmL, ForearmR, HandL, HandR,
            ThighL, ThighR, LegL, LegR, FootL, FootR

    Notes
    -----
    If one or many required markers are missing, warnings are raised and the
    corresponding segments's anthropometrics are ignored.

    """
    return kinematics
    # --- Pelvis

    # if segment == 'Pelvis':
    #     # First create frames that are correctly oriented but with the origin
    #     # at mid-AnteriorIliacSpines instead of L5S1.
    #     asis_center = 0.5 * (
    #         markers['AnteriorIliacSpineL'] + markers['AnteriorIliacSpineR'])
    #     pis_center = 0.5 * (
    #         markers['PosteriorIliacSpineL'] + markers['AnteriorIliacSpineR'])
    #     aligned_frames = geometry.create_frames(
    #         origin=asis_center,
    #         z=(markers['AnteriorIliacSpineR']
    #             - markers['AnteriorIliacSpineL']),
    #         xz=asis_center - pis_center
    #     )

    #     # Express the pelvis anchor points in now-local coordinates
    #     asis_l = geometry.get_local_coordinates(
    #         markers['AnteriorIliacSpineL'], aligned_frames)
    #     asis_r = geometry.get_local_coordinates(
    #         markers['AnteriorIliacSpineR'], aligned_frames)
    #     pis_l = geometry.get_local_coordinates(
    #         markers['PosteriorIliacSpineL'], aligned_frames)
    #     pis_r = geometry.get_local_coordinates(
    #         markers['PosteriorIliacSpineR'], aligned_frames)
    #     symphysis = geometry.get_local_coordinates(
    #         markers['PubicSymphysis'], aligned_frames)

    #     # Make some measurements based on the data we have
    #     pelvis_width = np.nanmean(asis_r - asis_l)[:, 2]
    #     pelvis_height = np.nanmean(0.5 * (asis_l + asis_r) - symphysis)[:, 1]
    #     pelvis_depth = 0.5 * (
    #         np.nanmean(asis_r - pis_r) + np.nanmean(asis_l - pis_l))


module_locals = locals()


def __dir__():
    return directory(module_locals)


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
