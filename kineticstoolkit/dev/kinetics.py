#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2024-2025 Félix Chénier

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
Provide experimental functions to manage kinetic data.

This module provides experimental functions for a future
kineticstoolkit.kinetics module, which will facilitate the configuration
of force platforms and the computation of the centre of pressure.

"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2024-2025 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import numpy as np
from kineticstoolkit.typing_ import check_param, ArrayLike
import kineticstoolkit.geometry as geometry


def create_forceplatform_frames(
    ar_corner: ArrayLike,
    pr_corner: ArrayLike,
    pl_corner: ArrayLike,
    al_corner: ArrayLike,
    offset: ArrayLike,
) -> np.ndarray:
    """
    Set the frames (rotations, positions) of a force platform.

    Creates a series of frames (Nx4x4) that defines the orientation and
    position of a force platform in space. The frames are defined in
    platform coordinates, with x anterior, y right and z down.

    Parameters
    ----------
    ar_corner
        Coordinates of the anterior-right corner as an Nx4 array. Use double-
        brackets for constants: [[x, y, z, 1.0]]
    pr_corner
        Coordinates of the posterior-right corner as an Nx4 array. Use double-
        brackets for constants: [[x, y, z, 1.0]]
    pl_corner
        Coordinates of the posterior-left corner as an Nx4 array. Use double-
        brackets for constants: [[x, y, z, 1.0]]
    al_corner
        Coordinates of the anterior-left corner as an Nx4 array. Use double-
        brackets for constants: [[x, y, z, 1.0]]
    offset
        Position of the sensor relative to the centre of the platform surface,
        in platform coordinates. Normally, this is [[0.0, 0.0, z, 1.0]] with z
        being positive since the sensor is normally below (+z) the surface of
        the force plate.

    """
    ar_corner = geometry.create_point_series(ar_corner)
    n_samples = ar_corner.shape[0]
    pr_corner = geometry.create_point_series(pr_corner, length=n_samples)
    pl_corner = geometry.create_point_series(pl_corner, length=n_samples)
    al_corner = geometry.create_point_series(al_corner, length=n_samples)
    offset = geometry.create_point_series(offset, length=n_samples)

    # Temporary origin at center of corners
    lcs = geometry.create_transform_series(
        positions=0.25 * (ar_corner + pr_corner + pl_corner + al_corner),
        x=0.5 * (ar_corner + al_corner) - 0.5 * (pr_corner + pl_corner),
        xy=0.5 * (ar_corner + pr_corner) - 0.5 * (pl_corner + al_corner),
    )

    # Set real position
    lcs[:, 0:4, 3] = geometry.get_global_coordinates(offset, lcs)

    return lcs


def calculate_cop(
    forces: ArrayLike, moments: ArrayLike, z: float, force_treshold: float = 10
) -> np.ndarray:
    """
    Calculate center of pressure on a force plate.

    This calculation is based on the assumption that Mx and My are both zero
    and that z is the vertical axis.

    Parameters
    ----------
    forces
        Nx4 array of forces [[Fx, Fy, Fz, 0.0], ...] expressed in local
        force platform coordinates.
    moments
        Nx4 array of moments [[Mx, My, Mz, 0.0], ...] expressed in local
        force platform coordinates.
    z
        Vertical distance between the top of the platform and the sensor.
        Positive if the sensor is below the top of the platform (most cases).

    Returns
    -------
    np.ndarray
        Nx4 array of center of pressure [[x, y, z, 1.0], ...]

    """
    check_param("z", z, float)
    forces = np.array(forces)
    moments = np.array(moments)

    non_nan = np.abs(forces[:, 2]) >= force_treshold

    cop = np.zeros(forces.shape)
    cop[:, :] = np.nan
    cop[non_nan, 0] = (forces[non_nan, 0] * z - moments[non_nan, 1]) / forces[
        non_nan, 2
    ]
    cop[non_nan, 1] = (moments[non_nan, 0] + forces[non_nan, 1] * z) / forces[
        non_nan, 2
    ]
    cop[non_nan, 2] = -z
    cop[:, 3] = 1.0

    return cop
