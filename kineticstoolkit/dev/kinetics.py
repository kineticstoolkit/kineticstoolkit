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


def create_forceplatform_lcs(
    ar_corner: ArrayLike,
    pr_corner: ArrayLike,
    pl_corner: ArrayLike,
    al_corner: ArrayLike,
) -> np.ndarray:
    """
    Create a local coordinate system for a force platform.

    Creates a series of transforms (Nx4x4) that define the orientation and
    position of a force platform in space, based on the position of the
    corners, following:

        - origin at the geometrical centre of the platform, at ground level,
          and not the force sensor's origin.
        - x pointing anteriorly.
        - y pointing right.
        - z pointing down.

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

    """
    ar_corner = geometry.create_point_series(ar_corner)
    n_samples = ar_corner.shape[0]
    pr_corner = geometry.create_point_series(pr_corner, length=n_samples)
    pl_corner = geometry.create_point_series(pl_corner, length=n_samples)
    al_corner = geometry.create_point_series(al_corner, length=n_samples)

    lcs = geometry.create_transform_series(
        positions=0.25 * (ar_corner + pr_corner + pl_corner + al_corner),
        x=0.5 * (ar_corner + al_corner) - 0.5 * (pr_corner + pl_corner),
        xy=0.5 * (ar_corner + pr_corner) - 0.5 * (pl_corner + al_corner),
    )

    return lcs


def calculate_cop(
    local_force: ArrayLike,
    local_moment: ArrayLike,
    force_threshold: float = 10,
) -> np.ndarray:
    """
    Calculate centre of pressure on a force plate in local coordinates.

    This calculation is based on the assumption that Mx and My are always zero
    in gait. All coordinates are expressed in local force platform coordinates,
    which are:

        - origin at the geometrical centre of the platform, at ground level,
          and not the force sensor's origin.
        - x and y pointing horizontally.
        - z pointing down.

    Parameters
    ----------
    local_force
        Nx4 force series [[Fx, Fy, Fz, 0.0], ...] expressed in local force
        platform coordinates.
    local_moments
        Nx4 moment series [[Mx, My, Mz, 0.0], ...] expressed in local force
        platform coordinates.
    force_threshold
        Minimal vertical force required to calculate the CoP. NaNs are
        returned for any vertical force under this value. Default is 5.

    Returns
    -------
    np.ndarray
        Nx4 array of centre of pressure [[x, y, z, 1.0], ...]

    """
    local_force = np.array(local_force)
    local_moment = np.array(local_moment)
    check_param("force_threshold", force_threshold, float)

    non_nan = np.abs(local_force[:, 2]) >= force_threshold

    cop = np.zeros(local_force.shape)
    cop[:, :] = np.nan
    cop[non_nan, 0] = (-local_moment[non_nan, 1]) / local_force[non_nan, 2]
    cop[non_nan, 1] = (local_moment[non_nan, 0]) / local_force[non_nan, 2]
    cop[non_nan, 2] = 0
    cop[:, 3] = 1.0

    return cop
