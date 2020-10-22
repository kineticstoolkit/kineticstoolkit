#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020 Félix Chénier

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
Provide 3d geometry and linear algebra functions related to biomechanics.

Note
----
As a convention, the first dimension of every array is always N and corresponds
to time. For constants, use a length of 1 as the first dimension.

"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import numpy as np
import kineticstoolkit.external.icp as icp
from kineticstoolkit.decorators import unstable
from typing import Tuple, List


listing = []  # type: List[str]


@unstable(listing)
def matmul(op1: np.ndarray, op2: np.ndarray) -> np.ndarray:
    """
    Matrix multiplication between series of matrices.

    This function is a wrapper for numpy's matmul function (operator @), that
    helps numpy to understand Kinetics Toolkit's convention that the first
    dimension always corresponds to time.

    It aligns and create additionnal dimensions if needed to avoid dimension
    mismatch errors.

    Parameters
    ----------
    op1
        Series of floats, vectors or matrices.
    op2
        Series of floats, vectors or matrices.

    Returns
    -------
    np.ndarray
        The product, as a series of Nx4 or Nx4xM matrices.

    """
    def perform_mul(op1, op2):
        if isinstance(op1, np.ndarray) and isinstance(op2, np.ndarray):
            return op1 @ op2
        else:
            return op1 * op2  # In the case where we have a series of floats.

    (op1, op2) = match_size(op1, op2)

    n_samples = op1.shape[0]

    # Get the expected shape by performing the first multiplication
    temp = perform_mul(op1[0], op2[0])
    result = np.empty((n_samples, *temp.shape))

    # Perform the multiplication
    for i_sample in range(n_samples):
        result[i_sample] = perform_mul(op1[i_sample], op2[i_sample])

    return result


@unstable(listing)
def create_rotation_matrices(axis: str, angles: np.ndarray) -> np.ndarray:
    """
    Create a Nx4x4 series of rotation matrices around a given axis.

    Parameters
    ----------
    axis
        Can be either 'x', 'y' or 'z'.

    angles
        Series of angles in radians.

    Returns
    -------
    np.ndarray
        A Nx4x4 series of rotation transformation matrices.

    """
    angles = np.array(angles)

    T = np.zeros((angles.shape[0], 4, 4))

    if axis == 'x':
        for i in range(angles.size):
            T[i, 1, 1] = np.cos(angles[i])
            T[i, 1, 2] = np.sin(-angles[i])
            T[i, 2, 1] = np.sin(angles[i])
            T[i, 2, 2] = np.cos(angles[i])
            T[i, 0, 0] = 1.0
            T[i, 3, 3] = 1.0

    elif axis == 'y':
        for i in range(angles.size):
            T[i, 0, 0] = np.cos(angles[i])
            T[i, 2, 0] = np.sin(-angles[i])
            T[i, 0, 2] = np.sin(angles[i])
            T[i, 2, 2] = np.cos(angles[i])
            T[i, 1, 1] = 1.0
            T[i, 3, 3] = 1.0

    elif axis == 'z':
        for i in range(angles.size):
            T[i, 0, 0] = np.cos(angles[i])
            T[i, 0, 1] = np.sin(-angles[i])
            T[i, 1, 0] = np.sin(angles[i])
            T[i, 1, 1] = np.cos(angles[i])
            T[i, 2, 2] = 1.0
            T[i, 3, 3] = 1.0

    else:
        raise ValueError("axis must be either 'x', 'y' or 'z'")

    return T


@unstable(listing)
def create_reference_frames(global_points: np.ndarray,
                            method: str = 'centroid') -> np.ndarray:
    """
    Create a Nx4x4 series of reference frames based on a point cloud series.

    Create reference frames based on global points, using the provided
    method, and returns this series of reference frames as a series of
    transformation matrices.

    Parameters
    ----------
    global_points
        A series of N sets of M points in a global reference frame (Nx4xM).

    method
        Optional. The method to use to construct a reference frame based on
        these points.

        Available values are:

        - 'centroid' : An arbitraty system that is useful for marker clusters.
            - Origin = Centroid of all points;
            - X is directed toward the first point;
            - Z is normal to the plane formed by the origin and both
              first points;
            - Y is the vectorial product of X and Z.

        - 'yz' : Reference frame based on vertical and lateral vectors.
            - Origin = First point;
            - Y is directed toward the second point (up);
            - Z is normal to the plane formed by the three markers (right);
            - X is the vectorial product of Y and Z.

        - 'zy' : Reference frame based on vertical and lateral vectors.
            - Origin = First point;
            - Z is directed toward the second point (right);
            - Y is normal to the plane formed by the three markers (up);
            - X is the vectorial product of Y and Z.

        - 'isb-humerus' : Reference frame of the humerus as suggested by
          the International Society of Biomechanics [1], when the elbow is
          flexed by 90 degrees and the forearm is in complete pronation.

            - Origin = GH joint;
            - X is directed forward;
            - Y is directed upward;
            - Z is directed to the right;
            - Point 1 = GH joint;
            - Point 2 = Elbow center;
            - Point 3 = Ulnar styloid.

        - 'isb-forarm' : Reference frame of the forearm as suggested by
          the International Society of Biomechanics [1].

            - Origin = Ulnar Styloid;
            - X is directed forward in anatomic position;
            - Y is directed upward in anatomic position;
            - Z is directed to the right in anatomic position;
            - Point 1 = Ulnar Styloid;
            - Point 2 = Elbow center;
            - Point 3 = Radial styloid.

    Returns
    -------
    np.ndarray
        Series of transformation matrices (Nx4x4).

    References
    ----------
    1. G. Wu et al., "ISB recommendation on definitions of joint
       coordinate systems of various joints for the reporting of human joint
       motion - Part II: shoulder, elbow, wrist and hand," Journal of
       Biomechanics, vol. 38, no. 5, pp. 981--992, 2005.

    """
    def normalize(v):
        norm = np.linalg.norm(v, axis=1)
        return v / norm[..., np.newaxis]

    if (method == 'centroid') or (method == 'ocx1'):
        # Origin
        origin = np.mean(global_points, 2)[:, 0:3]
        # X axis
        x = global_points[:, 0:3, 0] - origin
        x = normalize(x)
        # Z plane
        y_temp = global_points[:, 0:3, 1] - origin
        y_temp = normalize(y_temp)
        # Z axis
        z = np.cross(x, y_temp)
        z = normalize(z)
        # Y axis
        y = np.cross(z, x)

    elif (method == 'yz') or (method == 'o1y2'):
        origin = global_points[:, 0:3, 0]
        y = normalize(global_points[:, 0:3, 1] - origin)
        z_temp = normalize(global_points[:, 0:3, 2] - origin)
        x = normalize(np.cross(y, z_temp))
        z = np.cross(x, y)

    elif (method == 'zx') or (method == 'o1z2'):
        origin = global_points[:, 0:3, 0]
        z = normalize(global_points[:, 0:3, 1] - origin)
        x_temp = normalize(global_points[:, 0:3, 2] - origin)
        y = normalize(np.cross(z, x_temp))
        x = np.cross(y, z)

    elif method == 'isb-humerus':
        # Origin
        origin = global_points[:, 0:3, 0]
        # Y axis
        y = origin - global_points[:, 0:3, 1]
        y = normalize(y)
        # YF axis
        yf = global_points[:, 0:3, 2] - global_points[:, 0:3, 1]
        yf = normalize(yf)
        # Z axis
        z = np.cross(yf, y)
        z = normalize(z)
        # X axis
        x = np.cross(y, z)

    elif method == 'isb-forearm':
        # Origin
        origin = global_points[:, 0:3, 0]
        # Y axis
        y = global_points[:, 0:3, 1] - origin
        y = normalize(y)
        # Z axis
        z = np.cross(global_points[:, 0:3, 2] - global_points[:, 0:3, 1], y)
        z = normalize(z)
        # X axis
        x = np.cross(y, z)

    else:
        raise ValueError(f'Method ({method}) is not implemented.')

    T = np.zeros((len(x), 4, 4))
    T[:, 0:3, 0] = x
    T[:, 0:3, 1] = y
    T[:, 0:3, 2] = z
    T[:, 0:3, 3] = origin
    T[:, 3, 3] = np.ones(len(x))

    return T


@unstable(listing)
def get_local_coordinates(global_coordinates: np.ndarray,
                          reference_frames: np.ndarray) -> np.ndarray:
    """
    Express global coordinates in local reference frames.

    Parameters
    ----------
    global_coordinates
        The global coordinates, as a series of N points, vectors or matrices.
        For example:

        - A series of N points or vectors : Nx4
        - A series of N set of M points or vectors : Nx4xM
        - A series of N 4x4 transformation matrices : Nx4x4

    reference_frames
        A series of N reference frames (Nx4x4) to express the global
        coordinates in.

    Returns
    -------
    np.ndarray
        Series of local coordinates in the same shape than
        `global_coordinates`.

    """
    n_samples = global_coordinates.shape[0]

    # Transform NaNs in global coordinates to zeros to perform the operation,
    # then put back NaNs in the corresponding local coordinates.
    nan_index = np.isnan(global_coordinates)
    global_coordinates[nan_index] = 0

    # Invert the reference frame to obtain the inverse transformation
    ref_rot = reference_frames[:, 0:3, 0:3]
    ref_t = reference_frames[:, 0:3, 3]

    # Inverse rotation : transpose.
    inv_ref_rot = np.transpose(ref_rot, (0, 2, 1))

    # Inverse translation : we inverse-rotate the translation.
    inv_ref_t = np.zeros(ref_t.shape)
    inv_ref_t = matmul(inv_ref_rot, -ref_t)

    inv_ref_T = np.zeros((n_samples, 4, 4))  # init
    inv_ref_T[:, 0:3, 0:3] = inv_ref_rot
    inv_ref_T[:, 0:3, 3] = inv_ref_t
    inv_ref_T[:, 3, 3] = np.ones(n_samples)

    local_coordinates = np.zeros(global_coordinates.shape)  # init
    local_coordinates = matmul(inv_ref_T, global_coordinates)

    # Put back the NaNs
    local_coordinates[nan_index] = np.nan

    return local_coordinates


@unstable(listing)
def get_global_coordinates(local_coordinates: np.ndarray,
                           reference_frames: np.ndarray) -> np.ndarray:
    """
    Express local coordinates in the global reference frame.

    Parameters
    ----------
    local_coordinates
        The local coordinates, as a series of N points, vectors or matrices.
        For example:

        - A series of N points or vectors : Nx4
        - A series of N set of M points or vectors : Nx4xM
        - A series of N 4x4 transformation matrices : Nx4x4

    reference_frames
        A series of N reference frames (Nx4x4) the local coordinates are
        expressed in.

    Returns
    -------
    np.ndarray
        Series of global coordinates in the same shape than
        `local_coordinates`.

    """
    global_coordinates = np.zeros(local_coordinates.shape)
    global_coordinates = matmul(reference_frames, local_coordinates)
    return global_coordinates


@unstable(listing)
def isnan(input: np.ndarray, /) -> np.ndarray:
    """
    Check which samples has at least one NaN.

    Parameters
    ----------
    input
        Array where the first dimension corresponds to time.

    Returns
    -------
    np.ndarray
        Array of bool that is the same size of input's first dimension, with
        True for the samples that contain at least one NaN.

    """
    temp = np.isnan(input)
    while len(temp.shape) > 1:
        temp = (temp.sum(axis=1) > 0)
    return temp


@unstable(listing)
def match_size(op1: np.ndarray, op2: np.ndarray) -> np.ndarray:
    """
    Match the first dimension of op1 and op2.

    Broadcasts the first dimension of op1 or op2, if required, so that both
    inputs have the same size in first dimension. If no modification is
    required on an input, then the output is a reference to the same input.
    Otherwise, the output is a new variable.

    Returns
    -------
    2x np.ndarray
        References or copies of op1 and op2 now matched in size.

    """
    if op1.shape[0] == 1:
        op1 = np.repeat(op1, op2.shape[0], axis=0)

    if op2.shape[0] == 1:
        op2 = np.repeat(op2, op1.shape[0], axis=0)

    if op1.shape[0] != op2.shape[0]:
        raise ValueError(
            'Could not match first dimension of op1 and op2')

    return op1, op2


@unstable(listing)
def register_points(global_points: np.ndarray,
                    local_points: np.ndarray) -> np.ndarray:
    """
    Find the rigid transformations between two series of point clouds.

    Parameters
    ----------
    global_points : array of shape Nx4xM
        Destination points as a series of N sets of M points.
    local_points : array of shape Nx4xM
        Local points as a series of N sets of M points.
        global_points and local_points must have the same shape.

    Returns
    -------
    np.ndarray
        Array of shape Nx4x4, expressing a series of 4x4 rigid transformation
        matrices.

    """
    n_samples = global_points.shape[0]

    # Prealloc the transformation matrix
    T = np.zeros((n_samples, 4, 4))
    T[:, 3, 3] = np.ones(n_samples)

    for i_sample in range(n_samples):

        # Identify which global points are visible
        sample_global_points = global_points[i_sample]
        sample_global_points_missing = np.isnan(
                np.sum(sample_global_points, axis=0))

        # Identify which local points are visible
        sample_local_points = local_points[i_sample]
        sample_local_points_missing = np.isnan(
                np.sum(sample_local_points, axis=0))

        sample_points_missing = np.logical_or(
                sample_global_points_missing, sample_local_points_missing)

        # If at least 3 common points are visible between local and global
        # points, then we can regress the transformation.
        if sum(~sample_points_missing) >= 3:
            T[i_sample] = icp.best_fit_transform(
                    sample_local_points[0:3, ~sample_points_missing].T,
                    sample_global_points[0:3, ~sample_points_missing].T)[0]
        else:
            T[i_sample] = np.nan

    return T

def __dir__():
    return listing
