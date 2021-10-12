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
__copyright__ = "Copyright (C) 2020-2021 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import numpy as np
import scipy.spatial.transform as transform
import kineticstoolkit.external.icp as icp
from kineticstoolkit.decorators import unstable, directory
from typing import Optional, Tuple


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

    (op1, op2) = _match_size(op1, op2)

    n_samples = op1.shape[0]

    # Get the expected shape by performing the first multiplication
    temp = perform_mul(op1[0], op2[0])
    result = np.empty((n_samples, *temp.shape))

    # Perform the multiplication
    for i_sample in range(n_samples):
        result[i_sample] = perform_mul(op1[i_sample], op2[i_sample])

    return result


def create_transforms(seq: Optional[str] = None,
                      angles: Optional[np.ndarray] = None,
                      translations: Optional[np.ndarray] = None,
                      *,
                      degrees=False) -> np.ndarray:
    """
    Create an Nx4x4 series of homogeneous transforms.

    Warning
    -------
    This function, which has been introduced in 0.4, is still experimental and
    may change signature or behaviour in the future.

    Parameters
    ----------
    seq
        Optional. Specifies sequence of axes for rotations. Up to 3 characters
        belonging to the set {'X', 'Y', 'Z'} for intrinsic rotations (moving
        axes), or {'x', 'y', 'z'} for extrinsic rotations (fixed axes).
        Extrinsic and intrinsic rotations cannot be mixed in one function call.
        Required if angles is specified.

    angles
        Optional array_like, shape (N,) or (N, [1 or 2 or 3]).
        Angles are specified in radians (degrees is False) or degrees (degrees
        is True).

        For a single character seq, angles can be:

            - array_like with shape (N,), where each angle[i] corresponds to a
              single rotation
            - array_like with shape (N, 1), where each angle[i, 0] corresponds
              to a single rotation

        For 2- and 3-character wide seq, angles can be:

            - array_like with shape (W,) where W is the width of seq, which
              corresponds to a single rotation with W axes
            - array_like with shape (N, W) where each angle[i] corresponds to
              a sequence of Euler angles describing a single rotation

    translations
        Optional float or array_like, shape (N, 3) or (N, 4). This corresponds
        to the translation part of the generated series of homogeneous
        transforms.

    degrees
        If True, then the given angles are in degrees. Default is False.

    Returns
    -------
    np.ndarray
        An Nx4x4 series of homogeneous transforms.

    Example
    -------
    Create a series of two homogeneous transforms that rotates 0, then 90
    degrees around x:

        >>> import kineticstoolkit.lab as ktk
        >>> ktk.geometry.create_transforms('x', [0, 90], degrees=True)
        array([[[ 1.,  0.,  0.,  0.],
                [ 0.,  1.,  0.,  0.],
                [ 0.,  0.,  1.,  0.],
                [ 0.,  0.,  0.,  1.]],
        <BLANKLINE>
               [[ 1.,  0.,  0.,  0.],
                [ 0.,  0., -1.,  0.],
                [ 0.,  1.,  0.,  0.],
                [ 0.,  0.,  0.,  1.]]])

    """
    # Condition translations
    if translations is None:
        translations = np.zeros((1, 3))
    else:
        translations = np.array(translations)

    # Condition angles
    if angles is None:
        angles = np.array([0])
        seq = 'x'
    else:
        angles = np.array(angles)

    # Match sizes
    translations, angles = _match_size(translations, angles)
    n_samples = angles.shape[0]

    # Create the rotation matrix
    rotation = transform.Rotation.from_euler(seq, angles, degrees)
    R = rotation.as_matrix()
    if len(R.shape) == 2:  # Single rotation: add the Time dimension.
        R = R[np.newaxis, ...]

    # Construct the final series of transforms
    T = np.empty((n_samples, 4, 4))
    T[:, 0:3, 0:3] = R
    T[:, 0:3, 3] = translations
    T[:, 3, 0:3] = 0
    T[:, 3, 3] = 1
    return T


def get_angles(T: np.ndarray,
               seq: str,
               degrees: bool = False,
               flip: bool = False) -> np.ndarray:
    """
    Represent a series of transformation matrices as series of Euler angles.

    This function is a wrapper for scipy.transform.Rotation.as_euler. Please
    consult scipy help for the complete docstring.

    Euler angles suffer from the problem of gimbal lock, where the
    representation loses a degree of freedom and it is not possible to
    determine the first and third angles uniquely. In this case, a warning is
    raised, and the third angle is set to zero. Note however that the returned
    angles still represent the correct rotation.

    Warning
    -------
    This function, which has been introduced in 0.4, is still experimental and
    may change signature or behaviour in the future.

    Parameters
    ----------
    T
        An Nx4x4 series of transformation matrices.

    seq
        3 characters belonging to the set {'X', 'Y', 'Z'} for intrinsic
        rotations (moving axes), or {'x', 'y', 'z'} for extrinsic rotations
        (fixed axes). Adjacent axes cannot be the same. Extrinsic and
        intrinsic rotations cannot be mixed in one function call.

    degrees
        Returned angles are in degrees if this flag is True, else they are in
        radians. Default is False.

    flip
        Return an alternate sequence with the second angle inverted, that
        leads to the same rotation matrices. More specifically:

        First angle belongs to [-180, 180] degrees (both inclusive)

        Second angle belongs to:
            - Default case (alt_angles = False):
                - [-90, 90] degrees if all axes are different (like xyz)
                - [0, 180] degrees if first and third axes are the same
                  (like zxz)
            - Alternate case (alt_angles = True):
                - [-180, -90], [90, 180] degrees if all axes are different
                  (like xyz)
                - [-180, 0] degrees if first and third axes are the same
                  (like zxz)

        Third angle belongs to [-180, 180] degrees (both inclusive)

        One rationale for adding this special case is the calculation
        of shoulder angles: when following the ISB recommendation (YXY), X
        corresponds to the negative elevation and thus we expect to get
        negative values for Y. In this case, just use alt_angles = True.

    Returns
    -------
    np.ndarray
        A Tx3 array of Euler angles.


    """
    R = transform.Rotation.from_matrix(T[:, 0:3, 0:3])
    angles = R.as_euler(seq, degrees)

    offset = 180 if degrees else np.pi

    if flip:
        if seq[0] == seq[2]:  # Euler angles
            angles[:, 0] = np.mod(angles[:, 0], 2 * offset) - offset
            angles[:, 1] = -angles[:, 1]
            angles[:, 2] = np.mod(angles[:, 2], 2 * offset) - offset
        else:  # Tait-Bryan angles
            angles[:, 0] = np.mod(angles[:, 0], 2 * offset) - offset
            angles[:, 1] = offset - angles[:, 1]
            angles[angles[:, 1] > offset, :] -= (2 * offset)
            angles[:, 2] = np.mod(angles[:, 2], 2 * offset) - offset

    return angles


def create_frames(
        origin: np.ndarray,
        x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        z: Optional[np.ndarray] = None,
        xy: Optional[np.ndarray] = None,
        xz: Optional[np.ndarray] = None,
        yz: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Create a Nx4x4 series of frames based on series of points and vectors.

    Create reference frames based on points and vectors and return this series
    of reference frames as a series of transformation matrices.

    This function's behaviour is better explained using an example. We will
    create reference frames for the right humerus, based on the recommendations
    of the International Society of Biomechanics [1]. Lets say we have
    constructed series of points for the glenohumeral joint (GH), lateral elbow
    epicondyle (EL) and medial elbow epicondyle (EM). Following the ISB:

        1. The origin is GH;
        2. The y axis is the line between GH and the midpoint of EL and EM,
           pointing to GH;
        3. The x axis is the normal to the GH-EL-EM plane, pointing forward;
           which means that GH-EL-EM is a yz plane.
        4. The z axis is perpendicular to x and y, pointing to the right.

    Therefore:

        1. origin = GH
        2. y = GH - (EL + EM) / 2
        3. yz = EL - EM  # The x axis is formed by cross(y, yz)
        4. reference_frames = ktk.geometry.create_reference_frames(
            origin=origin, y=y, yz=yz)

    Warning
    -------
    This function, which has been introduced in 0.4, is still experimental and
    may change signature or behaviour in the future.

    Parameters
    ----------
    origin
        A series of N points (Nx4) that corresponds to the origin of the
        returned reference frames.

    x|y|z
        A series of N vectors (Nx4) that are aligned toward the {x|y|z}
        axis of the returned reference frames.

    xy|xz
        When x is specified, a series of N vectors (Nx4) in the {xy|xz} plane
        of the returned reference frames.

    xy|yz
        When y is specified, a series of N vectors (Nx4) in the {xy|yz} plane
        of the returned reference frames.

    xz|yz
        When z is specified, a series of N vectors (Nx4) in the {xz|yz} plane
        of the returned reference frames.

    Returns
    -------
    np.ndarray
        Series of transformation matrices (Nx4x4).

    Examples
    --------
    Create a translated reference frame using 3 points:

        >>> import kineticstoolkit.lab as ktk
        >>> origin = [[2., 2., 2., 1.]]
        >>> x = [[10., 0., 0., 0.]]
        >>> xy = [[10., 10., 0., 0.]]
        >>> rf = ktk.geometry.create_frames(origin, x=x, xy=xy)
        >>> rf
        array([[[1., 0., 0., 2.],
                [0., 1., 0., 2.],
                [0., 0., 1., 2.],
                [0., 0., 0., 1.]]])

    References
    ----------
    1. G. Wu et al., "ISB recommendation on definitions of joint
       coordinate systems of various joints for the reporting of human joint
       motion - Part II: shoulder, elbow, wrist and hand," Journal of
       Biomechanics, vol. 38, no. 5, pp. 981--992, 2005.

    """
    def normalize(v):
        """Normalize series of vectors."""
        norm = np.linalg.norm(v, axis=1)
        return v / norm[..., np.newaxis]

    def cross(v1, v2):
        """Cross on series of vectors of length 4."""
        c = v1.copy()
        c[:, 0:3] = np.cross(v1[:, 0:3], v2[:, 0:3])
        return c

    origin = np.array(origin)

    if x is not None:
        v_x = normalize(np.array(x))
        if xy is not None:
            v_z = normalize(cross(v_x, np.array(xy)))
            v_y = cross(v_z, v_x)
        elif xz is not None:
            v_y = -normalize(cross(v_x, np.array(xz)))
            v_z = cross(v_x, v_y)
        else:
            raise ValueError("Either xy or xz must be set.")

    elif y is not None:
        v_y = normalize(np.array(y))
        if yz is not None:
            v_x = normalize(cross(v_y, np.array(yz)))
            v_z = cross(v_x, v_y)
        elif xy is not None:
            v_z = -normalize(cross(v_y, np.array(xy)))
            v_x = cross(v_y, v_z)
        else:
            raise ValueError("Either xy or yz must be set.")

    elif z is not None:
        v_z = normalize(np.array(z))
        if xz is not None:
            v_y = normalize(cross(v_z, np.array(xz)))
            v_x = cross(v_y, v_z)
        elif yz is not None:
            v_x = -normalize(cross(v_z, np.array(yz)))
            v_y = cross(v_z, v_x)
        else:
            raise ValueError("Either yz or xz must be set.")

    else:
        raise ValueError("Either x, y or z must be set.")

    return np.stack((v_x, v_y, v_z, origin), axis=2)


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


@unstable
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


def _match_size(op1: np.ndarray, op2: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray]:
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


def register_points(global_points: np.ndarray,
                    local_points: np.ndarray) -> np.ndarray:
    """
    Find the homogeneous transforms between two series of point clouds.

    Warning
    -------
    This function, which has been introduced in 0.4, is still experimental and
    may change signature or behaviour in the future.

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
        Array of shape Nx4x4, expressing a series of 4x4 homogeneous
        transforms.

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


module_locals = locals()


def __dir__():  # pragma: no cover
    return directory(module_locals)


if __name__ == "__main__":  # pragma: no cover
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
