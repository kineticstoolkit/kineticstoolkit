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
to time.

"""
from __future__ import annotations


__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2021 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


def __dir__():
    return [
        "matmul",
        "create_transforms",
        "get_angles",
        "create_frames",
        "get_local_coordinates",
        "get_global_coordinates",
        "isnan",
        "register_points",
        "rotate",
        "translate",
        "scale",
    ]


import numpy as np
import scipy.spatial.transform as transform
import kineticstoolkit.external.icp as icp
from numpy.typing import ArrayLike
from kineticstoolkit.exceptions import check_types


def matmul(op1: ArrayLike, op2: ArrayLike, /) -> np.ndarray:
    """
    Matrix multiplication between series of matrices.

    This function is a wrapper for numpy's matmul function (operator @), that
    uses Kinetics Toolkit's convention that the first dimension always
    corresponds to time, to broadcast time correctly between operands.

    Parameters
    ----------
    op1
        Series of floats, vectors or matrices.
    op2
        Series of floats, vectors or matrices.

    Returns
    -------
    np.ndarray
        The product, usually as a series of Nx4 or Nx4xM matrices.

    Example
    -------
    A matrix multiplication between one matrix and a series of 3 vectors
    results in a series of 3 vectors.

    >>> import kineticstoolkit as ktk
    >>> mat_series = np.array([[[2.0, 0.0], [0.0, 1.0]]])
    >>> vec_series = np.array([[4.0, 5.0], [6.0, 7.0], [8.0, 9.0]])
    >>> ktk.geometry.matmul(mat_series, vec_series)
    array([[ 8.,  5.],
           [12.,  7.],
           [16.,  9.]])

    """
    check_types(matmul, locals())

    op1 = np.array(op1)
    op2 = np.array(op2)

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


def inv(matrix_series: ArrayLike, /) -> np.ndarray:
    """
    Calculate series of inverse transform.

    This function calculates a series of inverse homogeneous transforms.

    Parameters
    ----------
    matrix_series
        Nx4x4 series of homogeneous matrices, where each matrix is an
        homogeneous transform.

    Returns
    -------
    ArrayLike
        The Nx4x4 series of inverse homogeneous matrices.

    Note
    ----
    This function requires (and checks) that each matrix really is an
    homogeneous transform by evaluating the determinant of its rotation
    component. It then calculates the inverse matrix quickly using the
    transpose of the rotation component.

    """
    check_types(inv, locals())

    matrix_series = np.array(matrix_series)
    index_is_nan = isnan(matrix_series)

    _check_no_skewed_rotation(matrix_series, "matrix_series")

    invR = np.zeros((matrix_series.shape[0], 3, 3))
    invT = np.zeros((matrix_series.shape[0], 3))

    # Inverse rotation
    invR = np.transpose(matrix_series[:, 0:3, 0:3], (0, 2, 1))

    # Inverse translation
    invT = -matmul(invR, matrix_series[:, 0:3, 3])

    # output
    out = np.zeros(matrix_series.shape)
    out[:, 0:3, 0:3] = invR
    out[:, 0:3, 3] = invT
    out[:, 3, 3] = 1
    out[index_is_nan] = np.nan

    return out


def create_transforms(
    seq: str | None = None,
    angles: ArrayLike | None = None,
    translations: ArrayLike | None = None,
    scales: ArrayLike | None = None,
    *,
    degrees=False,
) -> np.ndarray:
    """
    Create series of transforms based on angles, translatations and scales.

    Create an Nx4x4 series of homogeneous transform matrices based on series of
    angles, translatations and scales.

    Parameters
    ----------
    seq
        Optional. Specifies sequence of axes for rotations. Up to 3 characters
        belonging to the set {'X', 'Y', 'Z'} for intrinsic rotations (moving
        axes), or {'x', 'y', 'z'} for extrinsic rotations (fixed axes).
        Extrinsic and intrinsic rotations cannot be mixed in one function call.
        Required if angles is specified.

    angles
        Optional array_like of shape (N,) or (N, [1 or 2 or 3]). Angles are
        specified in radians (if degrees is False) or degrees (if degrees is
        True).

        For a single-character `seq`, `angles` can be:

        - array_like with shape (N,), where each `angle[i]` corresponds to a
          single rotation;
        - array_like with shape (N, 1), where each `angle[i, 0]` corresponds
          to a single rotation.

        For 2- and 3-character `seq`, `angles` is an array_like with shape
        (N, W) where each `angle[i, :]` corresponds to a sequence of Euler
        angles and W is the length of `seq`.

    translations
        Optional array_like of shape (N, 3) or (N, 4). This corresponds
        to the translation part of the generated series of homogeneous
        transforms.

    scales
        Optional array_like of shape (N, ) that corresponds to the scale to
        apply uniformly on the three axes. By default, no scale is included.

    degrees
        If True, then the given angles are in degrees. Default is False.

    Returns
    -------
    np.ndarray
        An Nx4x4 series of homogeneous transforms.

    See also
    --------
    ktk.geometry.create_frames, ktk.geometry.rotate, ktk.geometry.translate,
    ktk.geometry.scale

    Examples
    --------
    Create a series of two homogeneous transforms that rotates 0, then 90
    degrees around x:

        >>> import kineticstoolkit.lab as ktk
        >>> ktk.geometry.create_transforms(seq='x', angles=[0, 90], degrees=True)
        array([[[ 1.,  0.,  0.,  0.],
                [ 0.,  1.,  0.,  0.],
                [ 0.,  0.,  1.,  0.],
                [ 0.,  0.,  0.,  1.]],
        <BLANKLINE>
               [[ 1.,  0.,  0.,  0.],
                [ 0.,  0., -1.,  0.],
                [ 0.,  1.,  0.,  0.],
                [ 0.,  0.,  0.,  1.]]])

    Create an homogeneous transform that converts millimeters to meters

        >>> import kineticstoolkit.lab as ktk
        >>> ktk.geometry.create_transforms(scales=[0.001])
        array([[[0.001, 0.   , 0.   , 0.   ],
                [0.   , 0.001, 0.   , 0.   ],
                [0.   , 0.   , 0.001, 0.   ],
                [0.   , 0.   , 0.   , 1.   ]]])

    """
    check_types(create_transforms, locals())

    # Condition translations
    if translations is None:
        translations = np.zeros((1, 3))
    else:
        translations = np.array(translations)

    # Condition angles
    if angles is None:
        angles = np.array([0])
        seq = "x"
    else:
        angles = np.array(angles)

    # Condition scales
    if scales is None:
        scales = np.array([1])
    else:
        scales = np.array(scales)

    # Convert scales to a series of scaling matrices
    temp = np.zeros((scales.shape[0], 4, 4))
    temp[:, 0, 0] = scales
    temp[:, 1, 1] = scales
    temp[:, 2, 2] = scales
    temp[:, 3, 3] = 1.0
    scales = temp

    # Match sizes
    translations, angles = _match_size(translations, angles)
    translations, scales = _match_size(translations, scales)
    translations, angles = _match_size(translations, angles)
    n_samples = angles.shape[0]

    # Create the rotation matrix
    rotation = transform.Rotation.from_euler(seq, angles, degrees)
    R = rotation.as_matrix()
    if len(R.shape) == 2:  # Single rotation: add the Time dimension.
        R = R[np.newaxis, ...]

    # Construct the final series of transforms (without scaling)
    T = np.empty((n_samples, 4, 4))
    T[:, 0:3, 0:3] = R
    T[:, 0:3, 3] = translations
    T[:, 3, 0:3] = 0
    T[:, 3, 3] = 1

    # Return the scaling + transform
    return T @ scales


def rotate(
    coordinates, /, seq: str, angles: ArrayLike, *, degrees: bool = False
) -> np.ndarray:
    """
    Rotate a series of coordinates along given axes.

    Parameters
    ----------
    coordinates
        Array_like of shape (N, ...): the coordinates to rotate.

    seq
        Specifies sequence of axes for rotations. Up to 3 characters
        belonging to the set {'X', 'Y', 'Z'} for intrinsic rotations (moving
        axes), or {'x', 'y', 'z'} for extrinsic rotations (fixed axes).
        Extrinsic and intrinsic rotations cannot be mixed in one function call.

    angles
        Array_like of shape (N,) or (N, [1 or 2 or 3]). Angles are
        specified in radians (if degrees is False) or degrees (if degrees is
        True).

        For a single-character `seq`, `angles` can be:

        - array_like with shape (N,), where each `angle[i]` corresponds to a
          single rotation;
        - array_like with shape (N, 1), where each `angle[i, 0]` corresponds
          to a single rotation.

        For 2- and 3-character `seq`, `angles` is an array_like with shape
        (N, W) where each `angle[i, :]` corresponds to a sequence of Euler
        angles and W is the length of `seq`.

    degrees
        If True, then the given angles are in degrees. Default is False.

    Returns
    -------
    np.ndarray
        Array_like of shape (N, ...): the rotated coordinates.

    See also
    --------
    ktk.geometry.translate, ktk.geometry.scale, ktk.geometry.create_transforms,
    ktk.geometry.matmul

    Examples
    --------
    Rotate the point (1, 0, 0) by theta degrees around z, then by 45 degrees
    around y, for theta in [0, 10, 20, 30, 40]:

        >>> import kineticstoolkit.lab as ktk
        >>> angles = np.array([[0, 45], [10, 45], [20, 45], [30, 45], [40, 45]])
        >>> ktk.geometry.rotate([[1, 0, 0, 1]], 'zx', angles, degrees=True)
        array([[1.        , 0.        , 0.        , 1.        ],
               [0.98480775, 0.1227878 , 0.1227878 , 1.        ],
               [0.93969262, 0.24184476, 0.24184476, 1.        ],
               [0.8660254 , 0.35355339, 0.35355339, 1.        ],
               [0.76604444, 0.45451948, 0.45451948, 1.        ]])

    """
    return matmul(create_transforms(seq, angles, degrees=degrees), coordinates)


def translate(coordinates, /, translations):
    """
    Translate a series of coordinates.

    Parameters
    ----------
    coordinates
        Array_like of shape (N, ...): the coordinates to translate.

    translations
        Array_like of shape (N, 3) or (N, 4): the translation on each axe
        (x, y, z).

    Returns
    -------
    np.ndarray
        Array_like of shape (N, ...): the translated coordinates.

    See also
    --------
    ktk.geometry.rotate, ktk.geometry.scale, ktk.geometry.create_transforms,
    ktk.geometry.matmul

    Examples
    --------
    Translate the point (1, 0, 0) by (x, 1, 0), for x in [0, 1, 2, 3, 4]:

        >>> import kineticstoolkit.lab as ktk
        >>> t = np.array([[0, 1, 0], [1, 1, 0], [2, 1, 0], [3, 1, 0], [4, 1, 0]])
        >>> ktk.geometry.translate([[1, 0, 0, 1]], t)
        array([[1., 1., 0., 1.],
               [2., 1., 0., 1.],
               [3., 1., 0., 1.],
               [4., 1., 0., 1.],
               [5., 1., 0., 1.]])
    """
    return matmul(create_transforms(translations=translations), coordinates)


def scale(coordinates, /, scales):
    """
    Scale a series of coordinates.

    Parameters
    ----------
    coordinates
        Array_like of shape (N, ...): the coordinates to scale.

    scales
        Array_like of shape (N, ) that corresponds to the scale to apply
        uniformly on the three axes.

    Returns
    -------
    np.ndarray
        Array_like of shape (N, ...): the translated coordinates.

    See also
    --------
    ktk.geometry.rotate, ktk.geometry.translate,
    ktk.geometry.create_transforms, ktk.geometry.matmul

    Examples
    --------
    Scale the point (1, 0, 0) by x, for x in [0, 1, 2, 3, 4]:

        >>> import kineticstoolkit.lab as ktk
        >>> s = np.array([0, 1, 2, 3, 4])
        >>> ktk.geometry.scale([[1, 0, 0, 1]], s)
        array([[0., 0., 0., 1.],
               [1., 0., 0., 1.],
               [2., 0., 0., 1.],
               [3., 0., 0., 1.],
               [4., 0., 0., 1.]])
    """
    return matmul(create_transforms(scales=scales), coordinates)


def get_angles(
    T: ArrayLike, seq: str, degrees: bool = False, flip: bool = False
) -> np.ndarray:
    """
    Extract Euler angles from a series of homogeneous matrices.

    In case of gimbal lock, a warning is raised, and the third angle is set to
    zero. Note however that the returned angles still represent the correct
    rotation.

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
        If True, the returned angles are in degrees. If False, they are in
        radians. Default is False.

    flip
        Return an alternate sequence with the second angle inverted, but that
        leads to the same rotation matrices. See below for more information.

    Returns
    -------
    np.ndarray
        An Nx3 series of Euler angles, with the second dimension containing
        the first, second and third angles, respectively.

    Notes
    -----
    The range of the returned angles is dependant on the `flip` parameter. If
    `flip` is False:

    - First angle belongs to [-180, 180] degrees (both inclusive)
    - Second angle belongs to:

        - [-90, 90] degrees if all axes are different. e.g., xyz
        - [0, 180] degrees if first and third axes are the same e.g., zxz

    - Third angle belongs to [-180, 180] degrees (both inclusive)

    If `flip` is True:

    - First angle belongs to [-180, 180] degrees (both inclusive)
    - Second angle belongs to:

        - [-180, -90], [90, 180] degrees if all axes are different. e.g., xyz
        - [-180, 0] degrees if first and third axes are the same e.g., zxz

    - Third angle belongs to [-180, 180] degrees (both inclusive)

    This function is a wrapper for scipy.transform.Rotation.as_euler. Please
    consult scipy help for more help on intrinsic/extrinsic angles and the
    `seq` parameter.

    """
    check_types(get_angles, locals())

    T = np.array(T)

    _check_no_skewed_rotation(T, "T")

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
            angles[angles[:, 1] > offset, :] -= 2 * offset
            angles[:, 2] = np.mod(angles[:, 2], 2 * offset) - offset

    return angles


def create_frames(
    origin: ArrayLike,
    x: ArrayLike | None = None,
    y: ArrayLike | None = None,
    z: ArrayLike | None = None,
    xy: ArrayLike | None = None,
    xz: ArrayLike | None = None,
    yz: ArrayLike | None = None,
) -> np.ndarray:
    """
    Create an Nx4x4 series of frames based on series of points and vectors.

    Parameters
    ----------
    origin
        A series of N points (Nx4) that corresponds to the origin of the
        series of frames to be created.

    x, y, z
        Define either `x`, `y` or `z`. A series of N vectors (Nx4) that
        are aligned toward the {x|y|z} series of frames to be created.

    xy, xz
        Only if `x` is specified. A series of N vectors (Nx4) in the {xy|xz}
        plane of the series of frames to be created. As a rule of thumb, use
        a series of N vectors that correspond roughly to the {z|-y} axis.

    xy, yz
        Only if `y` is specified. A series of N vectors (Nx4) in the {xy|yz}
        plane of the series of frames to be created. As a rule of thumb, use
        a series of N vectors that correspond roughly to the {z|x} axis.

    xz, yz
        Only if `z` is specified. A series of N vectors (Nx4) in the {xz|yz}
        plane of the series of frames to be created. As a rule of thumb, use
        a series of N vectors that correspond roughly to the {-y|x} axis.

    Returns
    -------
    np.ndarray
        Series of frames (Nx4x4).

    See also
    --------
    ktk.geometry.create_transforms

    """
    check_types(create_frames, locals())

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


def get_local_coordinates(
    global_coordinates: ArrayLike, reference_frames: ArrayLike
) -> np.ndarray:
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

    See also
    --------
    ktk.geometry.get_global_coordinates

    """
    check_types(get_local_coordinates, locals())

    global_coordinates = np.array(global_coordinates)
    reference_frames = np.array(reference_frames)

    _check_no_skewed_rotation(reference_frames, "reference_frames")

    (global_coordinates, reference_frames) = _match_size(
        global_coordinates, reference_frames
    )

    n_samples = global_coordinates.shape[0]

    # Transform NaNs in global coordinates to zeros to perform the operation,
    # then put back NaNs in the corresponding local coordinates.
    nan_index = np.isnan(global_coordinates)
    global_coordinates[nan_index] = 0

    # Invert the reference frame to obtain the inverse transformation
    inv_ref_T = inv(reference_frames)

    local_coordinates = np.zeros(global_coordinates.shape)  # init
    local_coordinates = matmul(inv_ref_T, global_coordinates)

    # Put back the NaNs
    local_coordinates[nan_index] = np.nan

    return local_coordinates


def get_global_coordinates(
    local_coordinates: ArrayLike, reference_frames: ArrayLike
) -> np.ndarray:
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

    See also
    --------
    ktk.geometry.get_local_coordinates

    """
    check_types(get_global_coordinates, locals())

    local_coordinates = np.array(local_coordinates)
    reference_frames = np.array(reference_frames)

    _check_no_skewed_rotation(reference_frames, "reference_frames")

    (local_coordinates, reference_frames) = _match_size(
        local_coordinates, reference_frames
    )

    global_coordinates = np.zeros(local_coordinates.shape)
    global_coordinates = matmul(reference_frames, local_coordinates)
    return global_coordinates


def isnan(array: ArrayLike, /) -> np.ndarray:
    """
    Check which samples contain at least one NaN.

    Parameters
    ----------
    in
        Array where the first dimension corresponds to time.

    Returns
    -------
    np.ndarray
        Array of bool that is the same size of input's first dimension, with
        True for the samples that contain at least one NaN.

    """
    check_types(isnan, locals())

    temp = np.isnan(array)
    while len(temp.shape) > 1:
        temp = temp.sum(axis=1) > 0
    return temp


def _match_size(
    op1: np.ndarray, op2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Match the first dimension of op1 and op2.

    Broadcasts the first dimension of op1 or op2, if required, so that both
    inputs have the same size in first dimension. If no modification is
    required on an input, then the output is a reference to the same input.
    Otherwise, the output is a new variable.

    Returns
    -------
    tuple of two np.ndarray
        op1 and op2, now matched in size.

    """
    if op1.shape[0] == 1:
        op1 = np.repeat(op1, op2.shape[0], axis=0)

    if op2.shape[0] == 1:
        op2 = np.repeat(op2, op1.shape[0], axis=0)

    if op1.shape[0] != op2.shape[0]:
        raise ValueError("Could not match first dimension of op1 and op2")

    return op1, op2


def _check_no_skewed_rotation(series: np.ndarray, param_name) -> None:
    """
    Check if all rotation matrices are orthogonal (det=1).

    Parameters
    ----------
    matrix_series : array of shape Nx4x4
        The input series. Inputs of other shapes are ignored.
    param_name
        Name of the parameters, to use in the error message.

    Raises
    ------
    ValueError
        If at least one skewed rotation matrix is found in the provided series.

    """
    if (
        len(series.shape) == 3
        and series.shape[1] == 4
        and series.shape[2] == 4
    ):
        index_is_nan = isnan(series)
        if not np.allclose(np.linalg.det(series[~index_is_nan, 0:3, 0:3]), 1):
            raise ValueError(
                f"Parameter {param_name} contains at least one rotation "
                "component that is not orthogonal. This may happen, for "
                "instance, if you attempted to average, resample, or filter a "
                "homogeneous transform, which is usually forbidden. If this "
                "is the case, then consider filtering quaternions or Euler "
                "angles instead. If you created a homogeneous transform from "
                "3D marker trajectories, then average/resample/filter the "
                "marker trajectories before creating the transform, instead "
                "of averaging/resampling/filtering the transform."
            )


def register_points(
    global_points: ArrayLike, local_points: ArrayLike
) -> np.ndarray:
    """
    Find the homogeneous transforms between two series of point clouds.

    Parameters
    ----------
    global_points
        Destination points as an Nx4xM series of N sets of M points.
    local_points
        Local points as an array of shape Nx4xM series of N sets of M points.
        global_points and local_points must have the same shape.

    Returns
    -------
    np.ndarray
        Array of shape Nx4x4, expressing a series of 4x4 homogeneous
        transforms.

    """
    check_types(register_points, locals())

    global_points = np.array(global_points)
    local_points = np.array(local_points)

    n_samples = global_points.shape[0]

    # Prealloc the transformation matrix
    T = np.zeros((n_samples, 4, 4))
    T[:, 3, 3] = np.ones(n_samples)

    for i_sample in range(n_samples):
        # Identify which global points are visible
        sample_global_points = global_points[i_sample]
        sample_global_points_missing = np.isnan(
            np.sum(sample_global_points, axis=0)
        )

        # Identify which local points are visible
        sample_local_points = local_points[i_sample]
        sample_local_points_missing = np.isnan(
            np.sum(sample_local_points, axis=0)
        )

        sample_points_missing = np.logical_or(
            sample_global_points_missing, sample_local_points_missing
        )

        # If at least 3 common points are visible between local and global
        # points, then we can regress the transformation.
        if sum(~sample_points_missing) >= 3:
            T[i_sample] = icp.best_fit_transform(
                sample_local_points[0:3, ~sample_points_missing].T,
                sample_global_points[0:3, ~sample_points_missing].T,
            )[0]
        else:
            T[i_sample] = np.nan

    return T


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
