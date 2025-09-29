#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020-2025 Félix Chénier

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
Provide 3D geometry and linear algebra functions related to biomechanics.

Note
----
As a convention, the first dimension of every array is always N and corresponds
to time.

"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2025 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"

import numpy as np
import scipy.spatial.transform as transform
import kineticstoolkit.external.icp as icp
from kineticstoolkit.typing_ import ArrayLike, check_param

import kineticstoolkit as ktk  # For doctests


def __dir__():
    return [
        "create_transform_series",
        "create_point_series",
        "create_vector_series",
        "is_transform_series",
        "is_point_series",
        "is_vector_series",
        "matmul",
        "invert",
        "rotate",
        "translate",
        "scale",
        "mirror",
        "get_local_coordinates",
        "get_global_coordinates",
        "get_angles",
        "get_quaternions",
        "get_distances",
        "register_points",
        "isnan",
    ]


# %% Matrix multiplication and inverse


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
    op1_array = np.array(op1)
    op2_array = np.array(op2)

    def perform_mul(op1, op2):
        if isinstance(op1, np.ndarray) and isinstance(op2, np.ndarray):
            return op1 @ op2
        else:
            return op1 * op2  # In the case where we have a series of floats.

    (op1_array, op2_array) = _match_size(op1_array, op2_array)

    n_samples = op1_array.shape[0]

    # Get the expected shape by performing the first multiplication
    temp = perform_mul(op1_array[0], op2_array[0])
    result = np.empty((n_samples, *temp.shape))

    # Perform the multiplication
    for i_sample in range(n_samples):
        result[i_sample] = perform_mul(
            op1_array[i_sample], op2_array[i_sample]
        )

    return result


def invert(matrix_series: ArrayLike, /) -> np.ndarray:
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
    This function requires (and checks) that the input is a transform series.
    It then calculates the inverse matrix quickly using the transpose of the
    rotation component.

    """
    matrix_series = np.array(matrix_series)
    index_is_nan = isnan(matrix_series)

    if not is_transform_series(matrix_series):
        raise ValueError(
            "The input must be a series of homogeneous transform series."
        )

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


# %% Rotate, translate, scale, mirror


def rotate(
    coordinates, /, seq: str, angles: ArrayLike, *, degrees: bool = False
) -> np.ndarray:
    """
    Rotate a series of coordinates along given axes.

    Parameters
    ----------
    coordinates
        ArrayLike of shape (N, ...): the coordinates to rotate.

    seq
        Specifies sequence of axes for rotations. Up to 3 characters
        belonging to the set {"X", "Y", "Z"} for intrinsic rotations (moving
        axes), or {"x", "y", "z"} for extrinsic rotations (fixed axes).
        Extrinsic and intrinsic rotations cannot be mixed in one function call.

    angles
        ArrayLike of shape (N,) or (N, [1 or 2 or 3]). Angles are
        specified in radians (if degrees is False) or degrees (if degrees is
        True).

        For a single-character `seq`, `angles` can be:

        - ArrayLike with shape (N,), where each `angle[i]` corresponds to a
          single rotation;
        - ArrayLike with shape (N, 1), where each `angle[i, 0]` corresponds
          to a single rotation.

        For 2- and 3-character `seq`, `angles` is an ArrayLike with shape
        (N, W) where each `angle[i, :]` corresponds to a sequence of Euler
        angles and W is the length of `seq`.

    degrees
        If True, then the given angles are in degrees. Default is False.

    Returns
    -------
    np.ndarray
        ArrayLike of shape (N, ...): the rotated coordinates.

    See Also
    --------
    ktk.geometry.translate, ktk.geometry.scale, ktk.geometry.mirror

    Examples
    --------
    Rotate the point (1, 0, 0) by theta degrees around z, then by 45 degrees
    around y, for theta in [0, 10, 20, 30, 40]:

        >>> import kineticstoolkit.lab as ktk
        >>> angles = np.array([[0, 45], [10, 45], [20, 45], [30, 45], [40, 45]])
        >>> ktk.geometry.rotate([[1, 0, 0, 1]], "zx", angles, degrees=True)
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
        ArrayLike of shape (N, ...): the coordinates to translate.

    translations
        ArrayLike of shape (N, 3) or (N, 4): the translation on each axis
        (x, y, z).

    Returns
    -------
    np.ndarray
        ArrayLike of shape (N, ...): the translated coordinates.

    See Also
    --------
    ktk.geometry.rotate, ktk.geometry.scale, ktk.geometry.mirror

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
        ArrayLike of shape (N, ...): the coordinates to scale.

    scales
        ArrayLike of shape (N, ) that corresponds to the scale to apply
        uniformly on the three axes.

    Returns
    -------
    np.ndarray
        ArrayLike of shape (N, ...): the scaled coordinates.

    See Also
    --------
    ktk.geometry.rotate, ktk.geometry.translate, ktk.geometry.mirror

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


def mirror(coordinates, /, axis: str = "z"):
    """
    Mirror a series of coordinates.

    Parameters
    ----------
    coordinates
        ArrayLike of shape (N, ...): the coordinates to mirror.

    axis
        Can be either "x", "y" or "z". The axis to mirror through. The default
        is "z".

    Returns
    -------
    np.ndarray
        ArrayLike of shape (N, ...): the mirrored coordinates.

    See Also
    --------
    ktk.geometry.rotate, ktk.geometry.translate, ktk.geometry.scale

    Examples
    --------
    Mirror the point (1, 2, 3) along the x, y and z axes respectively:

        >>> import kineticstoolkit.lab as ktk
        >>> import numpy as np
        >>> p = np.array([[1.0, 2.0, 3.0, 1.0]])

        >>> ktk.geometry.mirror(p, "x")
        array([[-1., 2., 3., 1.]])

        >>> ktk.geometry.mirror(p, "y")
        array([[ 1., -2., 3., 1.]])

        >>> ktk.geometry.mirror(p, "z")
        array([[ 1., 2., -3., 1.]])

    """
    check_param("axis", axis, str)

    retval = np.array(coordinates)
    if axis == "x":
        retval[:, 0] *= -1
    elif axis == "y":
        retval[:, 1] *= -1
    elif axis == "z":
        retval[:, 2] *= -1
    else:
        raise ValueError("axis must be either 'x', 'y' or 'z'")

    return retval


# %% Data extraction


def get_angles(
    T: ArrayLike, seq: str, degrees: bool = False, flip: bool = False
) -> np.ndarray:
    """
    Extract Euler angles from a transform series.

    In case of gimbal lock, a warning is raised, and the third angle is set to
    zero. Note however that the returned angles still represent the correct
    rotation.

    Parameters
    ----------
    T
        An Nx4x4 transform series.

    seq
        Specifies the sequence of axes for successive rotations. Up to 3
        characters belonging to the set {"X", "Y", "Z"} for intrinsic
        rotations (moving axes), or {"x", "y", "z"} for extrinsic rotations
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
    The range of the returned angles is dependent on the `flip` parameter. If
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
    T = np.array(T)
    check_param("seq", seq, str)
    check_param("degrees", degrees, bool)
    check_param("flip", flip, bool)

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


def get_quaternions(
    T: ArrayLike, canonical: bool = False, scalar_first: bool = False
) -> np.ndarray:
    """
    Extract quaternions from a transform series.

    Parameters
    ----------
    T
        An Nx4x4 transform series.

    canonical
        Whether to map the redundant double cover of rotation space to a
        unique "canonical" single cover. If True, then the quaternion is
        chosen from {q, -q} such that the w term is positive. If the w term is
        0, then the quaternion is chosen such that the first nonzero term of
        the x, y, and z terms is positive. Default is False.

    scalar_first
        Optional. If True, the quaternion order is (w, x, y, z). If False,
        the quaternion order is (x, y, z, w). Default is False.

    Returns
    -------
    np.ndarray
        An Nx4 series of quaternions.

    """
    T = np.array(T)
    check_param("scalar_first", scalar_first, bool)

    _check_no_skewed_rotation(T, "T")

    R = transform.Rotation.from_matrix(T[:, 0:3, 0:3])
    return R.as_quat(canonical=canonical, scalar_first=scalar_first)


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
        An Nx4x4 transform series that represents the local coordinate system.

    Returns
    -------
    np.ndarray
        Series of local coordinates in the same shape as
        `global_coordinates`.

    See Also
    --------
    ktk.geometry.get_global_coordinates

    """
    global_coordinates_array = np.array(global_coordinates)
    reference_frames_array = np.array(reference_frames)

    _check_no_skewed_rotation(reference_frames_array, "reference_frames")

    (global_coordinates_array, reference_frames_array) = _match_size(
        global_coordinates_array, reference_frames_array
    )

    # Transform NaNs in global coordinates to zeros to perform the operation,
    # then put back NaNs in the corresponding local coordinates.
    nan_index = np.isnan(global_coordinates_array)
    global_coordinates_array[nan_index] = 0

    # Invert the reference frame to obtain the inverse transformation
    inv_ref_T = invert(reference_frames_array)

    local_coordinates = np.zeros(global_coordinates_array.shape)  # init
    local_coordinates = matmul(inv_ref_T, global_coordinates_array)

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
        An Nx4x4 transform series that represents the local coordinate system.

    Returns
    -------
    np.ndarray
        Series of global coordinates in the same shape as
        `local_coordinates`.

    See Also
    --------
    ktk.geometry.get_local_coordinates

    """
    local_coordinates_array = np.array(local_coordinates)
    reference_frames_array = np.array(reference_frames)

    _check_no_skewed_rotation(reference_frames_array, "reference_frames")

    (local_coordinates_array, reference_frames_array) = _match_size(
        local_coordinates_array, reference_frames_array
    )

    global_coordinates = np.zeros(local_coordinates_array.shape)
    global_coordinates = matmul(
        reference_frames_array, local_coordinates_array
    )
    return global_coordinates


def get_distances(
    point_series1: ArrayLike, point_series2: ArrayLike, /
) -> np.ndarray:
    """
    Calculate the euclidian distance between two point series.

    Parameters
    ----------
    point_series1, point_series2
        Series of N points, as an Nx4 array-like of the form
        [[x, y, z, 1.0], ...]

    Returns
    -------
    np.ndarray
        Series of euclidian distance, as an Nx4 array of the form
        [[x, y, z, 0.0], ...]

    """
    point_series1 = np.array(point_series1)
    point_series2 = np.array(point_series2)

    return np.sqrt(
        np.sum(
            (point_series1 - point_series2) ** 2,
            axis=1,
        )
    )


# %% "is" functions


def isnan(array: ArrayLike, /) -> np.ndarray:
    """
    Check which samples contain at least one NaN.

    Parameters
    ----------
    array
        Array where the first dimension corresponds to time.

    Returns
    -------
    np.ndarray
        Array of bool that is the same size of input's first dimension, with
        True for the samples that contain at least one NaN.

    """
    temp = np.isnan(array)
    while len(temp.shape) > 1:
        temp = temp.sum(axis=1) > 0
    return temp


def is_transform_series(array: ArrayLike, /) -> bool:
    """
    Check that the input is an Nx4x4 series of homogeneous transforms.

    Parameters
    ----------
    array
        Array where the first dimension corresponds to time.

    Returns
    -------
    bool
        True if every sample (other than NaNs) of the input array is a
        4x4 homogeneous transform.

    """
    value = np.array(array)

    # Check the dimension
    if len(value.shape) != 3:
        return False
    if value.shape[1:] != (4, 4):
        return False

    # Check that we don't only have NaNs
    index = ~isnan(value)
    if np.sum(index) == 0:
        return False

    # Check the last line
    if not np.allclose(value[index, 3], [0.0, 0.0, 0.0, 1.0]):
        return False

    # Check that the rotation is not skewed
    try:
        _check_no_skewed_rotation(value, "value")
    except ValueError:
        return False

    return True


def _is_point_vector_series(array: ArrayLike, last_element: float) -> bool:
    """Check if the input is a `kind` series."""
    value = np.array(array)

    # Check the dimension
    if len(value.shape) != 2:
        return False
    if value.shape[1:] != (4,):
        return False

    # Check that we don't only have NaNs
    index = ~isnan(value)
    if np.sum(index) == 0:
        return False

    # Check the last line
    if not np.allclose(value[index, 3], last_element):
        return False

    return True


def is_point_series(array: ArrayLike) -> bool:
    """
    Check that the input is an Nx4 point series ([[x, y, z, 1.0], ...]).

    Parameters
    ----------
    array
        Array where the first dimension corresponds to time.

    Returns
    -------
    bool
        True if every sample (other than NaNs) of the input array is a point
        (an array of length 4 with the last component being 1.0)

    """
    return _is_point_vector_series(array, 1.0)


def is_vector_series(array: ArrayLike) -> bool:
    """
    Check that the input is an Nx4 vector series ([[x, y, z, 0.0], ...]).

    Parameters
    ----------
    array
        Array where the first dimension corresponds to time.

    Returns
    -------
    bool
        True if every sample (other than NaNs) of the input array is a vector
        (an array of length 4 with the last component being 0.0)

    """
    return _is_point_vector_series(array, 0.0)


# %% create_point_series, create_vector_series


def _single_input_to_point_vector_series(
    array: ArrayLike, last_element: float, length: int | None = None
) -> np.ndarray:
    """Implement of to_point_series and to_vector_series."""
    array = np.array(array, dtype=float)

    if len(array.shape) == 0:
        raise ValueError(
            f"The input must be an array, however a value of {array} was "
            "provided."
        )
    elif len(array.shape) > 2:
        raise ValueError(
            "The shape of the input must have a maximum of two dimensions, "
            f"however it has {len(array.shape)} dimensions."
        )

    # Init output
    n_samples = array.shape[0]
    output = np.zeros((n_samples, 4))
    if last_element == 1.0:
        output[:, 3] = 1

    # Fill output
    if len(array.shape) == 1:  # (N,)
        output[:, 0] = array
    else:
        for i in range(min(3, array.shape[1])):
            output[:, i] = array[:, i]

    # Repeat to the requested number of samples if applicable
    if n_samples == 1 and length is not None:
        output = np.repeat(output, length, axis=0)

    return output


def _multiple_inputs_to_point_vector_series(
    x: ArrayLike | None,
    y: ArrayLike | None,
    z: ArrayLike | None,
    last_element: float,
    length: int | None = None,
) -> np.ndarray:

    # Multiple array form
    if length is not None:
        temp = np.zeros((length, 4))
    else:
        temp = np.zeros((1, 4))

    if x is not None:
        x = np.array(x)
        if len(x.shape) > 1:
            raise ValueError("x should have only one dimension.")
        try:
            x, temp = _match_size(x, temp)
        except ValueError:
            raise ValueError("x has an incorrect length.")
        else:
            temp[:, 0] = x

    if y is not None:
        y = np.array(y)
        if len(y.shape) > 1:
            raise ValueError("y should have only one dimension.")
        try:
            y, temp = _match_size(y, temp)
        except ValueError:
            raise ValueError("y has an incorrect length.")
        else:
            temp[:, 1] = y

    if z is not None:
        z = np.array(z)
        if len(z.shape) > 1:
            raise ValueError("z should have only one dimension.")
        try:
            z, temp = _match_size(z, temp)
        except ValueError:
            raise ValueError("z has an incorrect length.")
        else:
            temp[:, 2] = z

    if last_element == 1:
        temp[:, 3] = 1.0
    else:
        temp[:, 3] = 0.0

    return temp


def create_point_series(
    array: ArrayLike | None = None,
    *,
    x: ArrayLike | None = None,
    y: ArrayLike | None = None,
    z: ArrayLike | None = None,
    length: int | None = None,
) -> np.ndarray:
    """
    Create an Nx4 point series ([[x, y, z, 1.0], ...]).

    **Single array**

    To create a point series based on a single array, use this form::

        create_point_series(
            array: ArrayLike | None = None,
            *,
            length: int | None = None,
        ) -> np.ndarray:

    **Multiple arrays**

    To create a point series based on multiple arrays (e.g., x, y, z), use
    this form::

        create_point_series(
            *,
            x: ArrayLike | None = None,
            y: ArrayLike | None = None,
            z: ArrayLike | None = None,
            length: int | None = None,
        ) -> np.ndarray:

    Parameters
    ----------
    array
        Used in single array input form.
        Array of one of these shapes where N corresponds to time:
        (N,), (N, 1): forms a point series on the x axis, with y=0 and z=0.
        (N, 2): forms a point series on the x, y axes, with z=0.
        (N, 3), (N, 4): forms a point series on the x, y, z axes.

    x
        Used in multiple arrays input form.
        Optional. Array of shape (N,) that contains the x values. If not
        provided, x values are filled with zero.

    y
        Used in multiple arrays input form.
        Optional. Array of shape (N,) that contains the y values. If not
        provided, y values are filled with zero.

    z
        Used in multiple arrays input form.
        Optional. Array of shape (N,) that contains the z values. If not
        provided, z values are filled with zero.

    length
        The number of samples in the resulting point series. If there is only
        one sample in the original array, this one sample will be duplicated
        to match length. Otherwise, an error is raised if the input
        array does not match length.

    Returns
    -------
    array
        An Nx4 array with every sample being [x, y, z, 1.0].

    Raises
    ------
    ValueError
        If the inputs have incorrect dimensions.

    Examples
    --------
    Single input form::

        # A series of 2 samples with x, y defined
        >>> ktk.geometry.create_point_series([[1.0, 2.0], [4.0, 5.0]])
        array([[1., 2., 0., 1.],
               [4., 5., 0., 1.]])

        # A series of 2 samples with x, y, z defined
        >>> ktk.geometry.create_point_series([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        array([[1., 2., 3., 1.],
               [4., 5., 6., 1.]])

        # Samething
        >>> ktk.geometry.create_point_series([[1.0, 2.0, 3.0, 1.0], [4.0, 5.0, 6.0, 1.0]])
        array([[1., 2., 3., 1.],
               [4., 5., 6., 1.]])

    Multiple inputs form::

        # A series of 2 samples with x, z defined
        >>> ktk.geometry.create_point_series(x=[1.0, 2.0, 3.0], z=[4.0, 5.0, 6.0])
        array([[1., 0., 4., 1.],
               [2., 0., 5., 1.],
               [3., 0., 6., 1.]])

    """
    if array is not None:
        # Single array form
        array = np.array(array)
        if is_point_series(array) and (
            length is None or array.shape[0] == length
        ):
            # Nothing to do
            return array
        else:
            return _single_input_to_point_vector_series(
                array, last_element=1.0, length=length
            )

    else:
        return _multiple_inputs_to_point_vector_series(
            x=x, y=y, z=z, last_element=1.0, length=length
        )


def create_vector_series(
    array: ArrayLike | None = None,
    *,
    x: ArrayLike | None = None,
    y: ArrayLike | None = None,
    z: ArrayLike | None = None,
    length: int | None = None,
) -> np.ndarray:
    """
    Create an Nx4 vector series ([[x, y, z, 0.0], ...]).

    **Single array**

    To create a vector series based on a single array, use this form::

        create_vector_series(
            array: ArrayLike | None = None,
            *,
            length: int | None = None,
        ) -> np.ndarray:

    **Multiple arrays**

    To create a vector series based on multiple arrays (e.g., x, y, z), use
    this form::

        create_vector_series(
            *,
            x: ArrayLike | None = None,
            y: ArrayLike | None = None,
            z: ArrayLike | None = None,
            length: int | None = None,
        ) -> np.ndarray:

    Parameters
    ----------
    array
        Used in single array input form.
        Array of one of these shapes where N corresponds to time:
        (N,), (N, 1): forms a vector series on the x axis, with y=0 and z=0.
        (N, 2): forms a vector series on the x, y axes, with z=0.
        (N, 3), (N, 4): forms a vector series on the x, y, z axes.

    x
        Used in multiple arrays input form.
        Optional. Array of shape (N,) that contains the x values. If not
        provided, x values are filled with zero.

    y
        Used in multiple arrays input form.
        Optional. Array of shape (N,) that contains the y values. If not
        provided, y values are filled with zero.

    z
        Used in multiple arrays input form.
        Optional. Array of shape (N,) that contains the z values. If not
        provided, z values are filled with zero.

    length
        The number of samples in the resulting vector series. If there is only
        one sample in the original array, this one sample will be duplicated
        to match length. Otherwise, an error is raised if the input
        array does not match length.

    Returns
    -------
    array
        An Nx4 array with every sample being [x, y, z, 0.0].

    Raises
    ------
    ValueError
        If the inputs have incorrect dimensions.

    Examples
    --------
    Single input form::

        # A series of 2 samples with x, y defined
        >>> ktk.geometry.create_vector_series([[1.0, 2.0], [4.0, 5.0]])
        array([[1., 2., 0., 0.],
               [4., 5., 0., 0.]])

        # A series of 2 samples with x, y, z defined
        >>> ktk.geometry.create_vector_series([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        array([[1., 2., 3., 0.],
               [4., 5., 6., 0.]])

        # Samething
        >>> ktk.geometry.create_vector_series([[1.0, 2.0, 3.0, 1.0], [4.0, 5.0, 6.0, 1.0]])
        array([[1., 2., 3., 0.],
               [4., 5., 6., 0.]])

    Multiple inputs form::

        # A series of 2 samples with x, z defined
        >>> ktk.geometry.create_vector_series(x=[1.0, 2.0, 3.0], z=[4.0, 5.0, 6.0])
        array([[1., 0., 4., 0.],
               [2., 0., 5., 0.],
               [3., 0., 6., 0.]])

    """
    if array is not None:
        # Single array form
        array = np.array(array)
        if is_vector_series(array) and (
            length is None or array.shape[0] == length
        ):
            # Nothing to do
            return array
        else:
            return _single_input_to_point_vector_series(
                array, last_element=0.0, length=length
            )

    else:
        return _multiple_inputs_to_point_vector_series(
            x=x, y=y, z=z, last_element=0.0, length=length
        )


# %% create_transform_series

# -------------
# From matrices
# -------------


def _matrices_to_frame_series(matrices: ArrayLike) -> np.ndarray:
    """Implement matrix form of _to_frame_series (rotational part)."""
    if is_transform_series(matrices):
        # Nothing to do
        return np.array(matrices)
    else:
        matrices_array = np.array(matrices)
        output = np.zeros((matrices_array.shape[0], 4, 4))
        output[:, 0:3, 0:3] = matrices_array[:, 0:3, 0:3]
        output[:, 3, 3] = 1

        if not np.all(isnan(output)) and not is_transform_series(output):
            raise ValueError(
                "The provided matrices are not a series of rotation matrices "
                "or homogeneous transforms."
            )
        return output


def _angles_to_frame_series(
    *,
    angles: ArrayLike,
    seq: str | None = None,
    degrees: bool = False,
) -> np.ndarray:
    """Implement angles form of to_frame_series (rotational part)."""
    # Condition angles
    angles_array = np.array(angles)
    n_samples = angles_array.shape[0]

    # Create the rotation matrix
    rotation = transform.Rotation.from_euler(seq, angles_array, degrees)
    R = rotation.as_matrix()
    if len(R.shape) == 2:  # Single rotation: add the Time dimension.
        R = R[np.newaxis, ...]

    # Construct and return the output
    output = np.zeros((n_samples, 4, 4))
    output[:, 0:3, 0:3] = R
    output[:, 3, 3] = 1.0
    return output


def _quaternions_to_frame_series(
    *,
    quaternions: ArrayLike,
    scalar_first: bool = False,
) -> np.ndarray:
    """Implement angles form of to_frame_series (rotational part)."""
    # Condition angles
    quaternions_array = np.array(quaternions)
    n_samples = quaternions_array.shape[0]

    # Create the rotation matrix
    rotation = transform.Rotation.from_quat(
        quaternions_array, scalar_first=scalar_first
    )
    R = rotation.as_matrix()
    if len(R.shape) == 2:  # Single rotation: add the Time dimension.
        R = R[np.newaxis, ...]

    # Construct and return the output
    output = np.zeros((n_samples, 4, 4))
    output[:, 0:3, 0:3] = R
    output[:, 3, 3] = 1.0
    return output


def _vectors_to_frame_series(
    x: ArrayLike | None = None,
    y: ArrayLike | None = None,
    z: ArrayLike | None = None,
    xy: ArrayLike | None = None,
    xz: ArrayLike | None = None,
    yz: ArrayLike | None = None,
):
    """Create a frame series from cross products, with a zero origin."""

    def normalize(v):
        """Normalize series of vectors."""
        if not is_vector_series(v):
            if is_point_series(v):
                v[:, 3] = 0.0  # Convert to a vector series
            else:
                raise ValueError(
                    "At least one of the provided vectors series is not a "
                    "vector series."
                )
        norm = np.linalg.norm(v, axis=1)
        return v / norm[..., np.newaxis]

    def cross(v1, v2):
        """Cross on series of vectors."""
        c = v1.copy()
        c[:, 0:3] = np.cross(v1[:, 0:3], v2[:, 0:3])
        return c

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

    return np.stack(
        (
            v_x,
            v_y,
            v_z,
            create_point_series([[0.0, 0.0, 0.0]], length=v_x.shape[0]),
        ),
        axis=2,
    )


def create_transform_series(
    matrices: ArrayLike | None = None,
    *,
    angles: ArrayLike | None = None,
    seq: str | None = None,
    degrees: bool = False,
    quaternions: ArrayLike | None = None,
    scalar_first: bool = False,
    x: ArrayLike | None = None,
    y: ArrayLike | None = None,
    z: ArrayLike | None = None,
    xy: ArrayLike | None = None,
    xz: ArrayLike | None = None,
    yz: ArrayLike | None = None,
    positions: ArrayLike | None = None,
    length: int | None = None,
) -> np.ndarray:
    """
    Create an Nx4x4 transform series from multiple input forms.

    **Matrix input form**

    If the input is a series of 3x3 rotation matrices or 4x4 homogeneous
    transforms, use this form::

        ktk.geometry.to_transform_series(
            matrices: ArrayLike,
            *,
            positions: ArrayLike | None = None,
            length: int | None = None,
            ) -> np.ndarray

    **Angle input form**

    If the input is a series of Euler/cardan angles, use this form::

        ktk.geometry.to_transform_series(
            *,
            angles: ArrayLike,
            seq: str,
            degrees: bool = False,
            positions: ArrayLike | None = None,
            length: int | None = None,
            ) -> np.ndarray

    **Quaternion input form**

    If the input is a series of quaternions, use this form::

        ktk.geometry.to_transform_series(
            *,
            quaternions: ArrayLike,
            scalar_first: bool = False,
            positions: ArrayLike | None = None,
            length: int | None = None,
            ) -> np.ndarray

    **Vector input form (using cross-product)**

    To create transform series that represent a local coordinate system based
    on the cross product of different vectors, use this form, where one of
    {x, y, z} and one of {xy, xz, yz} must be defined::

        ktk.geometry.to_transform_series(
            *,
            x: ArrayLike | None = None,
            y: ArrayLike | None = None,
            z: ArrayLike | None = None,
            xy: ArrayLike | None = None,
            xz: ArrayLike | None = None,
            yz: ArrayLike | None = None,
            positions: ArrayLike | None = None,
            length: int | None = None,
            ) -> np.ndarray

    With this input form, x, y or z sets the first axis of the local coordinate
    system. Then, xy, xz or yz forms a plane with the first vector; the second
    axis is the cross product of both vectors (perpendicular to this plane).
    Finally, the third axis is the cross product of the two first axes.

    Parameters
    ----------
    matrices
        Used in the matrix input form.
        Nx3x3 series or rotations or Nx4x4 series of homogeneous transforms.

    angles
        Used in the angles input form.
        Series of angles, either of shape (N,) or (N, 1) for rotations around
        only one axis, or (N, 2) or (N, 3) for rotations around consecutive
        axes.

    seq
        Used in the angles input form.
        Specifies the sequence of axes for successive rotations. Up to 3
        characters belonging to {"X", "Y", "Z"} for intrinsic rotations
        (moving axes), or {"x", "y", "z"} for extrinsic rotations (fixed
        axes). Extrinsic and intrinsic rotations cannot be mixed in one
        function call.

    degrees
        Used in the angles input form.
        Optional. If True, then the given angles are in degrees, otherwise
        they are in radians. Default is False (radians).

    quaternions
        Used in the quaternions input form. Nx4 series of quaternions.

    scalar_first
        Used in the quaternions input form.
        Optional. If True, the quaternion order is (w, x, y, z). If False,
        the quaternion order is (x, y, z, w). Default is False.

    x, y, z
        Used in the vector input form.
        Define either `x`, `y` or `z`. A series of N vectors (Nx4) that
        define the {x|y|z} axis of the frames to be created.

    xy
        Used in the vector input form.
        Only if `x` or `y` is specified. A series of N vectors (Nx4) in the xy
        plane, to create `z` using (x cross xy) or (xy cross y). Choose vectors
        that point roughly in the +x or +y direction.

    xz
        Used in the vector input form.
        Only if `x` or `z` is specified. A series of N vectors (Nx4) in the xz
        plane, to create `y` using (xz cross x) or (z cross xz). Choose vectors
        that point roughly in the +x or +z direction.

    yz
        Used in the vector input form.
        Only if `y` or `z` is specified. A series of N vectors (Nx4) in the yz
        plane, to create `x` using (y cross yz) or (yz cross z). Choose vectors
        that point roughly in the +y or +z direction.

    positions
        Optional. An Nx2, Nx3 or Nx4 point series that defines the position
        component (fourth column) of the transforms. Default value is
        [[0.0, 0.0, 0.0, 1.0]]. If the input is an Nx4x4 frame series and
        therefore already has positions, then the existing positions are kept
        unless `positions` is specified.

    length
        Optional. The number of samples in the resulting series. If there
        is only one sample in the original array, this one sample will be
        duplicated to match length. Otherwise, an error is raised if the input
        array does not match length.

    Returns
    -------
    np.ndarray
        An Nx4x4 transform series.

    Examples
    --------
    **Matrix input**

    Convert a 2x3x3 rotation matrix series and a 1x4 position series to
    an 2x4x4 homogeneous transform series:

    >>> positions = [[0.5, 0.6, 0.7]]
    >>> rotations = [[[ 1.,  0.,  0.],
    ...              [ 0.,  1.,  0.],
    ...              [ 0.,  0.,  1.]],
    ...             [[ 1.,  0.,  0.],
    ...              [ 0.,  0., -1.],
    ...              [ 0.,  1.,  0.]]]
    >>> ktk.geometry.create_transform_series(rotations, positions=positions)
    array([[[ 1. ,  0. ,  0. ,  0.5],
            [ 0. ,  1. ,  0. ,  0.6],
            [ 0. ,  0. ,  1. ,  0.7],
            [ 0. ,  0. ,  0. ,  1. ]],
    <BLANKLINE>
           [[ 1. ,  0. ,  0. ,  0.5],
            [ 0. ,  0. , -1. ,  0.6],
            [ 0. ,  1. ,  0. ,  0.7],
            [ 0. ,  0. ,  0. ,  1. ]]])

    **Angle input**

    Create a series of two homogeneous transforms that rotates 0, then 90
    degrees around x:

    >>> ktk.geometry.create_transform_series(angles=[0, 90], seq="x", degrees=True)
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
    # Check params
    if seq is not None:
        check_param("seq", seq, str)
    check_param("degrees", degrees, bool)
    check_param("scalar_first", scalar_first, bool)
    if length is not None:
        check_param("length", length, int)

    # Form the rotational part using the correct implementation
    if matrices is not None:
        output = _matrices_to_frame_series(matrices)
    elif angles is not None:
        output = _angles_to_frame_series(
            angles=angles, seq=seq, degrees=degrees
        )
    elif quaternions is not None:
        output = _quaternions_to_frame_series(
            quaternions=quaternions, scalar_first=scalar_first
        )
    elif x is not None or y is not None or z is not None:
        output = _vectors_to_frame_series(x=x, y=y, z=z, xy=xy, xz=xz, yz=yz)
    else:
        raise ValueError("Insufficient parameters.")

    # Match length and add origin
    if length is not None:
        if output.shape[0] == 1:
            output = np.repeat(output, length, axis=0)
        elif output.shape[0] != length:
            raise ValueError(
                f"The provided input must have a length of 1 or {length}"
                f"but it has a length of {output.shape[0]}."
            )

    # Add origin if needed
    if (
        matrices is not None
        and is_transform_series(matrices)
        and positions is None
    ):
        # This was already a frame series and we don't want to set the origin.
        return output

    # In any other case, set the origin.
    if positions is not None:
        try:
            positions = create_point_series(positions, length=length)
        except ValueError as e:
            raise ValueError(f"Parameter positions is invalid: {e}")
    else:
        try:
            positions = create_point_series(
                [[0.0, 0.0, 0.0, 1.0]], length=length
            )
        except ValueError as e:
            raise ValueError(f"Parameter positions is invalid: {e}")

    output[:, :, 3] = positions
    return output


# %% Point registration
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


# %% Private functions


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
    Check if all rotation matrices are orthogonal.

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
        series33 = series[:, 0:3, 0:3]
        if not np.allclose(
            matmul(series33, series33.transpose([0, 2, 1]))[~isnan(series)],
            np.eye(3),
        ):
            raise ValueError(
                f"Parameter {param_name} contains at least one rotation "
                "component that is not orthogonal. This may happen, for "
                "instance, if you attempted to average, resample, or filter a "
                "homogeneous transform, which is usually forbidden. If this "
                "is the case, then consider filtering quaternions or Euler "
                "angles instead. If you created a homogeneous transform from "
                "3D points, then average/resample/filter the "
                "point trajectories before creating the transform, instead "
                "of averaging/resampling/filtering the transform."
            )


# %% To deprecate in version 1.0


def create_transforms(
    seq: str | None = None,
    angles: ArrayLike | None = None,
    translations: ArrayLike | None = None,
    scales: ArrayLike | None = None,
    *,
    degrees=False,
) -> np.ndarray:
    """
    Create series of transforms based on angles, translations and scales.

    Create an Nx4x4 series of homogeneous transform matrices based on series of
    angles, translations and scales.

    Warning
    -------
    This function will be deprecated in Version 1.0. It is recommended to use
    create_transform_series instead.

    Parameters
    ----------
    seq
        Optional. Specifies sequence of axes for rotations. Up to 3 characters
        belonging to the set {"X", "Y", "Z"} for intrinsic rotations (moving
        axes), or {"x", "y", "z"} for extrinsic rotations (fixed axes).
        Extrinsic and intrinsic rotations cannot be mixed in one function call.
        Required if angles is specified.

    angles
        Optional ArrayLike of shape (N,) or (N, [1 or 2 or 3]). Angles are
        specified in radians (if degrees is False) or degrees (if degrees is
        True).

        For a single-character `seq`, `angles` can be:

        - ArrayLike with shape (N,), where each `angle[i]` corresponds to a
          single rotation;
        - ArrayLike with shape (N, 1), where each `angle[i, 0]` corresponds
          to a single rotation.

        For 2- and 3-character `seq`, `angles` is an ArrayLike with shape
        (N, W) where each `angle[i, :]` corresponds to a sequence of Euler
        angles and W is the length of `seq`.

    translations
        Optional ArrayLike of shape (N, 3) or (N, 4). This corresponds
        to the translation part of the generated series of homogeneous
        transforms.

    scales
        Optional ArrayLike of shape (N, ) that corresponds to the scale to
        apply uniformly on the three axes. By default, no scale is included.

    degrees
        If True, then the given angles are in degrees. Default is False.

    Returns
    -------
    np.ndarray
        An Nx4x4 series of homogeneous transforms.

    See Also
    --------
    ktk.geometry.create_frames, ktk.geometry.rotate, ktk.geometry.translate,
    ktk.geometry.scale

    Examples
    --------
    Create a series of two homogeneous transforms that rotates 0, then 90
    degrees around x:

        >>> import kineticstoolkit.lab as ktk
        >>> ktk.geometry.create_transforms(seq="x", angles=[0, 90], degrees=True)
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
    check_param("seq", seq, (str, None))
    check_param("degrees", degrees, bool)

    # Condition translations
    if translations is None:
        translations_array = np.zeros((1, 3))
    else:
        translations_array = np.array(translations)

    # Condition angles
    if angles is None:
        angles_array = np.array([0])
        seq = "x"
    else:
        angles_array = np.array(angles)

    # Condition scales
    if scales is None:
        scales_array = np.array([1])
    else:
        scales_array = np.array(scales)

    # Convert scales to a series of scaling matrices
    temp = np.zeros((scales_array.shape[0], 4, 4))
    temp[:, 0, 0] = scales_array
    temp[:, 1, 1] = scales_array
    temp[:, 2, 2] = scales_array
    temp[:, 3, 3] = 1.0
    scales_array = temp

    # Match sizes
    translations_array, angles_array = _match_size(
        translations_array, angles_array
    )
    translations_array, scales_array = _match_size(
        translations_array, scales_array
    )
    translations_array, angles_array = _match_size(
        translations_array, angles_array
    )
    n_samples = angles_array.shape[0]

    # Create the rotation matrix
    rotation = transform.Rotation.from_euler(seq, angles_array, degrees)
    R = rotation.as_matrix()
    if len(R.shape) == 2:  # Single rotation: add the Time dimension.
        R = R[np.newaxis, ...]

    # Construct the final series of transforms (without scaling)
    T = np.empty((n_samples, 4, 4))
    T[:, 0:3, 0:3] = R
    T[:, 0:3, 3] = translations_array
    T[:, 3, 0:3] = 0
    T[:, 3, 3] = 1

    # Return the scaling + transform
    return T @ scales_array


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

    Warning
    -------
    This function will be deprecated in Version 1.0. It is recommended to use
    create_transform_series instead.

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

    See Also
    --------
    ktk.geometry.create_transforms

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


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
