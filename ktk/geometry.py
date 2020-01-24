"""
ktk.geometry
============
This module contains functions related to 3D geometry and linear algebra
related to biomechanics. The first dimension of every array is always N and
corresponds to time. For constants, use a length of 1 as the first dimension.

Author : Felix Chenier
Date : December 2019
"""

import numpy as np


def matmul(op1, op2):
    """
    Matrix multiplication between series of matrices.

    This function is a wrapper for numpy's matmul function (operator @), that
    helps numpy to understand ktk's convention that every point or vector is
    always expressed as a series (first dimension is time).

    It aligns and create additionnal dimensions if needed to avoid dimension
    mismatch errors.

    Parameters
    ----------
    op1, op2 : array
        Series of floats, vectors or matrices.

    Returns
    -------
    array : The product, as a series of Nx4 or Nx4xM matrices.
    """
    def perform_mul(op1, op2):
        if isinstance(op1, np.ndarray) and isinstance(op2, np.ndarray):
            return op1 @ op2
        else:
            return op1 * op2  # In the case where we have a series of floats.

    # Match the time size of each input
    if op1.shape[0] == 1:
        op1 = np.repeat(op1, op2.shape[0], axis=0)

    if op2.shape[0] == 1:
        op2 = np.repeat(op2, op1.shape[0], axis=0)

    if op1.shape[0] != op2.shape[0]:
        raise ValueError(
                'Could not match first dimension of op1 and op2')

    n_samples = op1.shape[0]

    # Get the expected shape by performing the first multiplication
    temp = perform_mul(op1[0], op2[0])
    result = np.empty((n_samples, *temp.shape))

    # Perform the multiplication
    for i_sample in range(n_samples):
        result[i_sample] = perform_mul(op1[i_sample], op2[i_sample])

    return result


def create_rotation_matrices(axis, angles):
    """
    Create a Nx4x4 series of rotation matrices around a given axis.

    Parameters
    ----------
    axis : str
        Can be either 'x', 'y' or 'z'.

    angle : array
        Series of angles in radians.

    Returns
    -------
    A Nx4x4 series of rotation matrices.
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


def create_reference_frames(global_points, method='ocx1'):
    """
    Create a Nx4x4 series of reference frames based on a point cloud series.

    Create reference frames based on global points, using the provided
    method, and returns this series of reference frames as a series of
    transformation matrices.

    Parameters
    ----------
    global_points : array
        A series of N sets of M points in a global reference frame (Nx4xM).

    method : str (optional)
        The method to use to construct a reference frame based on these points.
        Default is 'ocx1'.

        Available values are:

        'ocx1' : An arbitraty system that is useful for marker clusters
        (default).
        - Origin = Centroid of all points;
        - X is directed toward the first point;
        - Z is normal to the plane formed by the origin and both
        first points;
        - Y is the vectorial product of X and Z.

        'o1z2' : Reference frame based on lateral and anterior
        vectors.
        - Origin = First point;
        - Z is directed toward the second point (right);
        - Y is normal to the plane formed by the three markers (up);
        - X is the vectorial product of Y and Z.

        'isb-humerus' : Reference frame of the humerus as suggested by
        the International Society of Biomechanics [1], when the elbow is
        flexed by 90 degrees and the forearm is in complete pronation.
        - Origin = GH joint;
        - X is directed forward;
        - Y is directed upward;
        - Z is directed to the right;
        - Point 1 = GH joint;
        - Point 2 = Elbow center;
        - Point 3 = Ulnar styloid.

        'isb-forarm' : Reference frame of the forearm as suggested by
        the International Society of Biomechanics [1].
        - Origin = Ulnar Styloid;
        - X is directed forward in anatomic position;
        - Y is directed upward in anatomic position;
        - Z is directed to the right in anatomic position;
        - Point 1 = Ulnar Styloid;
        - Point 2 = Elbow center;
        - Point 3 = Radial styloid.

    [1] G. Wu et al., "ISB recommendation on definitions of joint
    coordinate systems of various joints for the reporting of human joint
    motion - Part II: shoulder, elbow, wrist and hand," Journal of
    Biomechanics, vol. 38, no. 5, pp. 981--992, 2005.


    Returns
    -------
    array : Series of transformation matrices (Nx4x4).
    """
    def normalize(v):

        norm = np.linalg.norm(v, axis=1)
        return v / norm[..., np.newaxis]

    if method == 'ocx1':
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

    elif method == 'o1z2':
        # Origin
        origin = global_points[:, 0:3, 0]
        # Z axis
        z = global_points[:, 0:3, 1] - origin
        z = normalize(z)
        # X plane
        x_temp = global_points[:, 0:3, 2] - origin
        x_temp = normalize(x_temp)
        # Y axis
        y = np.cross(z, x_temp)
        y = normalize(y)
        # X axis
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


def get_local_coordinates(global_coordinates, reference_frames):
    """
    Express global coordinates in local reference frames.

    Parameters
    ----------
    global_coordinates : array
        The global coordinates, as a series of N points, vectors or matrices.
        For example:
            - A series of N points or vectors : Nx4
            - A series of N set of M points or vectors : Nx4xM
            - A series of N 4x4 transformation matrices : Nx4x4

    reference_frames : array
        A series of N reference frames (Nx4x4) to express the global
        coordinates in.

    Returns
    -------
    array of 2 or 3 dimensions : The series of local coordinates.
    """
    n_samples = global_coordinates.shape[0]

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

    return local_coordinates


def get_global_coordinates(local_coordinates, reference_frames):
    """
    Express local coordinates in the global reference frame.

    Parameters
    ----------
    local_coordinates : array
        The local coordinates, as a series of N points, vectors or matrices.
        For example:
            - A series of N points or vectors : Nx4
            - A series of N set of M points or vectors : Nx4xM
            - A series of N 4x4 transformation matrices : Nx4x4

    reference_frames : array
        A series of N reference frames (Nx4x4) the local coordinates are
        expressed in.

    Returns
    -------
    array of 2 or 3 dimensions : The series of global coordinates.
    """
    global_coordinates = np.zeros(local_coordinates.shape)
    global_coordinates = matmul(reference_frames, local_coordinates)
    return global_coordinates
