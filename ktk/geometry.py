"""
ktk.geometry
============
This module contains functions related to 3D geometry and linear algebra
related to biomechanics.

Author : Felix Chenier
Date : December 2019
"""

import numpy as np


def create_reference_frame(global_points, method='ocx1'):
    """
    Create a reference frame based on a point cloud.

    Create a reference frame based on global points, using the provided
    method, and returns this reference frame as a transformation matrix
    or series of transformation matrices.

    Parameters
    ----------
    global_points : array
        - A set of M points in a global reference frame (4xM) or
        - A series of N sets of M points in a global reference frame (Nx4xM).

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

        'o1z2' : A reference frame based on lateral and anterior
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
    array : Transformation matrix (4x4) or series of transformation matrices
        (Nx4x4).
    """

    # Transform to a series
    original_size = len(global_points.shape)
    if original_size == 2:
        global_points = global_points[np.newaxis, :, :]

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

    if original_size == 2:
        # Convert back to a simple matrix (not a series of matrices)
        T = T[0, :, :]

    return T


def get_local_coordinates(global_coordinates, reference_frame):
    """
    Express global coordinates in a local reference frame.

    Parameters
    ----------
    global_coordinates : array of 2 or 3 dimensions
        The global coordinates, as either a matrix or series of N matrices.
        The matrices could be either transformation matrices (4x4) or (Nx4x4),
        vectors (4x1) or (Nx4x1) or sets of M vectors (4xM) or (Nx4xM).
        For example:
            - A point or vector : 4x1
            - A set of M points or vectors : 4xM
            - A transformation matrix : 4x4
            - A series of N points or vectors : Nx4x1
            - A series of N set of M points or vectors : Nx4xM
            - A series of N 4x4 transformation matrices : Nx4x4

    reference_frame : array of 2 or 3 dimensions
        The reference frame in which the local coordinates will be expressed.
        It can be either:
            - A reference frame : 4x4
            - A series of N reference frames : Nx4x4

    Returns
    -------
    array of 2 or 3 dimensions
        The local coordinates, with the dimensions that adapt to
        global_coordinates and reference_frame.
    """
    (_global_coordinates, _reference_frame) = _match_size(
            global_coordinates, reference_frame)

    n_samples = _global_coordinates.shape[0]

    # Invert the reference frame to obtain the inverse transformation
    ref_rot = _reference_frame[:, 0:3, 0:3]
    ref_t = _reference_frame[:, 0:3, 3]

    # Inverse rotation : transpose.
    inv_ref_rot = np.transpose(ref_rot, (0, 2, 1))

    # Inverse translation : we inverse-rotate the translation.
    inv_ref_t = np.zeros(ref_t.shape)
    for i_sample in range(n_samples):
        inv_ref_t[i_sample] = inv_ref_rot[i_sample] @ -ref_t[i_sample]

    inv_ref_T = np.zeros((n_samples, 4, 4))  # init
    inv_ref_T[:, 0:3, 0:3] = inv_ref_rot
    inv_ref_T[:, 0:3, 3] = inv_ref_t
    inv_ref_T[:, 3, 3] = np.ones(n_samples)

    local_coordinates = np.zeros(_global_coordinates.shape)  # init
    for i_sample in range(n_samples):
        local_coordinates[i_sample] = \
                inv_ref_T[i_sample] @ _global_coordinates[i_sample]

    if len(global_coordinates.shape) == 2 and len(reference_frame.shape) == 2:
        local_coordinates = local_coordinates[0, :, :]

    return local_coordinates


def get_global_coordinates(local_coordinates, reference_frame):
    """
    Express local coordinates in a global reference frame.

    Parameters
    ----------
    local_coordinates : array of 2 or 3 dimensions
        The local coordinates, as either a matrix or series of N matrices.
        The matrices could be either transformation matrices (4x4) or (Nx4x4),
        vectors (4x1) or (Nx4x1) or sets of M vectors (4xM) or (Nx4xM).
        For example:
            - A point or vector : 4x1
            - A set of M points or vectors : 4xM
            - A transformation matrix : 4x4
            - A series of N points or vectors : Nx4x1
            - A series of N set of M points or vectors : Nx4xM
            - A series of N 4x4 transformation matrices : Nx4x4

    reference_frame : array of 2 or 3 dimensions
        The reference frame in which the local coordinates will be expressed.
        It can be either:
            - A reference frame : 4x4
            - A series of N reference frames : Nx4x4

    Returns
    -------
    array of 2 or 3 dimensions
        The global coordinates, with the dimensions that adapt to
        local_coordinates and reference_frame.
    """
    (_local_coordinates, _reference_frame) = _match_size(
            local_coordinates, reference_frame)

    n_samples = _local_coordinates.shape[0]

    global_coordinates = np.zeros(_local_coordinates.shape)
    for i_sample in range(n_samples):
        global_coordinates[i_sample] = \
                _reference_frame[i_sample] @ _local_coordinates[i_sample]

    if len(local_coordinates.shape) == 2 and len(reference_frame.shape) == 2:
        global_coordinates = global_coordinates[0, :, :]

    return global_coordinates


def _match_size(op1, op2):
    """
    Match the time size so that op1 and op2 have the same time dimension.

    This is a by-copy operation, the inputs are not modified.

    Parameters
    ----------
    op1, op2 : array or shape 4xM or Nx4xM

    Returns
    -------
    (op1, op2) where op1 and op2 are both of shape Nx4xM with matching N.
    """
    op1 = op1.copy()
    op2 = op2.copy()

    if len(op1.shape) == 2:
        op1 = op1[np.newaxis, :, :]

    if len(op2.shape) == 2:
        op2 = op2[np.newaxis, :, :]

    if op1.shape[0] == 1:
        op1 = np.repeat(op1, op2.shape[0], axis=0)

    if op2.shape[0] == 1:
        op2 = np.repeat(op2, op1.shape[0], axis=0)

    if op1.shape[0] != op2.shape[0]:
        raise ValueError(
                'Could not make op1 and op2 of matching first dimension.')

    return (op1, op2)
