"""
ktk.geometry
============
This module contains functions related to 3D geometry and linear algebra
related to biomechanics.

Author : Felix Chenier
Date : December 2019
"""

import numpy as np

def get_local_coordinates(global_coordinates, reference_frame):
    """
    Express global coordinates in a local reference frame.

    Parameters
    ----------
    global_coordinates : np.array with 3 dimensions
        The global coordinates, as either:
            - A series of N clouds of M points or vectors, or
            - A series of N 4x4 transformation matrices.
        For example:
            - A point or vector : shape = (1, 4, 1)
            - A cloud of M points or vectors : shape = (1, 4, M)
            - A series of N points or vectors : shape = (N, 4, 1)
            - A series of N clouds of M points or vectors: shape = (N, 4, M)
            - A series of N 4x4 transformation matrices : shape = (N, 4, 4)

    reference_frame : np.array with 3 dimensions
        The reference frame in which the local coordinates will be expressed.
        It can be either:
            - A reference frame : shape = (1, 4, 4)
            - A series of N reference frames : shape = (N, 4, 4)

    Returns
    -------
    np.array with 3 dimensions
        The local coordinates, with the dimensions that adapt to
        global_coordinates and reference_frame:
        Dimension 0 = Time,
        Dimension 1 = Line or x/y/z
        Dimension 2 = Column or i_point or i_vector
    """
    global_coordinates = global_coordinates.copy()

    # Invert the reference frame to obtain the inverse transformation
    ref_rot = reference_frame[:, 0:3, 0:3]
    ref_t = reference_frame[:, 0:3, 3]

    inv_ref_rot = np.transpose(ref_rot, (0, 2, 1))
    inv_ref_t = inv_ref_rot @ ref_t

    inv_ref_T = global_coordinates.copy()  # init
    inv_ref_T[:, 0:3, 0:3] = inv_ref_rot
    inv_ref_T[:, 0:3, 3] = inv_ref_t

    local_coordinates = inv_ref_T @ global_coordinates

    return local_coordinates
