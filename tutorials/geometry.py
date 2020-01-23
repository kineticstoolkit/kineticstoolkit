# %%
"""
geometry
========
This module allows the creation of reference frames, and simplifies passing
coordinates from local to global reference frames and vice-versa.

Points, vectors, sets of points, sets of vectors and matrices are all
expressed as 2-dimension arrays. By design, no component of this module is
expressed as 1-dimension arrays.

- Points are represented as 2d arrays of shape 4x1:

        [[x],
         [y],
         [z],
         [1]]

- Vectors are represented as 2d arrays of shape 4x1:

        [[x],
         [y],
         [z],
         [0]]

- Sets of M points are represented as 2d arrays of shape 4xM:

        [[x1, x2, x3, ..., xM],
         [y1, y2, y3, ..., yM],
         [z1, z2, z3, ..., zM],
         [1., 1., 1., ..., 1.]]

- Sets of M vectors are represented as 2d arrays of shape 4xM:

        [[x1, x2, x3, ..., xM],
         [y1, y2, y3, ..., yM],
         [z1, z2, z3, ..., zM],
         [0., 0., 0., ..., 0.]]

- Transformations matrices are represented as 2d arrays of shape 4x4:

        [[R11, R12, R13, Tx],
         [R21, R22, R23, Ty],
         [R31, R32, R33, Tz],
         [0. , 0. , 0. , 1.]]

Series of points, vectors, sets of points, sets of vectors and matrices are
all expressed as 3-dimension arrays, with the first dimension corresponding
to time.
"""
import ktk
import numpy as np

# %%
"""
Create a rotation matrix
------------------------
The function `ktk.geometry.create_rotation_matrix` allows creating 4x4 or
series or 4x4 matrices around a given axis. For example:
"""
T = ktk.geometry.create_rotation_matrix('x', 0)

# %%
"""
creates a null rotation matrix around the x axis (thus the identity matrix):
"""
T

# %% exclude
assert np.sum(np.abs(T - np.eye(4))) < 1E-15

# %%
T = ktk.geometry.create_rotation_matrix('x', np.pi/2)

# %%
"""
creates a rotation of 90 degrees around the x axis:
"""
T

# %% exclude
assert np.sum(np.abs(T - np.array([
       [1.,  0.,  0.,  0.],
       [0.,  0., -1.,  0.],
       [0.,  1.,  0.,  0.],
       [0.,  0.,  0.,  1.]]
       ))) < 1E-15

# %%
"""
and
"""
T = ktk.geometry.create_rotation_matrix('z', np.linspace(0, 2 * np.pi, 100))

# %%
"""
creates a series of 100 rotation matrices around the z axis, from 0 to
360 degrees:
"""
T

# %%
"""
Create reference frames
-----------------------
Let's say we have the position of three markers, and we want to create a
reference frame based on these markers. The function `create_reference_frame`
aims to do this.

If the markers are at these positions (0, 0, 0), (1, 0, 0) and (0, 1, 0):
"""
global_markers = np.array(
        [[0., 1., 0.],
         [0., 0., 1.],
         [0., 0., 0.],
         [1., 1., 1.]])

# %%
"""
A reference frame can be made of these markers. By default,
`ktk.geometry.create_reference_frame` creates the origin at the centroid,
aligns the x axis on the first marker and the z axis perpendicular to the
plane formed by the origin, the first marker and the second marker.

There are many construction methods for the function, defined in the
function's help. For example, reference frames can be generated from anatomical
marker locations using the recommandations from the International Society of
Biomechanics.
"""
T = ktk.geometry.create_reference_frame(global_markers)
T

# %%
"""
Get local coordinates
---------------------
We now have a reference frame defined from these markers. If we are
interested to know the local position of these markers in this new reference
frame, we can use the function `ktk.geometry.get_local_coordinates`.
"""
local_markers = ktk.geometry.get_local_coordinates(global_markers, T)
local_markers

# %%
"""
Get global coordinates
----------------------
In the case we have the markers' local coordinates and we would like to express
these markers in the global reference frame, we would use the alternate
function `ktk.geometry.get_global_coordinates`.
"""
ktk.geometry.get_global_coordinates(local_markers, T)

# %% exclude

import ktk
import numpy as np

# Unit test for create_reference_frame
global_marker1 = np.array([0.0, 0.0, 0.0, 1])
global_marker2 = np.array([1.0, 0.0, 0.0, 1])
global_marker3 = np.array([0.0, 1.0, 0.0, 1])
global_markers = np.array([global_marker1, global_marker2, global_marker3]).T

T = ktk.geometry.create_reference_frame(global_markers)
local_markers = ktk.geometry.get_local_coordinates(global_markers, T)

# Verify that the distances between markers are the same
local_distance01 = np.sqrt(np.sum(
        (local_markers[:, 0] - local_markers[:, 1]) ** 2))
local_distance12 = np.sqrt(np.sum(
        (local_markers[:, 1] - local_markers[:, 2]) ** 2))
local_distance20 = np.sqrt(np.sum(
        (local_markers[:, 2] - local_markers[:, 0]) ** 2))

global_distance01 = np.sqrt(np.sum(
        (global_markers[:, 0] - global_markers[:, 1]) ** 2))
global_distance12 = np.sqrt(np.sum(
        (global_markers[:, 1] - global_markers[:, 2]) ** 2))
global_distance20 = np.sqrt(np.sum(
        (global_markers[:, 2] - global_markers[:, 0]) ** 2))

assert np.abs(local_distance01 - global_distance01) < 1E-10
assert np.abs(local_distance12 - global_distance12) < 1E-10
assert np.abs(local_distance20 - global_distance20) < 1E-10

# Verify that the determinant is null
assert np.abs(np.linalg.det(global_markers[0:3, 0:3])) < 1E-10
assert np.abs(np.linalg.det(local_markers[0:3, 0:3])) < 1E-10

# Verify that the transformation times local markers gives the global markers
test_global = T @ local_markers

assert np.sum(np.abs(test_global - global_markers)) < 1E-10

# Verify that it works also with a Nx4xM matrix
global_markers = np.repeat(global_markers[np.newaxis, ...], 10, axis=0)
for i_sample in range(10):
    global_markers[i_sample, 0, 0] = i_sample + 0.55  # Just so that every
                                                      # sample is different.

T = ktk.geometry.create_reference_frame(global_markers)
local_markers = ktk.geometry.get_local_coordinates(global_markers, T)
test_global = T @ local_markers
assert np.sum(np.abs(test_global - global_markers)) < 1E-10

# Check get_global_coordinates
test_global2 = ktk.geometry.get_global_coordinates(local_markers, T)
assert np.sum(np.abs(test_global - test_global2)) < 1E-10
