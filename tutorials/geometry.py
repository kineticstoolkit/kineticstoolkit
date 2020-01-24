# %%
"""
geometry
========
This module simplifies the calculation of linear algebric operations on
series of points, vectors and transformation matrices.

Every point, vector or matrix is considered as a series, where the dimensions
of the array are:
    - First dimension (N) : time. To represent constants, simply use N=1.
    - Second dimension (4) : point/vector coordinates, or matrix line.
    - Third dimension (M, optional) : point index (for sets of point),
                                      or matrix column.

More precisely, the points, vectors and matrices are expressed as following
(here, N=2):

Series of points:

    [[x(t1), y(t1), z(t1), 1.],
     [y(t2), y(t2), z(t2), 1.]]

Series of vectors:

    [[x(t1), y(t1), z(t1), 0.],
     [y(t2), y(t2), z(t2), 0.]]

Series of sets of point:

    [[[x1(t1), x2(t1), x3(t1), x4(t1)],
      [y1(t1), y2(t1), y3(t1), y4(t1)],
      [z1(t1), z2(t1), z3(t1), z4(t1)],
      [  1.,     1.,     1.,      1. ]],

     [[x1(t2), x2(t2), x3(t2), x4(t2)],
      [y1(t2), y2(t2), y3(t2), y4(t2)],
      [z1(t2), z2(t2), z3(t2), z4(t2)],
      [  1.,     1.,     1.,     1.  ]]]

Series of matrices:

    [[[R11(t1), R12(t1), R13(t1), Tx(t1)],
      [R21(t1), R22(t1), R23(t1), Ty(t1)],
      [R31(t1), R32(t1), R33(t1), Tz(t1)],
      [   0.,      0.,      0.,     1.  ]],

     [[R11(t2), R12(t2), R13(t2), Tx(t2)],
      [R21(t2), R22(t2), R23(t2), Ty(t2)],
      [R31(t2), R32(t2), R33(t2), Tz(t2)],
      [   0.,      0.,      0.,     1.  ]]]

"""
import ktk
import numpy as np

# %%
"""
To facilitate the multiplication of series of matrices, ktk.geometry provides
a `matmul` function that matches both matrices' time dimensions, then applies
the numpy's @ or * operator accordingly on each time iteration. Therefore, when
dealing with series of floats, vectors, points or matrices, it is advisable to
use ktk.geometry's `matmul` instead of the @ operator. For example:

Matrix multiplication between a matrix and a series of points:
"""
result = ktk.geometry.matmul(
        np.array([[[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]]]),
        np.array([[0, 0, 0, 1],
                  [2, 0, 0, 1],
                  [3, 1, 0, 1]]))
result

# %% exclude
assert np.sum(np.abs(result - np.array([[0, 0, 0, 1],
                                        [2, 0, 0, 1],
                                        [3, 1, 0, 1]]))) < 1E-15

# %%
"""
Multiplication between a series of floats and a series of vectors:
"""
result = ktk.geometry.matmul(
        np.array([0., 0.5, 1., 1.5]),
        np.array([[1, 0, 0, 0],
                  [2, 0, 0, 0],
                  [3, 0, 0, 0],
                  [4, 0, 0, 0]]))
result

# %% exclude
assert np.sum(np.abs(result - np.array([[0., 0., 0., 0.],
                                        [1., 0., 0., 0.],
                                        [3., 0., 0., 0.],
                                        [6., 0., 0., 0.]]))) < 1E-15
# %%
"""
Dot product between a series of points and a single point:
"""
result = ktk.geometry.matmul(
        np.array([[1, 0, 0, 1],
                  [0, 1, 0, 1],
                  [0, 0, 1, 1]]),
        np.array([[2, 3, 4, 1]]))
result

# %% exclude
assert np.sum(np.abs(result - np.array([3, 4, 5]))) < 1E-15

# %%
"""
Create a series of rotation matrices
------------------------------------
The function `ktk.geometry.create_rotation_matrices` allows creating series of
Nx4x4 matrices around a given axis. For example:
"""
T = ktk.geometry.create_rotation_matrices('x', [0])

# %%
"""
creates the identity matrix:
"""
T

# %% exclude
assert np.sum(np.abs(T[0] - np.eye(4))) < 1E-15

# %%
T = ktk.geometry.create_rotation_matrices('x', [np.pi/2])

# %%
"""
creates a rotation of 90 degrees around the x axis:
"""
T

# %% exclude
assert np.sum(np.abs(T[0] - np.array([
       [1.,  0.,  0.,  0.],
       [0.,  0., -1.,  0.],
       [0.,  1.,  0.,  0.],
       [0.,  0.,  0.,  1.]]
       ))) < 1E-15

# %%
"""
and
"""
T = ktk.geometry.create_rotation_matrices('z', np.linspace(0, 2 * np.pi, 100))

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
reference frame based on these markers. The function `create_reference_frames`
aims to do this.

If the markers are at these positions (0, 0, 0), (1, 0, 0) and (0, 1, 0):
"""
global_markers = np.array(
        [[[0., 1., 0.],
          [0., 0., 1.],
          [0., 0., 0.],
          [1., 1., 1.]]])

# %%
"""
A reference frame can be made of these markers. By default,
`ktk.geometry.create_reference_frames` creates the origin at the centroid,
aligns the x axis on the first marker and the z axis perpendicular to the
plane formed by the origin, the first marker and the second marker.

There are many construction methods for the function, defined in the
function's help. For example, reference frames can be generated from anatomical
marker locations using the recommandations from the International Society of
Biomechanics.
"""
T = ktk.geometry.create_reference_frames(global_markers)
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
import numpy as np

# Unit test for create_reference_frame
global_marker1 = np.array([[0.0, 0.0, 0.0, 1]])
global_marker2 = np.array([[1.0, 0.0, 0.0, 1]])
global_marker3 = np.array([[0.0, 1.0, 0.0, 1]])
global_markers = np.block([global_marker1[:, :, np.newaxis],
                           global_marker2[:, :, np.newaxis],
                           global_marker3[:, :, np.newaxis]])
# Repeat for N=5
global_markers = np.repeat(global_markers, 5, axis=0)

T = ktk.geometry.create_reference_frames(global_markers)
local_markers = ktk.geometry.get_local_coordinates(global_markers, T)

# Verify that the distances between markers are the same
local_distance01 = np.sqrt(np.sum(
        (local_markers[0, :, 0] - local_markers[0, :, 1]) ** 2))
local_distance12 = np.sqrt(np.sum(
        (local_markers[0, :, 1] - local_markers[0, :, 2]) ** 2))
local_distance20 = np.sqrt(np.sum(
        (local_markers[0, :, 2] - local_markers[0, :, 0]) ** 2))

global_distance01 = np.sqrt(np.sum(
        (global_markers[0, :, 0] - global_markers[0, :, 1]) ** 2))
global_distance12 = np.sqrt(np.sum(
        (global_markers[0, :, 1] - global_markers[0, :, 2]) ** 2))
global_distance20 = np.sqrt(np.sum(
        (global_markers[0, :, 2] - global_markers[0, :, 0]) ** 2))

assert np.abs(local_distance01 - global_distance01) < 1E-10
assert np.abs(local_distance12 - global_distance12) < 1E-10
assert np.abs(local_distance20 - global_distance20) < 1E-10

# Verify that the determinant is null
assert np.abs(np.linalg.det(global_markers[0, 0:3, 0:3])) < 1E-10
assert np.abs(np.linalg.det(local_markers[0, 0:3, 0:3])) < 1E-10

# Verify that the transformation times local markers gives the global markers
test_global = T @ local_markers

assert np.sum(np.abs(test_global - global_markers)) < 1E-10

