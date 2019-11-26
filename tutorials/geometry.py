# %% markdown
"""
ktk.geometry Tutorial
=====================
This is not yet a tutorial. This is the beginning of some unit tests.
"""

# %% hide

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

assert np.sum(np.abs(test_global[:, 0] - global_markers[:, 0])) < 1E-10

# Verify that it works also with a Nx4xM matrix
global_markers = np.repeat(global_markers[np.newaxis, ...], 10, axis=0)
for i_sample in range(10):
    global_markers[i_sample, 0, 0] = i_sample + 0.55  # Just so that every
                                                      # sample is different.

T = ktk.geometry.create_reference_frame(global_markers)
local_markers = ktk.geometry.get_local_coordinates(global_markers, T)
test_global = T @ local_markers
assert np.sum(np.abs(test_global[:, 0] - global_markers[:, 0])) < 1E-10
