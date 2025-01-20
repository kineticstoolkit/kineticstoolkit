#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020-2024 Félix Chénier

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for Kinetics Toolkit's geometry module."""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2024 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import kineticstoolkit as ktk
import numpy as np


def test_matmul():
    """Test matmul function."""
    # Matrix multiplication between a matrix and a series of points:
    result = ktk.geometry.matmul(
        [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]],
        [[0, 0, 0, 1], [2, 0, 0, 1], [3, 1, 0, 1]],
    )

    assert np.allclose(result, [[0, 0, 0, 1], [2, 0, 0, 1], [3, 1, 0, 1]])

    # Multiplication between a series of floats and a series of vectors:
    result = ktk.geometry.matmul(
        [0.0, 0.5, 1.0, 1.5],
        [[1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0]],
    )

    assert np.allclose(
        result,
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
            [6.0, 0.0, 0.0, 0.0],
        ],
    )

    # Dot product between a series of points and a single point:
    result = ktk.geometry.matmul(
        [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]],
        [[2, 3, 4, 1]],
    )

    assert np.allclose(result, [3, 4, 5])


def test_create_point_series():
    """Test create_point_series."""
    assert np.allclose(
        ktk.geometry.create_point_series([0, 3, 6]),
        [[0, 0, 0, 1], [3, 0, 0, 1], [6, 0, 0, 1]],
    )
    assert np.allclose(
        ktk.geometry.create_point_series([[0], [3], [6]]),
        [[0, 0, 0, 1], [3, 0, 0, 1], [6, 0, 0, 1]],
    )
    assert np.allclose(
        ktk.geometry.create_point_series([[0, 1], [3, 4], [6, 7]]),
        [[0, 1, 0, 1], [3, 4, 0, 1], [6, 7, 0, 1]],
    )
    assert np.allclose(
        ktk.geometry.create_point_series([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
        [[0, 1, 2, 1], [3, 4, 5, 1], [6, 7, 8, 1]],
    )
    assert np.allclose(
        ktk.geometry.create_point_series(
            [[0, 1, 2, 1], [3, 4, 5, 1], [6, 7, 8, 1]]
        ),
        [[0, 1, 2, 1], [3, 4, 5, 1], [6, 7, 8, 1]],
    )
    assert np.allclose(
        ktk.geometry.create_point_series(
            [[0, 1, 2, 0], [3, 4, 5, 0], [6, 7, 8, 0]]
        ),
        [[0, 1, 2, 1], [3, 4, 5, 1], [6, 7, 8, 1]],
    )
    assert np.allclose(
        ktk.geometry.create_point_series(x=[1, 2, 3]),
        [[1, 0, 0, 1], [2, 0, 0, 1], [3, 0, 0, 1]],
    )
    assert np.allclose(
        ktk.geometry.create_point_series(y=[1, 2, 3]),
        [[0, 1, 0, 1], [0, 2, 0, 1], [0, 3, 0, 1]],
    )
    assert np.allclose(
        ktk.geometry.create_point_series(z=[1, 2, 3]),
        [[0, 0, 1, 1], [0, 0, 2, 1], [0, 0, 3, 1]],
    )
    assert np.allclose(
        ktk.geometry.create_point_series(x=[1, 2, 3], z=[4, 5, 6]),
        [[1, 0, 4, 1], [2, 0, 5, 1], [3, 0, 6, 1]],
    )


def test_create_vector_series():
    """Test create_vector_series."""
    assert np.allclose(
        ktk.geometry.create_vector_series([0, 3, 6]),
        [[0, 0, 0, 0], [3, 0, 0, 0], [6, 0, 0, 0]],
    )
    assert np.allclose(
        ktk.geometry.create_vector_series([[0], [3], [6]]),
        [[0, 0, 0, 0], [3, 0, 0, 0], [6, 0, 0, 0]],
    )
    assert np.allclose(
        ktk.geometry.create_vector_series([[0, 1], [3, 4], [6, 7]]),
        [[0, 1, 0, 0], [3, 4, 0, 0], [6, 7, 0, 0]],
    )
    assert np.allclose(
        ktk.geometry.create_vector_series([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
        [[0, 1, 2, 0], [3, 4, 5, 0], [6, 7, 8, 0]],
    )
    assert np.allclose(
        ktk.geometry.create_vector_series(
            [[0, 1, 2, 1], [3, 4, 5, 1], [6, 7, 8, 1]]
        ),
        [[0, 1, 2, 0], [3, 4, 5, 0], [6, 7, 8, 0]],
    )
    assert np.allclose(
        ktk.geometry.create_vector_series(
            [[0, 1, 2, 0], [3, 4, 5, 0], [6, 7, 8, 0]]
        ),
        [[0, 1, 2, 0], [3, 4, 5, 0], [6, 7, 8, 0]],
    )
    assert np.allclose(
        ktk.geometry.create_vector_series(x=[1, 2, 3]),
        [[1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]],
    )
    assert np.allclose(
        ktk.geometry.create_vector_series(y=[1, 2, 3]),
        [[0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]],
    )
    assert np.allclose(
        ktk.geometry.create_vector_series(z=[1, 2, 3]),
        [[0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0]],
    )
    assert np.allclose(
        ktk.geometry.create_vector_series(x=[1, 2, 3], z=[4, 5, 6]),
        [[1, 0, 4, 0], [2, 0, 5, 0], [3, 0, 6, 0]],
    )


def test_create_transform_series_with_angle_inputs():
    """Test create_transforms."""
    # Identity matrix
    T = ktk.geometry.create_transform_series(seq="x", angles=[0])
    assert np.allclose(T[0], np.eye(4))

    # Rotation of 90 degrees around the x axis
    T = ktk.geometry.create_transform_series(seq="x", angles=[np.pi / 2])
    assert np.allclose(
        T[0],
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )

    # Series of 100 rotation matrices around the z axis, from 0 to
    # 360 degrees, with a series of translations of 2 to the right.
    T = ktk.geometry.create_transform_series(
        seq="z", angles=np.linspace(0, 2 * np.pi, 100), positions=[[2, 0, 0]]
    )
    assert np.allclose(
        T[0],
        np.array([[1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
    )
    assert T.shape[0] == 100


def test_create_transform_series_with_vector_input():
    """Test create_transform_series with vector input."""

    # First test with a length of 2 and a non-zero position. The next will be
    # with a length of 1 and a zero position, and will generate the identity
    # matrix.
    x = np.array([[1.0, 0.0, 0.0, 0.0]])
    xy = np.array([[0.0, 1.0, 0.0, 0.0]])
    positions = np.array([[23.0, 0.0, 0.0, 1.0]])
    length = 2
    result = ktk.geometry.create_transform_series(
        x=x, xy=xy, positions=positions, length=length
    )

    expected_result = np.array(
        [
            [
                [1.0, 0.0, 0.0, 23.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [1.0, 0.0, 0.0, 23.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ]
    )

    x = np.array([[1.0, 0.0, 0.0, 0.0]])
    xy = np.array([[0.0, 2.0, 0.0, 0.0]])
    positions = np.array([[0.0, 0.0, 0.0, 1.0]])
    length = 1
    result = ktk.geometry.create_transform_series(
        x=x, xy=xy, positions=positions, length=length
    )

    expected_result = np.array(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ]
    )

    assert np.allclose(result, expected_result)

    x = np.array([[1.0, 0.0, 0.0, 0.0]])
    xz = np.array([[0.0, 0.0, 3.0, 0.0]])
    positions = np.array([[0.0, 0.0, 0.0, 1.0]])
    length = 1
    result = ktk.geometry.create_transform_series(
        x=x, xz=xz, positions=positions, length=length
    )

    assert np.allclose(result, expected_result)

    y = np.array([[0.0, 1.0, 0.0, 0.0]])
    yz = np.array([[0.0, 0.0, 4.0, 0.0]])
    positions = np.array([[0.0, 0.0, 0.0, 1.0]])
    length = 1
    result = ktk.geometry.create_transform_series(
        y=y, yz=yz, positions=positions, length=length
    )

    assert np.allclose(result, expected_result)

    y = np.array([[0.0, 1.0, 0.0, 0.0]])
    xy = np.array([[5.0, 0.0, 0.0, 0.0]])
    positions = np.array([[0.0, 0.0, 0.0, 1.0]])
    length = 1
    result = ktk.geometry.create_transform_series(
        y=y, xy=xy, positions=positions, length=length
    )

    assert np.allclose(result, expected_result)

    z = np.array([[0.0, 0.0, 1.0, 0.0]])
    xz = np.array([[6.0, 0.0, 0.0, 0.0]])
    positions = np.array([[0.0, 0.0, 0.0, 1.0]])
    length = 1
    result = ktk.geometry.create_transform_series(
        z=z, xz=xz, positions=positions, length=length
    )

    assert np.allclose(result, expected_result)

    z = np.array([[0.0, 0.0, 1.0, 0.0]])
    yz = np.array([[0.0, 7.0, 0.0, 0.0]])
    positions = np.array([[0.0, 0.0, 0.0, 1.0]])
    length = 1
    result = ktk.geometry.create_transform_series(
        z=z, yz=yz, positions=positions, length=length
    )

    assert np.allclose(result, expected_result)

    # Do the same with random vectors

    # Rotate 90 degrees around x
    test = ktk.geometry.create_transform_series(
        positions=[[0, 0, 0, 1]], z=[[0, -2, 0, 0]], yz=[[0, 2, 2, 0]]
    )
    assert np.allclose(test, ktk.geometry.create_transforms("x", [np.pi / 2]))

    # Rotate 90 degrees around y
    test = ktk.geometry.create_transform_series(
        positions=[[0, 0, 0, 1]], x=[[0, 0, -2, 0]], xy=[[0, 2, -2, 0]]
    )
    assert np.allclose(test, ktk.geometry.create_transforms("y", [np.pi / 2]))

    # Rotate 90 degrees around z
    test = ktk.geometry.create_transform_series(
        positions=[[0, 0, 0, 1]], x=[[0, 2, 0, 0]], xy=[[-2, 2, 0, 0]]
    )
    assert np.allclose(test, ktk.geometry.create_transforms("z", [np.pi / 2]))


def test_is_frame_point_vector_series():
    """Test is_transform_series, is_point_series and is_vector_series."""
    assert (
        ktk.geometry.is_transform_series(
            [
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            ]
        )
        == True
    )
    assert (
        ktk.geometry.is_transform_series(
            [
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                [[np.nan, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            ]
        )
        == True
    )
    assert (
        ktk.geometry.is_transform_series(
            [
                [[np.nan, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                [[np.nan, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            ]
        )
        == False
    )
    assert (
        ktk.geometry.is_transform_series(
            [
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            ]
        )
        == False
    )


def test_inv():
    """Test inverse matrix series."""
    # Try with simple rotations and translations
    T = ktk.geometry.create_transforms(
        "z", [90], degrees=True, translations=[[1, 0, 0]]
    )
    assert np.allclose(
        T, [[0, -1, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    assert np.allclose(ktk.geometry.inv(T), np.linalg.inv(T))

    # Series of 100 rotation matrices around the z axis, from 0 to
    # 360 degrees, with a series of translations of (2,1,3).
    T = ktk.geometry.create_transforms(
        "z", np.linspace(0, 2 * np.pi, 100), translations=[[2, 1, 3]]
    )
    assert np.allclose(ktk.geometry.inv(ktk.geometry.inv(T)), T)

    # See if the matrix is not a rigid transform
    T[10, 0, 0] = 0.0

    try:
        ktk.geometry.inv(T)
        raise ValueError("This should raise an error")
    except ValueError:
        pass


def test_rotate_translate_scale():
    """
    Test rotate, translate and scale.

    The real test is in create_transforms, these tests are only to be
    sure that these shortcut functions still work.

    """
    angles = np.array([[0, 45], [10, 45], [20, 45], [30, 45], [40, 45]])
    assert np.allclose(
        ktk.geometry.rotate([[1, 0, 0, 1]], "zx", angles, degrees=True),
        np.array(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.98480775, 0.1227878, 0.1227878, 1.0],
                [0.93969262, 0.24184476, 0.24184476, 1.0],
                [0.8660254, 0.35355339, 0.35355339, 1.0],
                [0.76604444, 0.45451948, 0.45451948, 1.0],
            ]
        ),
        atol=1e-3,
    )

    t = np.array([[0, 1, 0], [1, 1, 0], [2, 1, 0], [3, 1, 0], [4, 1, 0]])
    assert np.allclose(
        ktk.geometry.translate([[1, 0, 0, 1]], t),
        np.array(
            [
                [1.0, 1.0, 0.0, 1.0],
                [2.0, 1.0, 0.0, 1.0],
                [3.0, 1.0, 0.0, 1.0],
                [4.0, 1.0, 0.0, 1.0],
                [5.0, 1.0, 0.0, 1.0],
            ]
        ),
    )

    s = np.array([0, 1, 2, 3, 4])
    assert np.allclose(
        ktk.geometry.scale([[1, 0, 0, 1]], s),
        np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [2.0, 0.0, 0.0, 1.0],
                [3.0, 0.0, 0.0, 1.0],
                [4.0, 0.0, 0.0, 1.0],
            ]
        ),
    )


def test_mirror():
    """Run the doctest."""
    p = np.array([[1.0, 2.0, 3.0, 1.0]])

    assert np.allclose(ktk.geometry.mirror(p, "x"), [[-1.0, 2.0, 3.0, 1.0]])

    assert np.allclose(ktk.geometry.mirror(p, "y"), [[1.0, -2.0, 3.0, 1.0]])

    assert np.allclose(ktk.geometry.mirror(p, "z"), [[1.0, 2.0, -3.0, 1.0]])


def test_get_local_global_coordinates():
    """Test get_local_coordinates and get_global_coordinates."""
    global_marker1 = np.array([[0.0, 0.0, 0.0, 1]])
    global_marker2 = np.array([[1.0, 0.0, 0.0, 1]])
    global_marker3 = np.array([[0.0, 1.0, 0.0, 1]])
    global_markers = np.block(
        [
            global_marker1[:, :, np.newaxis],
            global_marker2[:, :, np.newaxis],
            global_marker3[:, :, np.newaxis],
        ]
    )
    # Repeat for N=5
    global_markers = np.repeat(global_markers, 5, axis=0)

    T = ktk.geometry.create_frames(
        origin=global_marker1,
        x=global_marker2 - global_marker1,
        xy=global_marker3 - global_marker1,
    )
    local_markers = ktk.geometry.get_local_coordinates(global_markers, T)

    # Verify that the distances between markers are the same
    local_distance01 = np.sqrt(
        np.sum((local_markers[0, :, 0] - local_markers[0, :, 1]) ** 2)
    )
    local_distance12 = np.sqrt(
        np.sum((local_markers[0, :, 1] - local_markers[0, :, 2]) ** 2)
    )
    local_distance20 = np.sqrt(
        np.sum((local_markers[0, :, 2] - local_markers[0, :, 0]) ** 2)
    )

    global_distance01 = np.sqrt(
        np.sum((global_markers[0, :, 0] - global_markers[0, :, 1]) ** 2)
    )
    global_distance12 = np.sqrt(
        np.sum((global_markers[0, :, 1] - global_markers[0, :, 2]) ** 2)
    )
    global_distance20 = np.sqrt(
        np.sum((global_markers[0, :, 2] - global_markers[0, :, 0]) ** 2)
    )

    assert np.abs(local_distance01 - global_distance01) < 1e-10
    assert np.abs(local_distance12 - global_distance12) < 1e-10
    assert np.abs(local_distance20 - global_distance20) < 1e-10

    # Verify that the determinant is null
    assert np.abs(np.linalg.det(global_markers[0, 0:3, 0:3])) < 1e-10
    assert np.abs(np.linalg.det(local_markers[0, 0:3, 0:3])) < 1e-10

    # Verify that the transformation times local markers gives the global
    # markers
    test_global = T @ local_markers

    assert np.sum(np.abs(test_global - global_markers)) < 1e-10


def test_get_local_global_broadcast():
    """Test fix for issue #136."""
    global_marker1 = np.array([[0.0, 0.0, 0.0, 1]])
    global_marker2 = np.array([[1.0, 0.0, 0.0, 1]])
    global_marker3 = np.array([[0.0, 1.0, 0.0, 1]])
    global_markers = np.block(
        [
            global_marker1[:, :, np.newaxis],
            global_marker2[:, :, np.newaxis],
            global_marker3[:, :, np.newaxis],
        ]
    )
    T = ktk.geometry.create_frames(
        origin=global_marker1,
        x=global_marker2 - global_marker1,
        xy=global_marker3 - global_marker1,
    )

    # None of these calls should crash
    ktk.geometry.get_local_coordinates(np.repeat(global_markers, 5, axis=0), T)
    ktk.geometry.get_global_coordinates(
        np.repeat(global_markers, 5, axis=0), T
    )
    ktk.geometry.get_local_coordinates(global_markers, np.repeat(T, 5, axis=0))
    ktk.geometry.get_global_coordinates(
        global_markers, np.repeat(T, 5, axis=0)
    )


def test_get_angles():
    """Test get_angles and create_transforms."""
    np.random.seed(0)

    # Test with only one angle
    angles = np.random.rand(10) * 2 * np.pi
    T = ktk.geometry.create_transforms("X", angles)
    test_angles = ktk.geometry.get_angles(T, "XYZ")
    assert np.allclose(angles, np.mod(test_angles[:, 0], 2 * np.pi))

    # Test with three angles (tait-bryan), and with this time in degrees
    angles = np.zeros((10, 3))
    angles[:, 0] = (np.random.rand(10) * 2 - 1) * 180
    angles[:, 1] = (np.random.rand(10) - 1) * 90
    angles[:, 2] = (np.random.rand(10) * 2 - 1) * 180
    T = ktk.geometry.create_transforms("XYZ", angles, degrees=True)
    test_angles = ktk.geometry.get_angles(T, "XYZ", degrees=True)
    assert np.allclose(angles, test_angles)

    # Test with proper euler angles
    angles = np.zeros((10, 3))
    angles[:, 0] = (np.random.rand(10) * 2 - 1) * 180
    angles[:, 1] = np.random.rand(10) * 180
    angles[:, 2] = (np.random.rand(10) * 2 - 1) * 180
    T = ktk.geometry.create_transforms("XYX", angles, degrees=True)
    test_angles = ktk.geometry.get_angles(T, "XYX", degrees=True)
    assert np.allclose(angles, test_angles)

    # Test flip with tait-bryan
    angles = np.zeros((10, 3))
    angles[:, 0] = (np.random.rand(10) * 2 - 1) * 180
    angles[:, 1] = np.random.rand(10) * 180 + 90
    angles[angles[:, 1] > 180, :] -= 360
    angles[:, 2] = (np.random.rand(10) * 2 - 1) * 180
    T = ktk.geometry.create_transforms("XYZ", angles, degrees=True)
    test_angles = ktk.geometry.get_angles(T, "XYZ", degrees=True, flip=True)
    assert np.allclose(angles, test_angles)

    # Test flip with proper euler angles
    angles = np.zeros((10, 3))
    angles[:, 0] = (np.random.rand(10) * 2 - 1) * 180
    angles[:, 1] = -np.random.rand(10) * 180
    angles[:, 2] = (np.random.rand(10) * 2 - 1) * 180
    T = ktk.geometry.create_transforms("XYX", angles, degrees=True)
    test_angles = ktk.geometry.get_angles(T, "XYX", degrees=True, flip=True)
    assert np.allclose(angles, test_angles)


def test_get_quaternions():
    """Test get_quaternions and construction of transforms based on quats."""
    # Build a transform series with different angles and positions
    np.random.seed(0)
    angles = np.zeros((10, 3))
    angles[:, 0] = (np.random.rand(10) * 2 - 1) * 180
    angles[:, 1] = (np.random.rand(10) - 1) * 90
    angles[:, 2] = (np.random.rand(10) * 2 - 1) * 180
    positions = np.zeros((10, 3))
    positions[:, 0] = np.random.rand(10)
    positions[:, 1] = np.random.rand(10)
    positions[:, 2] = np.random.rand(10)
    T = ktk.geometry.create_transform_series(
        angles=angles, seq="xyz", positions=positions
    )

    # Extract quaternions and build a new transform series based on these quats
    quaternions = ktk.geometry.get_quaternions(T)
    T2 = ktk.geometry.create_transform_series(
        quaternions=quaternions, positions=positions
    )

    assert np.allclose(T, T2)


def test_create_transforms_tobedeprecated():
    """Test create_transforms."""
    # Identity matrix
    T = ktk.geometry.create_transforms("x", [0])
    assert np.allclose(T[0], np.eye(4))

    # Rotation of 90 degrees around the x axis
    T = ktk.geometry.create_transforms("x", [np.pi / 2])
    assert np.allclose(
        T[0],
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )

    # Rotation of 90 degrees around the x axis with a scaling of 1000
    T = ktk.geometry.create_transforms("x", [np.pi / 2], scales=[1000])
    assert np.allclose(
        T[0],
        np.array(
            [
                [1000.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1000.0, 0.0],
                [0.0, 1000.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )

    # Series of 100 rotation matrices around the z axis, from 0 to
    # 360 degrees, with a series of translations of 2 to the right.
    T = ktk.geometry.create_transforms(
        "z", np.linspace(0, 2 * np.pi, 100), translations=[[2, 0, 0]]
    )
    assert np.allclose(
        T[0],
        np.array([[1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
    )
    assert T.shape[0] == 100


def test_create_frames_tobedeprecated():
    """Test create_frames."""
    # Create identity
    test = ktk.geometry.create_frames(
        [[0, 0, 0, 1]], x=[[2, 0, 0, 0]], xy=[[2, 2, 0, 0]]
    )
    assert np.allclose(test, np.eye(4)[np.newaxis])

    # Rotate 90 degrees around y
    test = ktk.geometry.create_frames(
        [[0, 0, 0, 1]], x=[[0, 0, -2, 0]], xy=[[0, 2, -2, 0]]
    )
    assert np.allclose(test, ktk.geometry.create_transforms("y", [np.pi / 2]))

    # Rotate 90 degrees around z
    test = ktk.geometry.create_frames(
        [[0, 0, 0, 1]], x=[[0, 2, 0, 0]], xy=[[-2, 2, 0, 0]]
    )
    assert np.allclose(test, ktk.geometry.create_transforms("z", [np.pi / 2]))

    # Create identity using other vectors and planes
    test = ktk.geometry.create_frames(
        [[0, 0, 0, 1]], x=[[2, 0, 0, 0]], xz=[[2, 0, 2, 0]]
    )
    assert np.allclose(test, np.eye(4)[np.newaxis])

    test = ktk.geometry.create_frames(
        [[0, 0, 0, 1]], y=[[0, 2, 0, 0]], xy=[[2, 2, 0, 0]]
    )
    assert np.allclose(test, np.eye(4)[np.newaxis])

    test = ktk.geometry.create_frames(
        [[0, 0, 0, 1]], y=[[0, 2, 0, 0]], yz=[[0, 2, 2, 0]]
    )
    assert np.allclose(test, np.eye(4)[np.newaxis])

    test = ktk.geometry.create_frames(
        [[0, 0, 0, 1]], z=[[0, 0, 2, 0]], xz=[[2, 0, 2, 0]]
    )
    assert np.allclose(test, np.eye(4)[np.newaxis])

    test = ktk.geometry.create_frames(
        [[0, 0, 0, 1]], z=[[0, 0, 2, 0]], yz=[[0, 2, 2, 0]]
    )
    assert np.allclose(test, np.eye(4)[np.newaxis])


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
