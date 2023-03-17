#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020-2021 Félix Chénier

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
__copyright__ = "Copyright (C) 2020-2021 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import kineticstoolkit as ktk
import numpy as np


def test_matmul():
    """Test matmul function."""
    # Matrix multiplication between a matrix and a series of points:
    result = ktk.geometry.matmul(
        np.array([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]]),
        np.array([[0, 0, 0, 1], [2, 0, 0, 1], [3, 1, 0, 1]]),
    )

    assert (
        np.sum(
            np.abs(
                result - np.array([[0, 0, 0, 1], [2, 0, 0, 1], [3, 1, 0, 1]])
            )
        )
        < 1e-15
    )

    # Multiplication between a series of floats and a series of vectors:
    result = ktk.geometry.matmul(
        np.array([0.0, 0.5, 1.0, 1.5]),
        np.array([[1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0]]),
    )

    assert (
        np.sum(
            np.abs(
                result
                - np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [3.0, 0.0, 0.0, 0.0],
                        [6.0, 0.0, 0.0, 0.0],
                    ]
                )
            )
        )
        < 1e-15
    )

    # Dot product between a series of points and a single point:
    result = ktk.geometry.matmul(
        np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]),
        np.array([[2, 3, 4, 1]]),
    )

    assert np.sum(np.abs(result - np.array([3, 4, 5]))) < 1e-15


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


def test_create_transforms():
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


def rotate_translate_scale():
    """
    Test rotate, translate and scale.
    The real test is in create_transforms, these tests are only to be sure
    that these shortcut functions still work.
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


def test_create_frames_get_local_global_coordinates():
    """
    Test create_frames, get_local_coordinates and
    get_global_coordinates.
    """
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
    """Test fix for issue #136"""
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


def test_create_frames():
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


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
