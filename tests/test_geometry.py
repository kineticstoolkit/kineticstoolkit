#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020 Félix Chénier
#
# This file is not for redistribution.
"""Unit tests for ktk.geometry."""

import ktk
import numpy as np


def test_matmul():
    """Test matmul function."""
    # Matrix multiplication between a matrix and a series of points:
    result = ktk.geometry.matmul(
        np.array([[[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]]]),
        np.array([[0, 0, 0, 1],
                  [2, 0, 0, 1],
                  [3, 1, 0, 1]]))

    assert np.sum(np.abs(result - np.array([[0, 0, 0, 1],
                                            [2, 0, 0, 1],
                                            [3, 1, 0, 1]]))) < 1E-15

    # Multiplication between a series of floats and a series of vectors:
    result = ktk.geometry.matmul(
        np.array([0., 0.5, 1., 1.5]),
        np.array([[1, 0, 0, 0],
                  [2, 0, 0, 0],
                  [3, 0, 0, 0],
                  [4, 0, 0, 0]]))

    assert np.sum(np.abs(result - np.array([[0., 0., 0., 0.],
                                            [1., 0., 0., 0.],
                                            [3., 0., 0., 0.],
                                            [6., 0., 0., 0.]]))) < 1E-15

    # Dot product between a series of points and a single point:
    result = ktk.geometry.matmul(
        np.array([[1, 0, 0, 1],
                  [0, 1, 0, 1],
                  [0, 0, 1, 1]]),
        np.array([[2, 3, 4, 1]]))

    assert np.sum(np.abs(result - np.array([3, 4, 5]))) < 1E-15


def test_create_rotation_matrices():
    """Test create_rotation_matrices function."""
    # Identity matrix
    T = ktk.geometry.create_rotation_matrices('x', [0])
    assert np.sum(np.abs(T[0] - np.eye(4))) < 1E-15

    # Rotation of 90 degrees around the x axis
    T = ktk.geometry.create_rotation_matrices('x', [np.pi/2])
    assert np.sum(np.abs(T[0] - np.array([
        [1., 0., 0., 0.],
        [0., 0., -1., 0.],
        [0., 1., 0., 0.],
        [0., 0., 0., 1.]]
        ))) < 1E-15

    # Series of 100 rotation matrices around the z axis, from 0 to
    # 360 degrees
    T = ktk.geometry.create_rotation_matrices(
        'z', np.linspace(0, 2 * np.pi, 100))


def test_create_reference_frames_get_local_global_coordinates():
    """
    Test create_reference_frames, get_local_coordinates and
    get_global_coordinates.
    """
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

    # Verify that the transformation times local markers gives the global
    # markers
    test_global = T @ local_markers

    assert np.sum(np.abs(test_global - global_markers)) < 1E-10


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])

