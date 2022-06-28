#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020 Félix Chénier

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"

"""
Unit tests for the inversedynamics module.
"""
import kineticstoolkit as ktk
import numpy as np
import warnings


def test_calculate_proximal_wrenches_2d_static():
    """
    Test calculate_proximal_wrenches for a 2d static case

                    |  80 N
                    |                 y ^
                    |                   |
         m=3kg      V                   |
    o======1=======2m                    -----> x

    This test uses this 2d figure on a 5 second-long simulated static trial
    with automatic reconstruction of COMPosition, ComAcceleration,
    SegmentAngles, AngularVelocity and AngularAcceleration.
    """
    n_points = 100
    ts = ktk.TimeSeries(time=np.linspace(0, 1, n_points))
    ts.data["ProximalJointPosition"] = np.repeat(
        np.array([[0, 0, 0, 1]]), n_points, axis=0
    )
    ts.data["DistalJointPosition"] = np.repeat(
        np.array([[2, 0, 0, 1]]), n_points, axis=0
    )
    ts.data["ForceApplicationPosition"] = np.repeat(
        np.array([[2, 0, 0, 1]]), n_points, axis=0
    )
    ts.data["DistalForces"] = np.repeat(
        np.array([[0, 80, 0, 0]]), n_points, axis=0
    )
    ts.data["DistalMoments"] = np.repeat(
        np.array([[0, 0, 0, 0]]), n_points, axis=0
    )

    inertial_constants = {
        "Mass": 3,
        "COMProximalRatio": 0.5,
        "GyrationCOMRatio": 0.1,
    }

    # Catch warnings because we use automatic com/angle/vel/acc calculation,
    # which generate warnings.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        prox = ktk.inversedynamics.calculate_proximal_wrench(
            ts, inertial_constants
        )

    assert np.allclose(
        np.nanmedian(prox.data["ProximalForces"], axis=0),
        [0.0, 80 + 3.0 * 9.81, 0.0, 0.0],
    )

    assert np.allclose(
        np.nanmedian(prox.data["ProximalMoments"], axis=0),
        [0.0, 0.0, 160 + 3.0 * 9.81, 0.0],
    )


def test_calculate_proximal_wrenches_2d_dynamic():
    """
    Test various dynamic situations based on dummy kinematic values.

                    |  80 N
                    |                 y ^
                    |                   |
         m=3kg      V                   |
    o======1=======2m                    -----> x

    These tests are only on one-point timeseries, with fake accelerations
    and velocities. The aim is to test the wrench equations, not the
    calculation of accelerations, velocities, etc.
    """
    ts = ktk.TimeSeries(time=np.array([0]))
    ts.data["ProximalJointPosition"] = np.array([[0, 0, 0, 1]])
    ts.data["DistalJointPosition"] = np.array([[2, 0, 0, 1]])
    ts.data["COMPosition"] = np.array([[1, 0, 0, 1]])
    ts.data["ForceApplicationPosition"] = np.array([[2, 0, 0, 1]])
    ts.data["DistalForces"] = np.array([[0, 80, 0, 0]])
    ts.data["DistalMoments"] = np.array([[0, 0, 0, 0]])

    inertial_constants = {
        "Mass": 3,
        "GyrationCOMRatio": 0.1,
    }

    # Test 1: Fully static
    ts.data["COMAcceleration"] = np.array([[0, 0, 0, 0]])
    ts.data["AngularVelocity"] = np.array([[0, 0, 0]])
    ts.data["AngularAcceleration"] = np.array([[0, 0, 0]])

    prox = ktk.inversedynamics.calculate_proximal_wrench(
        ts, inertial_constants
    )

    assert np.all(
        np.abs(
            prox.data["ProximalForces"][0] - [0.0, 80 + 3.0 * 9.81, 0.0, 0.0]
        )
        < 1e-10
    )
    assert np.all(
        np.abs(
            prox.data["ProximalMoments"][0] - [0.0, 0.0, 160 + 3.0 * 9.81, 0.0]
        )
        < 1e-10
    )

    # Test 2: The origin is fixed and the segment is not turning but has an
    # angular acceleration of 1 rad/s2. This means the COM has an upward
    # linear acceleration of 1 m/s2. We expect the y proximal force to have
    # an additional (ma) component upward. For the moments, we expect an
    # additional z proximal moment of (Ialpha) = I
    # I = mk^2 + md^2 = 0.04 * 3 + 3 = 3.12
    ts.data["COMAcceleration"] = np.array([[0, 1, 0, 0]])
    ts.data["AngularVelocity"] = np.array([[0, 0, 0]])
    ts.data["AngularAcceleration"] = np.array([[0, 0, 1]])

    prox = ktk.inversedynamics.calculate_proximal_wrench(
        ts, inertial_constants
    )

    assert np.allclose(
        prox.data["ProximalForces"][0], [0.0, (80 + 3.0 * 9.81) + 3, 0.0, 0.0]
    )
    assert np.allclose(
        prox.data["ProximalMoments"][0],
        [0.0, 0.0, (160 + 3.0 * 9.81) + 3.12, 0.0],
    )

    # Test 3: Like test 2 but by swapping x and z (Fz <--> Fx, -Mx <--> Mz)
    ts = ktk.TimeSeries(time=np.array([0]))
    ts.data["ProximalJointPosition"] = np.array([[0, 0, 0, 1]])
    ts.data["DistalJointPosition"] = np.array([[0, 0, 2, 1]])
    ts.data["COMPosition"] = np.array([[0, 0, 1, 1]])
    ts.data["ForceApplicationPosition"] = np.array([[0, 0, 2, 1]])
    ts.data["DistalForces"] = np.array([[0, 80, 0, 0]])
    ts.data["DistalMoments"] = np.array([[0, 0, 0, 0]])

    inertial_constants = {
        "Mass": 3,
        "GyrationCOMRatio": 0.1,
    }

    ts.data["COMAcceleration"] = np.array([[0, 1, 0, 0]])
    ts.data["AngularVelocity"] = np.array([[0, 0, 0]])
    ts.data["AngularAcceleration"] = np.array([[-1, 0, 0]])

    prox = ktk.inversedynamics.calculate_proximal_wrench(
        ts, inertial_constants
    )

    assert np.allclose(
        prox.data["ProximalForces"][0], [0.0, (80 + 3.0 * 9.81) + 3, 0.0, 0.0]
    )
    assert np.allclose(
        prox.data["ProximalMoments"][0],
        [-((160 + 3.0 * 9.81) + 3.12), 0.0, 0.0, 0.0],
    )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
