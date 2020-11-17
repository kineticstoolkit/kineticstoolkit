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

def test_calculate_proximal_wrenches_2d_static():
    """
    Test calculate_proximal_wrenches for a 2d static case

                    |  80 N
                    |                 y ^
                    |                   |
         m=3kg     \ /                  |
    o======1=======2m                    -----> x

    """
    n_points = 100
    ts = ktk.TimeSeries(time=np.linspace(0, 10, n_points))
    ts.data['ProximalJointPosition'] = np.repeat(
        np.array([[0, 0, 0, 1]]), n_points, axis=0)
    ts.data['DistalJointPosition'] = np.repeat(
        np.array([[2, 0, 0, 1]]), n_points, axis=0)
    ts.data['ForceApplicationPosition'] = np.repeat(
        np.array([[2, 0, 0, 1]]), n_points, axis=0)
    ts.data['DistalForces'] = np.repeat(
        np.array([[0, 80, 0, 0]]), n_points, axis=0)
    ts.data['DistalMoments'] = np.repeat(
        np.array([[0, 0, 0, 0]]), n_points, axis=0)

    inertial_constants = {
        'Mass': 3,
        'COMProximalRatio': 0.5,
        'GyrationCOMRatio': 0.1,
    }

    prox = ktk.inversedynamics.calculate_proximal_wrench(
        ts, inertial_constants)

    assert np.all(np.abs(
        np.median(prox.data['ProximalForces'], axis=0) - np.array(
        [0., 80 + 3. * 9.81, 0., 0.])) < 1E-10)
    assert np.all(np.abs(
        np.median(prox.data['ProximalMoments'], axis=0) - np.array(
        [0., 0., 160 + 3. * 9.81, 0.])) < 1E-10)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
