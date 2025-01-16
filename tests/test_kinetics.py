#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2025 Félix Chénier

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for ktk.dev.kinetics."""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2025 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"

import kineticstoolkit as ktk
import numpy as np


def test_calculate_cop():
    # For now, a non-regression test based on a visual comparison between
    # the result and the COP returned by ezc3d. The discrepancy led to
    # this ezc3d issue: https://github.com/pyomeca/ezc3d/issues/354
    # While it's reviewed, I trust more my calculation than ezc3d and it's
    # temporarily a simple non-regression test.
    contents = ktk.read_c3d(
        ktk.doc.download("c3d_test_suite/ezc3d/BTS.c3d"),
        convert_point_unit=True,
    )
    lcs = contents["ForcePlatforms"].data["FP0_LCS"]
    force = contents["ForcePlatforms"].data["FP0_Force"]
    moment = contents["ForcePlatforms"].data["FP0_Moment"]

    local_force = ktk.geometry.get_local_coordinates(force, lcs)
    local_moment = ktk.geometry.get_local_coordinates(moment, lcs)
    z = -ktk.geometry.get_local_coordinates(
        contents["ForcePlatforms"].data["FP0_Corner1"], lcs
    )[0, 2]
    local_cop = ktk.dev.kinetics.calculate_cop(
        local_force, local_moment, sensor_offset=z
    )

    reference_local_cop = ktk.geometry.get_local_coordinates(
        contents["ForcePlatforms"].data["FP0_COP"], lcs
    )

    # Code to generate comparison in
    # https://github.com/pyomeca/ezc3d/issues/354

    # plt.plot(local_cop[:, 0], label="CoPx (with sensor offset)")
    # plt.plot(local_cop[:, 1], label="CoPy (with sensor offset)")
    # plt.plot(local_cop[:, 2], label="CoPz (with sensor offset)")

    # plt.plot(reference_local_cop[:, 0], label="CoPx (ezc3d)")
    # plt.plot(reference_local_cop[:, 1], label="CoPy (ezc3d)")
    # plt.plot(reference_local_cop[:, 2], label="CoPz (ezc3d)")

    # plt.legend()

    assert np.allclose(np.nanmean(local_cop), 0.8128015215784059)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
