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
    # Test that calculate_cop gives the same results as ezc3d
    for file in (
        "c3d_test_suite/ezc3d/BTS.c3d",
        "c3d_test_suite/ezc3d/FP_Type3.c3d",
        "c3d_test_suite/ezc3d/Label2.c3d",
        "c3d_test_suite/ezc3d/Qualisys.c3d",
    ):

        contents = ktk.read_c3d(
            ktk.doc.download(file),
            convert_point_unit=True,
        )

        lcs = contents["ForcePlatforms"].data["FP0_LCS"]
        force = contents["ForcePlatforms"].data["FP0_Force"]
        moment = contents["ForcePlatforms"].data["FP0_MomentAtCenter"]

        cop = ktk.geometry.get_global_coordinates(
            ktk.dev.kinetics.calculate_cop(
                ktk.geometry.get_local_coordinates(force, lcs),
                ktk.geometry.get_local_coordinates(moment, lcs),
            ),
            lcs,
        )

        ref_cop = contents["ForcePlatforms"].data["FP0_COP"]

        assert np.sum(~np.isnan(cop[:, 0])) > 0  # Not only nans
        assert np.sum(~np.isnan(ref_cop[:, 0])) > 0  # Not only nans

        nonnan = ~(np.isnan(cop[:, 0]) | np.isnan(ref_cop[:, 0]))
        assert np.sum(nonnan) > 0  # Not only nans

        # Tolerance of half a millimeter for FP_Type3, I don't know why there
        # is this very small offset but I guess it's a negligible error.
        assert np.allclose(cop[nonnan], ref_cop[nonnan], atol=0.0005)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
