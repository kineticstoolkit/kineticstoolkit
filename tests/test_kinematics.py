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
"""
Unit tests for ktk.kinematics.

For now this is the original tutorial without any additional check.

"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2024 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"

import kineticstoolkit as ktk
import numpy as np


def test_reconstruction():
    """Simplified copy of the tutorial."""
    # Read the markers
    markers = ktk.read_c3d(
        ktk.doc.download("kinematics_racing_propulsion.c3d")
    )["Points"]

    clusters = dict()

    # Read the static trial
    markers_static = ktk.read_c3d(
        ktk.doc.download("kinematics_racing_static.c3d")
    )["Points"]

    clusters["ArmR"] = ktk.kinematics.create_cluster(
        markers_static,
        names=["ArmR1", "ArmR2", "ArmR3", "LateralEpicondyleR"],
    )

    clusters["ForearmR"] = ktk.kinematics.create_cluster(
        markers_static, names=["ForearmR1", "ForearmR2", "ForearmR3"]
    )

    clusters["Probe"] = {
        "ProbeTip": np.array([[0.0, 0.0, 0.0, 1.0]]),
        "Probe1": np.array([[0.0021213, -0.0158328, 0.0864285, 1.0]]),
        "Probe2": np.array([[0.0021213, 0.0158508, 0.0864285, 1.0]]),
        "Probe3": np.array([[0.0020575, 0.0160096, 0.1309445, 1.0]]),
        "Probe4": np.array([[0.0021213, 0.0161204, 0.1754395, 1.0]]),
        "Probe5": np.array([[0.0017070, -0.0155780, 0.1753805, 1.0]]),
        "Probe6": np.array([[0.0017762, -0.0156057, 0.1308888, 1.0]]),
    }

    def process_probing_acquisition(file_name, cluster, point_name):
        # Load the markers
        markers_probing = ktk.read_c3d(file_name)["Points"]

        # Find the probe tip
        markers_probing.merge(
            ktk.kinematics.track_cluster(
                markers_probing, clusters["Probe"]
            ).get_subset("ProbeTip"),
            in_place=True,
        )

        # Test bugfix #85
        # Units and other data_info lost in kinetics.track_cluster()
        assert markers_probing.data_info["ProbeTip"]["Unit"] == "m"

        # Extend the cluster
        markers_probing.rename_data("ProbeTip", point_name, in_place=True)
        cluster = ktk.kinematics.extend_cluster(
            markers_probing, cluster, point_name
        )

        return cluster

    clusters["ArmR"] = process_probing_acquisition(
        ktk.doc.download("kinematics_racing_probing_acromion_R.c3d"),
        clusters["ArmR"],
        "AcromionR",
    )

    clusters["ArmR"] = process_probing_acquisition(
        ktk.doc.download("kinematics_racing_probing_medial_epicondyle_R.c3d"),
        clusters["ArmR"],
        "MedialEpicondyleR",
    )

    clusters["ForearmR"] = process_probing_acquisition(
        ktk.doc.download("kinematics_racing_probing_radial_styloid_R.c3d"),
        clusters["ForearmR"],
        "RadialStyloidR",
    )

    clusters["ForearmR"] = process_probing_acquisition(
        ktk.doc.download("kinematics_racing_probing_ulnar_styloid_R.c3d"),
        clusters["ForearmR"],
        "UlnarStyloidR",
    )

    for cluster in clusters:
        markers = markers.merge(
            ktk.kinematics.track_cluster(markers, clusters[cluster]),
            overwrite=False,
            on_conflict="mute",
        )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
