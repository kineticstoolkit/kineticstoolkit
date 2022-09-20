#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020 Félix Chénier
#
# This file is not for redistribution.
"""
Unit tests for ktk.kinematics.

For now this is the original tutorial without any additional check.
"""

import kineticstoolkit as ktk
import numpy as np
import warnings
import os


def test_read_n3d_file_deprecated():
    """Regression test."""
    markers = ktk.kinematics.read_n3d_file(
        ktk.doc.download("kinematics_sample_optotrak.n3d")
    )

    tol = 1e-4
    assert np.abs(np.sum(markers.data["Marker0"]) - 172.3365) < tol
    assert np.abs(np.sum(markers.data["Marker40"]) + 45.3753) < tol
    assert markers.time_info["Unit"] == "s"
    assert markers.data_info["Marker40"]["Unit"] == "m"

    labels = [
        "Probe1",
        "Probe2",
        "Probe3",
        "Probe4",
        "Probe5",
        "Probe6",
        "FRArrD",
        "FRArrG",
        "FRav",
        "ScapulaG1",
        "ScapulaG2",
        "ScapulaG3",
        "ScapulaD1",
        "ScapulaD2",
        "ScapulaD3",
        "Tete1",
        "Tete2",
        "Tete3",
        "Sternum",
        "BrasG1",
        "BrasG2",
        "BrasG3",
        "EpicondyleLatG",
        "AvBrasG1",
        "AvBrasG2",
        "AvBrasG3",
        "NAG",
        "GantG1",
        "GantG2",
        "GantG3",
        "BrasD1",
        "BrasD2",
        "BrasD3",
        "EpicondyleLatD",
        "AvBrasD1",
        "AvBrasD2",
        "AvBrasD3",
        "NAD",
        "GantD1",
        "GantD2",
        "GantD3",
    ]

    markers = ktk.kinematics.read_n3d_file(
        ktk.doc.download("kinematics_sample_optotrak.n3d"),
        labels=labels,
    )

    assert np.abs(np.sum(markers.data["Probe1"]) - 172.3365) < tol
    assert np.abs(np.sum(markers.data["GantD3"]) + 45.3753) < tol
    assert markers.time_info["Unit"] == "s"
    assert markers.data_info["GantD3"]["Unit"] == "m"


def test_reconstruction_deprecated():
    """Simplified copy of the tutorial."""

    # Read the markers
    markers = ktk.kinematics.read_c3d_file(
        ktk.doc.download("kinematics_racing_propulsion.c3d")
    )
    rigid_body_definitions = dict()

    # Read the static trial
    markers_static = ktk.kinematics.read_c3d_file(
        ktk.doc.download("kinematics_racing_static.c3d")
    )

    rigid_body_definitions["ArmR"] = ktk.kinematics.define_rigid_body(
        markers_static, marker_names=["ArmR1", "ArmR2", "ArmR3"]
    )

    rigid_body_definitions["ForearmR"] = ktk.kinematics.define_rigid_body(
        markers_static,
        marker_names=[
            "ForearmR1",
            "ForearmR2",
            "ForearmR3",
            "LateralEpicondyleR",
        ],
    )

    rigid_body_definitions["Probe"] = {
        "Probe1": np.array([[0.0021213, -0.0158328, 0.0864285, 1.0]]),
        "Probe2": np.array([[0.0021213, 0.0158508, 0.0864285, 1.0]]),
        "Probe3": np.array([[0.0020575, 0.0160096, 0.1309445, 1.0]]),
        "Probe4": np.array([[0.0021213, 0.0161204, 0.1754395, 1.0]]),
        "Probe5": np.array([[0.0017070, -0.0155780, 0.1753805, 1.0]]),
        "Probe6": np.array([[0.0017762, -0.0156057, 0.1308888, 1.0]]),
    }

    def process_probing_acquisition(file_name, rigid_body_name):

        # Load the markers
        markers_probing = ktk.kinematics.read_c3d_file(file_name)

        # Track the reference rigid body
        rigid_bodies_probing = ktk.kinematics.track_rigid_body(
            markers_probing,
            local_points=rigid_body_definitions[rigid_body_name],
            label=rigid_body_name,
        )

        # Track the probe
        rigid_bodies_probing = rigid_bodies_probing.merge(
            ktk.kinematics.track_rigid_body(
                markers_probing,
                local_points=rigid_body_definitions["Probe"],
                label="Probe",
            )
        )

        # Return the local coordinates of the probed virtual marker
        return ktk.kinematics.define_local_position(
            rigid_bodies_probing,
            source_name="Probe",
            rigid_body_name=rigid_body_name,
        )

    rigid_body_definitions["ArmR"]["AcromionR"] = process_probing_acquisition(
        ktk.doc.download("kinematics_racing_probing_acromion_R.c3d"),
        "ArmR",
    )

    rigid_body_definitions["ArmR"][
        "MedialEpicondyleR"
    ] = process_probing_acquisition(
        ktk.doc.download("kinematics_racing_probing_medial_epicondyle_R.c3d"),
        "ArmR",
    )

    rigid_body_definitions["ForearmR"][
        "RadialStyloidR"
    ] = process_probing_acquisition(
        ktk.doc.download("kinematics_racing_probing_radial_styloid_R.c3d"),
        "ForearmR",
    )

    rigid_body_definitions["ForearmR"][
        "UlnarStyloidR"
    ] = process_probing_acquisition(
        ktk.doc.download("kinematics_racing_probing_ulnar_styloid_R.c3d"),
        "ForearmR",
    )

    rigid_bodies = ktk.TimeSeries()

    for rigid_body_name in ["ArmR", "ForearmR"]:
        rigid_bodies = rigid_bodies.merge(
            ktk.kinematics.track_rigid_body(
                markers,
                local_points=rigid_body_definitions[rigid_body_name],
                label=rigid_body_name,
                include_markers=True,
            )
        )


def test_reconstruction():
    """Simplified copy of the tutorial."""
    # Read the markers
    markers = ktk.kinematics.read_c3d_file(
        ktk.doc.download("kinematics_racing_propulsion.c3d")
    )

    clusters = dict()

    # Read the static trial
    markers_static = ktk.kinematics.read_c3d_file(
        ktk.doc.download("kinematics_racing_static.c3d")
    )

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
        markers_probing = ktk.kinematics.read_c3d_file(file_name)

        # Find the probe tip
        markers_probing = ktk.kinematics.track_cluster(
            markers_probing, clusters["Probe"]
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
            ktk.kinematics.track_cluster(markers, clusters[cluster])
        )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
