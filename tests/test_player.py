#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright Félix Chénier 2020
"""
Tests for ktk.Player
"""
import kineticstoolkit as ktk
import matplotlib.pyplot as plt
import warnings
import numpy as np


def test_instanciate_and_to_html5():
    """Test that instanciating a Player does not crash."""

    # Load markers
    kinematics = ktk.load(
        ktk.doc.download("inversedynamics_kinematics.ktk.zip")
    )

    kinematics = kinematics["Kinematics"]

    # The player can be instanciated to show markers
    pl = ktk.Player(kinematics["Markers"], target=[-5, 0, 0])
    plt.pause(0.2)
    pl.close()

    # The player can be instanciated to show rigid bodies
    pl = ktk.Player(kinematics["ReferenceFrames"], target=[-5, 0, 0])
    plt.pause(0.2)
    pl.close()

    # Or the player can be instanciated to show both markers and rigid bodies
    pl = ktk.Player(
        kinematics["Markers"],
        kinematics["ReferenceFrames"],
        target=[-5, 0, 0],
        up="z",
    )
    plt.pause(0.2)

    # Test that to_html5 doesn't crash
    assert pl.to_html5() is not None

    pl.close()


def test_issue137():
    """
    Player should complain, but not fail, if some TimeSeries are not Nx4 or
    Nx4x4
    """

    # Load markers
    kinematics = ktk.load(
        ktk.doc.download("inversedynamics_kinematics.ktk.zip")
    )

    kinematics = kinematics["Kinematics"]["Markers"]
    kinematics.data["test"] = kinematics.time

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        pl = ktk.Player(kinematics)  # Shouldn't crash
        plt.pause(0.2)
        pl.close()


def test_scripting():
    """Test that every property assignation works or crashes as expected."""
    # %%
    # Download and read markers from a sample C3D file
    filename = ktk.doc.download("kinematics_tennis_serve.c3d")
    markers = ktk.read_c3d(filename)["Points"]

    # # Create another person
    # markers2 = markers.copy()
    # keys = list(markers2.data.keys())
    # for key in keys:
    #     markers2.data[key] += [[1.0, 0.0, 0.0, 0.0]]
    #     markers2.rename_data(key, key.replace("Derrick", "Viktor"), in_place=True)

    interconnections = dict()  # Will contain all segment definitions

    interconnections["LLowerLimb"] = {
        "Color": [0, 0.5, 1],  # In RGB format (here, greenish blue)
        "Links": [  # List of lines that span lists of markers
            ["*LTOE", "*LHEE", "*LANK", "*LTOE"],
            ["*LANK", "*LKNE", "*LASI"],
            ["*LKNE", "*LPSI"],
        ],
    }

    interconnections["RLowerLimb"] = {
        "Color": [0, 0.5, 1],
        "Links": [
            ["*RTOE", "*RHEE", "*RANK", "*RTOE"],
            ["*RANK", "*RKNE", "*RASI"],
            ["*RKNE", "*RPSI"],
        ],
    }

    interconnections["LUpperLimb"] = {
        "Color": [0, 0.5, 1],
        "Links": [
            ["*LSHO", "*LELB", "*LWRA", "*LFIN"],
            ["*LELB", "*LWRB", "*LFIN"],
            ["*LWRA", "*LWRB"],
        ],
    }

    interconnections["RUpperLimb"] = {
        "Color": [1, 0.5, 0],
        "Links": [
            ["*RSHO", "*RELB", "*RWRA", "*RFIN"],
            ["*RELB", "*RWRB", "*RFIN"],
            ["*RWRA", "*RWRB"],
        ],
    }

    interconnections["Head"] = {
        "Color": [1, 0.5, 1],
        "Links": [
            ["*C7", "*LFHD", "*RFHD", "*C7"],
            ["*C7", "*LBHD", "*RBHD", "*C7"],
            ["*LBHD", "*LFHD"],
            ["*RBHD", "*RFHD"],
        ],
    }

    interconnections["TrunkPelvis"] = {
        "Color": [0.5, 1, 0.5],
        "Links": [
            ["*LASI", "*STRN", "*RASI"],
            ["*STRN", "*CLAV"],
            ["*LPSI", "*T10", "*RPSI"],
            ["*T10", "*C7"],
            ["*LASI", "*LSHO", "*LPSI"],
            ["*RASI", "*RSHO", "*RPSI"],
            [
                "*LPSI",
                "*LASI",
                "*RASI",
                "*RPSI",
                "*LPSI",
            ],
            [
                "*LSHO",
                "*CLAV",
                "*RSHO",
                "*C7",
                "*LSHO",
            ],
        ],
    }

    # In this file, the up axis is z:
    p = ktk.Player(markers, up="z", interconnections=interconnections)

    # %% Front view
    p.azimuth = 0.0
    p.elevation = 0.0
    p.target = (0.0, 0.0, 0.0)
    p.perspective = False

    # %% Right view
    p.azimuth = np.pi / 2

    # %% Top view
    p.azimuth = 0.0
    p.elevation = np.pi / 2

    # %% Standard view
    p.elevation = 0.1
    p.azimuth = np.pi / 4
    p.perspective = True

    # %% Styling points and interconnections
    p.point_size = 8.0
    p.interconnection_width = 5.0

    # %% Styling frames
    p.frame_size = 1.0
    p.frame_width = 3.0

    # %% Sizing grid
    p.grid_size = 8.0
    p.grid_origin = (0, -1.0, 0.0)

    # %% Styling grid and background
    p.grid_color = (1.0, 0.5, 1.0)
    p.background_color = (1.0, 1.0, 1.0)

    # %% Styling grid and background
    p.grid_color = (0.0, 0.5, 0.5)
    p.background_color = (0.0, 0.0, 0.2)

    # %% Remove data
    ts = p.get_contents()
    p.set_contents(ktk.TimeSeries())

    # %% Put back data
    p.set_contents(ts)

    # %% Remove interconnections
    inter = p.get_interconnections()
    p.set_interconnections({})

    # %% Put back interconnections
    p.set_interconnections(inter)

    # %% Keep only seconds 4 to 5
    p.set_contents(markers.get_ts_between_times(4.0, 5.0))

    # %% Close
    p.close()

    # %%


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
