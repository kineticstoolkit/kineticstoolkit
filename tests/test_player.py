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
Tests for ktk.Player
"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2024 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"

import kineticstoolkit as ktk
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os


def init() -> bool:
    """
    Initialize the graphic aggrator to Qt5Agg.

    This tries to ask Matplotlib to use Qt5Agg, which sets the interactive
    mode and therefore prevents a warning telling that Player must be used
    in interactive mode.

    If it fails, it's ok, unless we are running on macOS. For weird reasons,
    macOS headless mode on GitHub's continuous integration fails with bus
    errors or segmentation faults. Since KTK is primlarily developed on macOS,
    it does not bother me that the Player is not tested specifically on macOS
    during continuous integration, because it is tested locally. It is still
    tested on Linux and Windows.

    Returns
    -------
    bool
        True if the test must be run, False if it must not be run.

    """
    try:
        mpl.use("Qt5Agg")
    except ImportError:
        if ktk.config.is_mac:
            return False
    return True


def test_instanciate():
    """Test that instanciating a Player does not crash."""
    if not init():
        return

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


def test_issue137():
    """
    Player should not fail if some TimeSeries are not Nx4 or Nx4x4
    """
    if not init():
        return

    kinematics = ktk.load(
        ktk.doc.download("inversedynamics_kinematics.ktk.zip")
    )
    kinematics = kinematics["Kinematics"]["Markers"]
    kinematics.data["test"] = kinematics.time
    pl = ktk.Player(kinematics)  # Shouldn't crash
    plt.pause(0.2)
    pl.close()


def test_scripting():
    """Test that every property assignation works or crashes as expected."""
    # %%
    # Download and read markers from a sample C3D file
    if not init():
        return
    # %%

    filename = ktk.doc.download("kinematics_tennis_serve.c3d")
    markers = ktk.read_c3d(filename)["Points"]

    # Create another person
    markers2 = markers.copy()
    keys = list(markers2.data.keys())
    for key in keys:
        markers2.data[key] += [[2.0, 2.0, 0.0, 0.0]]
        markers2.rename_data(
            key, key.replace("Derrick", "Viktor"), in_place=True
        )

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
    p = ktk.Player(
        markers,
        markers2,
        up="z",
        anterior="-y",
        interconnections=interconnections,
    )

    # Check that the leading wildcard propagated well to the two subjects
    # Trailign wildcard is tested in test_interconnection_wildcard_as_suffix
    try:
        assert p._extended_interconnections["Derrick:LLowerLimb"]["Links"] == [
            ["Derrick:LTOE", "Derrick:LHEE", "Derrick:LANK", "Derrick:LTOE"],
            ["Derrick:LANK", "Derrick:LKNE", "Derrick:LASI"],
            ["Derrick:LKNE", "Derrick:LPSI"],
        ]
        assert p._extended_interconnections["Derrick:RLowerLimb"]["Links"] == [
            ["Derrick:RTOE", "Derrick:RHEE", "Derrick:RANK", "Derrick:RTOE"],
            ["Derrick:RANK", "Derrick:RKNE", "Derrick:RASI"],
            ["Derrick:RKNE", "Derrick:RPSI"],
        ]
        assert p._extended_interconnections["Derrick:LUpperLimb"]["Links"] == [
            ["Derrick:LSHO", "Derrick:LELB", "Derrick:LWRA", "Derrick:LFIN"],
            ["Derrick:LELB", "Derrick:LWRB", "Derrick:LFIN"],
            ["Derrick:LWRA", "Derrick:LWRB"],
        ]
        assert p._extended_interconnections["Derrick:RUpperLimb"]["Links"] == [
            ["Derrick:RSHO", "Derrick:RELB", "Derrick:RWRA", "Derrick:RFIN"],
            ["Derrick:RELB", "Derrick:RWRB", "Derrick:RFIN"],
            ["Derrick:RWRA", "Derrick:RWRB"],
        ]
        assert p._extended_interconnections["Derrick:Head"]["Links"] == [
            ["Derrick:C7", "Derrick:LFHD", "Derrick:RFHD", "Derrick:C7"],
            ["Derrick:C7", "Derrick:LBHD", "Derrick:RBHD", "Derrick:C7"],
            ["Derrick:LBHD", "Derrick:LFHD"],
            ["Derrick:RBHD", "Derrick:RFHD"],
        ]
        assert p._extended_interconnections["Derrick:TrunkPelvis"][
            "Links"
        ] == [
            ["Derrick:LASI", "Derrick:STRN", "Derrick:RASI"],
            ["Derrick:STRN", "Derrick:CLAV"],
            ["Derrick:LPSI", "Derrick:T10", "Derrick:RPSI"],
            ["Derrick:T10", "Derrick:C7"],
            ["Derrick:LASI", "Derrick:LSHO", "Derrick:LPSI"],
            ["Derrick:RASI", "Derrick:RSHO", "Derrick:RPSI"],
            [
                "Derrick:LPSI",
                "Derrick:LASI",
                "Derrick:RASI",
                "Derrick:RPSI",
                "Derrick:LPSI",
            ],
            [
                "Derrick:LSHO",
                "Derrick:CLAV",
                "Derrick:RSHO",
                "Derrick:C7",
                "Derrick:LSHO",
            ],
        ]
        assert p._extended_interconnections["Viktor:LLowerLimb"]["Links"] == [
            ["Viktor:LTOE", "Viktor:LHEE", "Viktor:LANK", "Viktor:LTOE"],
            ["Viktor:LANK", "Viktor:LKNE", "Viktor:LASI"],
            ["Viktor:LKNE", "Viktor:LPSI"],
        ]
        assert p._extended_interconnections["Viktor:RLowerLimb"]["Links"] == [
            ["Viktor:RTOE", "Viktor:RHEE", "Viktor:RANK", "Viktor:RTOE"],
            ["Viktor:RANK", "Viktor:RKNE", "Viktor:RASI"],
            ["Viktor:RKNE", "Viktor:RPSI"],
        ]
        assert p._extended_interconnections["Viktor:LUpperLimb"]["Links"] == [
            ["Viktor:LSHO", "Viktor:LELB", "Viktor:LWRA", "Viktor:LFIN"],
            ["Viktor:LELB", "Viktor:LWRB", "Viktor:LFIN"],
            ["Viktor:LWRA", "Viktor:LWRB"],
        ]
        assert p._extended_interconnections["Viktor:RUpperLimb"]["Links"] == [
            ["Viktor:RSHO", "Viktor:RELB", "Viktor:RWRA", "Viktor:RFIN"],
            ["Viktor:RELB", "Viktor:RWRB", "Viktor:RFIN"],
            ["Viktor:RWRA", "Viktor:RWRB"],
        ]
        assert p._extended_interconnections["Viktor:Head"]["Links"] == [
            ["Viktor:C7", "Viktor:LFHD", "Viktor:RFHD", "Viktor:C7"],
            ["Viktor:C7", "Viktor:LBHD", "Viktor:RBHD", "Viktor:C7"],
            ["Viktor:LBHD", "Viktor:LFHD"],
            ["Viktor:RBHD", "Viktor:RFHD"],
        ]
        assert p._extended_interconnections["Viktor:TrunkPelvis"]["Links"] == [
            ["Viktor:LASI", "Viktor:STRN", "Viktor:RASI"],
            ["Viktor:STRN", "Viktor:CLAV"],
            ["Viktor:LPSI", "Viktor:T10", "Viktor:RPSI"],
            ["Viktor:T10", "Viktor:C7"],
            ["Viktor:LASI", "Viktor:LSHO", "Viktor:LPSI"],
            ["Viktor:RASI", "Viktor:RSHO", "Viktor:RPSI"],
            [
                "Viktor:LPSI",
                "Viktor:LASI",
                "Viktor:RASI",
                "Viktor:RPSI",
                "Viktor:LPSI",
            ],
            [
                "Viktor:LSHO",
                "Viktor:CLAV",
                "Viktor:RSHO",
                "Viktor:C7",
                "Viktor:LSHO",
            ],
        ]

    except:
        raise AssertionError(
            "The _extended_interconnections is not as expected."
        )

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

    # or

    p.set_view("top")

    # %% Standard view
    p.elevation = 0.1
    p.azimuth = np.pi / 4
    p.perspective = True

    # %% Styling points and interconnections
    p.default_point_color = "r"
    p.default_point_color = [1, 0, 0]
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

    # %% Play and pause
    p.play()
    plt.pause(0.5)
    p.pause()

    # %% Close
    p.close()

    # %%


def test_set_current_time():
    """Test that setting the current time on construction works."""
    if not init():
        return

    filename = ktk.doc.download("kinematics_tennis_serve.c3d")
    markers = ktk.read_c3d(filename)["Points"]
    ktk.Player(markers, current_time=2.5)


def test_to_image_video():
    """Test that to_image and to_video work."""
    if not init():
        return

    filename = ktk.doc.download("kinematics_tennis_serve.c3d")
    markers = ktk.read_c3d(filename)["Points"]
    p = ktk.Player(markers.get_ts_between_times(0, 0.1))
    p.to_video("test.mp4")
    p.to_video("test.mp4", fps=30)
    p.to_video("test.mp4", fps=30, downsample=4)
    p.to_image("test.png")
    assert os.path.exists("test.mp4")
    assert os.path.exists("test.png")
    os.remove("test.mp4")
    os.remove("test.png")


def test_interconnection_wildcards_as_suffix():
    if not init():
        return

    filename = ktk.doc.download("kinematics_tennis_serve.c3d")
    markers = ktk.read_c3d(filename)["Points"]
    keys = list(markers.data.keys())
    for key in keys:
        markers.data[key] += [[2.0, 2.0, 0.0, 0.0]]
        markers.rename_data(key, key.replace("Derrick:", ""), in_place=True)

    # Create another person
    markers2 = markers.copy()
    keys = list(markers2.data.keys())
    for key in keys:
        markers.rename_data(key, f"{key}_Derrick", in_place=True)
        markers2.data[key] += [[2.0, 2.0, 0.0, 0.0]]
        markers2.rename_data(key, f"{key}_Viktor", in_place=True)

    interconnections = dict()  # Will contain all segment definitions

    interconnections["LLowerLimb"] = {
        "Color": [0, 0.5, 1],  # In RGB format (here, greenish blue)
        "Links": [  # List of lines that span lists of markers
            ["LTOE*", "LHEE*", "LANK*", "LTOE*"],
            ["LANK*", "LKNE*", "LASI*"],
            ["LKNE*", "LPSI*"],
        ],
    }

    interconnections["RLowerLimb"] = {
        "Color": [0, 0.5, 1],
        "Links": [
            ["RTOE*", "RHEE*", "RANK*", "RTOE*"],
            ["RANK*", "RKNE*", "RASI*"],
            ["RKNE*", "RPSI*"],
        ],
    }

    interconnections["LUpperLimb"] = {
        "Color": [0, 0.5, 1],
        "Links": [
            ["LSHO*", "LELB*", "LWRA*", "LFIN*"],
            ["LELB*", "LWRB*", "LFIN*"],
            ["LWRA*", "LWRB*"],
        ],
    }

    interconnections["RUpperLimb"] = {
        "Color": [1, 0.5, 0],
        "Links": [
            ["RSHO*", "RELB*", "RWRA*", "RFIN*"],
            ["RELB*", "RWRB*", "RFIN*"],
            ["RWRA*", "RWRB*"],
        ],
    }

    interconnections["Head"] = {
        "Color": [1, 0.5, 1],
        "Links": [
            ["C7*", "LFHD*", "RFHD*", "C7*"],
            ["C7*", "LBHD*", "RBHD*", "C7*"],
            ["LBHD*", "LFHD*"],
            ["RBHD*", "RFHD*"],
        ],
    }

    interconnections["TrunkPelvis"] = {
        "Color": [0.5, 1, 0.5],
        "Links": [
            ["LASI*", "STRN*", "RASI*"],
            ["STRN*", "CLAV*"],
            ["LPSI*", "T10*", "RPSI*"],
            ["T10*", "C7*"],
            ["LASI*", "LSHO*", "LPSI*"],
            ["RASI*", "RSHO*", "RPSI*"],
            [
                "LPSI*",
                "LASI*",
                "RASI*",
                "RPSI*",
                "LPSI*",
            ],
            [
                "LSHO*",
                "CLAV*",
                "RSHO*",
                "C7*",
                "LSHO*",
            ],
        ],
    }

    # In this file, the up axis is z:
    p = ktk.Player(
        markers,
        markers2,
        up="z",
        anterior="-y",
        interconnections=interconnections,
    )

    # This check is a bit flaky because implementing the same function
    # differently would make this test fail even if the function works. But
    # I like false-positives better than false-negatives for tests.
    try:
        assert p._extended_interconnections["_ViktorLLowerLimb"]["Links"] == [
            ["LTOE_Viktor", "LHEE_Viktor", "LANK_Viktor", "LTOE_Viktor"],
            ["LANK_Viktor", "LKNE_Viktor", "LASI_Viktor"],
            ["LKNE_Viktor", "LPSI_Viktor"],
        ]
        assert p._extended_interconnections["_ViktorRLowerLimb"]["Links"] == [
            ["RTOE_Viktor", "RHEE_Viktor", "RANK_Viktor", "RTOE_Viktor"],
            ["RANK_Viktor", "RKNE_Viktor", "RASI_Viktor"],
            ["RKNE_Viktor", "RPSI_Viktor"],
        ]
        assert p._extended_interconnections["_ViktorLUpperLimb"]["Links"] == [
            ["LSHO_Viktor", "LELB_Viktor", "LWRA_Viktor", "LFIN_Viktor"],
            ["LELB_Viktor", "LWRB_Viktor", "LFIN_Viktor"],
            ["LWRA_Viktor", "LWRB_Viktor"],
        ]
        assert p._extended_interconnections["_ViktorRUpperLimb"]["Links"] == [
            ["RSHO_Viktor", "RELB_Viktor", "RWRA_Viktor", "RFIN_Viktor"],
            ["RELB_Viktor", "RWRB_Viktor", "RFIN_Viktor"],
            ["RWRA_Viktor", "RWRB_Viktor"],
        ]
        assert p._extended_interconnections["_ViktorHead"]["Links"] == [
            ["C7_Viktor", "LFHD_Viktor", "RFHD_Viktor", "C7_Viktor"],
            ["C7_Viktor", "LBHD_Viktor", "RBHD_Viktor", "C7_Viktor"],
            ["LBHD_Viktor", "LFHD_Viktor"],
            ["RBHD_Viktor", "RFHD_Viktor"],
        ]
        assert p._extended_interconnections["_ViktorTrunkPelvis"]["Links"] == [
            ["LASI_Viktor", "STRN_Viktor", "RASI_Viktor"],
            ["STRN_Viktor", "CLAV_Viktor"],
            ["LPSI_Viktor", "T10_Viktor", "RPSI_Viktor"],
            ["T10_Viktor", "C7_Viktor"],
            ["LASI_Viktor", "LSHO_Viktor", "LPSI_Viktor"],
            ["RASI_Viktor", "RSHO_Viktor", "RPSI_Viktor"],
            [
                "LPSI_Viktor",
                "LASI_Viktor",
                "RASI_Viktor",
                "RPSI_Viktor",
                "LPSI_Viktor",
            ],
            [
                "LSHO_Viktor",
                "CLAV_Viktor",
                "RSHO_Viktor",
                "C7_Viktor",
                "LSHO_Viktor",
            ],
        ]
        assert p._extended_interconnections["_DerrickLLowerLimb"]["Links"] == [
            ["LTOE_Derrick", "LHEE_Derrick", "LANK_Derrick", "LTOE_Derrick"],
            ["LANK_Derrick", "LKNE_Derrick", "LASI_Derrick"],
            ["LKNE_Derrick", "LPSI_Derrick"],
        ]
        assert p._extended_interconnections["_DerrickRLowerLimb"]["Links"] == [
            ["RTOE_Derrick", "RHEE_Derrick", "RANK_Derrick", "RTOE_Derrick"],
            ["RANK_Derrick", "RKNE_Derrick", "RASI_Derrick"],
            ["RKNE_Derrick", "RPSI_Derrick"],
        ]
        assert p._extended_interconnections["_DerrickLUpperLimb"]["Links"] == [
            ["LSHO_Derrick", "LELB_Derrick", "LWRA_Derrick", "LFIN_Derrick"],
            ["LELB_Derrick", "LWRB_Derrick", "LFIN_Derrick"],
            ["LWRA_Derrick", "LWRB_Derrick"],
        ]
        assert p._extended_interconnections["_DerrickRUpperLimb"]["Links"] == [
            ["RSHO_Derrick", "RELB_Derrick", "RWRA_Derrick", "RFIN_Derrick"],
            ["RELB_Derrick", "RWRB_Derrick", "RFIN_Derrick"],
            ["RWRA_Derrick", "RWRB_Derrick"],
        ]
        assert p._extended_interconnections["_DerrickHead"]["Links"] == [
            ["C7_Derrick", "LFHD_Derrick", "RFHD_Derrick", "C7_Derrick"],
            ["C7_Derrick", "LBHD_Derrick", "RBHD_Derrick", "C7_Derrick"],
            ["LBHD_Derrick", "LFHD_Derrick"],
            ["RBHD_Derrick", "RFHD_Derrick"],
        ]
        assert p._extended_interconnections["_DerrickTrunkPelvis"][
            "Links"
        ] == [
            ["LASI_Derrick", "STRN_Derrick", "RASI_Derrick"],
            ["STRN_Derrick", "CLAV_Derrick"],
            ["LPSI_Derrick", "T10_Derrick", "RPSI_Derrick"],
            ["T10_Derrick", "C7_Derrick"],
            ["LASI_Derrick", "LSHO_Derrick", "LPSI_Derrick"],
            ["RASI_Derrick", "RSHO_Derrick", "RPSI_Derrick"],
            [
                "LPSI_Derrick",
                "LASI_Derrick",
                "RASI_Derrick",
                "RPSI_Derrick",
                "LPSI_Derrick",
            ],
            [
                "LSHO_Derrick",
                "CLAV_Derrick",
                "RSHO_Derrick",
                "C7_Derrick",
                "LSHO_Derrick",
            ],
        ]

    except:
        raise AssertionError(
            "The _extended_interconnections dictionary is not as expected."
        )
    p.close()


def test_old_parameter_names():
    """Test the old parameter names."""
    if not init():
        return

    filename = ktk.doc.download("kinematics_tennis_serve.c3d")
    markers = ktk.read_c3d(filename)["Points"]

    p = ktk.Player(
        markers,
        segments={
            "Head": {
                "Color": [1, 0.5, 1],
                "Links": [
                    ["*C7", "*LFHD", "*RFHD", "*C7"],
                    ["*C7", "*LBHD", "*RBHD", "*C7"],
                    ["*LBHD", "*LFHD"],
                    ["*RBHD", "*RFHD"],
                ],
            }
        },
        segment_width=0,
        current_frame=10,
    )
    plt.pause(0.2)
    p.close()


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
