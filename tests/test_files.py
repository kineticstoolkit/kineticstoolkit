#!/usr/bin/env python3
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

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2024 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"
"""Unit tests for load and save functions."""


import kineticstoolkit as ktk
import numpy as np
import pandas as pd
import os
import warnings


def test_save_load():
    """Test the save and load functions."""
    # Create a test variable with all possible supported combinations
    random_variable = np.random.rand(5, 2, 2)
    ts = ktk.TimeSeries()
    ts.time = np.linspace(0, 9, 10)
    ts.data["signal1"] = np.random.rand(10)
    ts.data["signal2"] = np.random.rand(10, 3)
    ts.data["signal3"] = np.random.rand(10, 3, 3)
    ts = ts.add_data_info("signal1", "Unit", "m/s")
    ts = ts.add_data_info("signal2", "Unit", "km/h")
    ts = ts.add_data_info("signal3", "Unit", "N")
    ts = ts.add_data_info("signal3", "SignalType", "force")
    ts = ts.add_event(1.53, "TestEvent1")
    ts = ts.add_event(7.2, "TestEvent2")
    ts = ts.add_event(1, "TestEvent3")

    a = dict()
    a["TestTimeSeries"] = ts
    a["TestInt"] = 10
    a["TestFloat"] = np.pi
    a["TestBool"] = True
    a["TestStr"] = """Test string with 'quotes' and "double quotes"."""
    a["TestComplex"] = 34.05 + 2j
    a["TestArray"] = random_variable
    a["TestList"] = [0, "test", True]
    #    a['TestTuple'] = (1, 'test2', False)
    a["TestBigList"] = np.arange(-1, 1, 1e-4).tolist()
    a["TestDict"] = {"key1": "value1", "key2": 10, "key3": True}
    a["TestComplexDict"] = {"key1": "value1", "key2": 10, "key3": None}
    a["TestComplexList"] = [
        "value1",
        10,
        ts.copy(),
        "value2",
        12,
        None,
        True,
        np.pi,
        (34.05 + 2j),
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        {"key1": "value1", "key2": 10, "key3": ts.copy()},
    ]
    a["TestComplexTuple"] = tuple(a["TestComplexList"])
    a["TestDataFrame"] = ts.to_dataframe()
    a["TestSeries"] = a["TestDataFrame"]["signal1"]

    ktk.save("test.ktk.zip", a)
    b = ktk.load("test.ktk.zip")

    assert a["TestTimeSeries"] == b["TestTimeSeries"]
    assert a["TestInt"] == b["TestInt"]
    assert a["TestFloat"] == b["TestFloat"]
    assert a["TestBool"] == b["TestBool"]
    assert a["TestStr"] == b["TestStr"]
    assert a["TestComplex"] == b["TestComplex"]
    assert np.sum(np.abs(a["TestArray"] - b["TestArray"])) < 1e-10
    assert a["TestList"] == b["TestList"]
    #    assert a['TestTuple'] == b['TestTuple']
    assert a["TestBigList"] == b["TestBigList"]
    assert a["TestDict"] == b["TestDict"]
    assert a["TestComplexDict"]["key1"] == b["TestComplexDict"]["key1"]
    assert a["TestComplexDict"]["key2"] == b["TestComplexDict"]["key2"]
    assert a["TestComplexDict"]["key3"] == b["TestComplexDict"]["key3"]
    for i in range(10):
        assert a["TestComplexList"][i] == b["TestComplexList"][i]
    assert a["TestComplexList"][10] == b["TestComplexList"][10]
    for i in range(10):
        assert a["TestComplexTuple"][i] == b["TestComplexTuple"][i]
    assert a["TestComplexTuple"][10] == b["TestComplexTuple"][10]
    pd.testing.assert_frame_equal(a["TestDataFrame"], b["TestDataFrame"])
    pd.testing.assert_series_equal(a["TestSeries"], b["TestSeries"])

    c = "full_standard_test"
    ktk.save("test.ktk.zip", c)
    d = ktk.load("test.ktk.zip")
    assert d == c


def test_read_c3d():
    """Test read_c3d."""
    # Read the same file as the older kinematics.read_c3d_file

    # The warning thing is just to test that loading a file that were saved
    # in mm rather than m launches the warning. I want to ensure that launching
    # the warning doesn't crash something (e.g., inexistent strings in the
    # f-strings).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c3d = ktk.read_c3d(ktk.doc.download("kinematics_racing_static.c3d"))

    markers = c3d["Points"]
    assert "Analogs" not in c3d

    assert markers.time_info["Unit"] == "s"
    assert markers.data_info["ForearmL1"]["Unit"] == "m"

    ktk.write_c3d("test.c3d", markers)
    markers2 = ktk.read_c3d("test.c3d")["Points"]

    assert np.allclose(markers.data["ForearmL1"], markers2.data["ForearmL1"])
    assert np.allclose(markers.data["ForearmL1"].mean(), 0.14476261166589602)

    # --------------------------------------------------
    # Test a file with more data in it (analogs, events)
    filename = ktk.doc.download("walk.c3d")
    c3d = ktk.read_c3d(filename, convert_point_unit=True)

    assert len(c3d["Points"].time) == 221
    assert len(c3d["Points"].data) == 96
    assert len(c3d["Points"].events) == 8
    assert len(c3d["Analogs"].time) == 4420
    assert len(c3d["Analogs"].data) == 248
    assert len(c3d["Analogs"].events) == 8

    assert (
        c3d["Points"].get_index_at_time(
            c3d["Points"]
            .events[c3d["Points"]._get_event_index("Foot Strike", 0)]
            .time
        )
        == 14
    )


def test_read_c3d_many_analogs():
    """Test fix https://github.com/kineticstoolkit/kineticstoolkit/issues/231"""
    contents = ktk.read_c3d(
        ktk.doc.download("c3d_test_suite/others/many_analogs.c3d"),
        convert_point_unit=False,
    )
    assert len(contents["Analogs"].data) == 922


def test_read_c3d_testsuite1():
    """Run the c3d.org test suite 1 and check if every file is equivalent."""
    # We do not test for mips files because it's not supported by ezc3d
    test = []
    for key in ["pi", "pr", "vi", "vr"]:
        test.append(
            ktk.read_c3d(
                ktk.doc.download(f"c3d_test_suite/Sample01/Eb015{key}.c3d"),
                convert_point_unit=True,
            )
        )
    for i in range(1, 4):
        assert test[i]["Points"]._is_equivalent(test[0]["Points"], equal=False)
        assert test[i]["Analogs"]._is_equivalent(
            test[0]["Analogs"], equal=False
        )


def test_read_c3d_testsuite2():
    """Run the c3d.org test suite 2 and check if every file is equivalent."""
    # We do not test for mips files because it's not supported by ezc3d
    #
    # Note: we selected a 1mm tolerance because it seems that the files have a
    # some glitch data in the <1mm range, most probably due to int/real
    # representation.
    test = []
    for key in [
        "dec_int",
        "dec_real",
        "pc_int",
        "pc_real",
    ]:
        test.append(
            ktk.read_c3d(
                ktk.doc.download(f"c3d_test_suite/Sample02/{key}.c3d"),
                convert_point_unit=True,
            )
        )
    for i in range(1, 4):
        assert test[i]["Points"]._is_equivalent(
            test[0]["Points"], equal=False, atol=1e-3
        )
        assert test[i]["Analogs"]._is_equivalent(
            test[0]["Analogs"], equal=False
        )


def test_read_c3d_testsuite8():
    """Run the c3d.org test suite 8 and check if every file is equivalent."""
    test = []
    for key in [
        "EB015PI",
        "TESTAPI",
        "TESTBPI",
        "TESTCPI",
        "TESTDPI",
    ]:
        test.append(
            ktk.read_c3d(
                ktk.doc.download(f"c3d_test_suite/Sample08/{key}.c3d"),
                convert_point_unit=True,
            )
        )
    for i in range(1, 5):
        assert test[i]["Points"]._is_equivalent(test[0]["Points"], equal=False)
        assert test[i]["Analogs"]._is_equivalent(
            test[0]["Analogs"], equal=False
        )


# Note: We do not run testsuite36 because floats in FRAMES is not supported by
# ezc3d.


def test_read_c3d_more_than_255_analogs():
    """https://github.com/felixchenier/kineticstoolkit/issues/191"""
    test = ktk.read_c3d(
        ktk.doc.download("c3d_test_suite/others/sample_256plus_channels.c3d"),
        convert_point_unit=True,
    )
    assert len(test["Points"].data) == 18
    assert len(test["Points"].events) == 16
    assert len(test["Analogs"].data) == 275
    assert len(test["Analogs"].events) == 16


def test_read_c3d_event_name_format():
    """https://github.com/felixchenier/kineticstoolkit/issues/194"""
    # Load a file with contexts, but without reading contexts
    test = ktk.read_c3d(
        ktk.doc.download("c3d_test_suite/others/sample_256plus_channels.c3d"),
        convert_point_unit=True,
    )
    assert test["Points"].events[0].name == "Foot Strike"
    assert test["Analogs"].events[0].name == "Foot Strike"

    # Load a file with contexts, reading contexts
    test = ktk.read_c3d(
        ktk.doc.download("c3d_test_suite/others/sample_256plus_channels.c3d"),
        convert_point_unit=True,
        include_event_context=True,
    )
    assert test["Points"].events[0].name == "Right:Foot Strike"
    assert test["Analogs"].events[0].name == "Right:Foot Strike"

    # Load a file without contexts, ensure that it reads ok.
    test = ktk.read_c3d(
        ktk.doc.download("walk.c3d"),
        convert_point_unit=True,
    )
    assert test["Points"].events[0].name == "Foot Strike"
    assert test["Analogs"].events[0].name == "Foot Strike"


def test_read_c3d_duplicate_labels():
    # Non-regression test based on a sample problematic file
    contents = ktk.read_c3d(
        ktk.doc.download("c3d_test_suite/others/duplicate_labels.c3d"),
        convert_point_unit=True,
    )
    assert np.sum(contents["Points"].isnan("T8")) == 300
    assert np.sum(contents["Points"].isnan("T8_1")) == 1047
    assert np.sum(contents["Points"].isnan("T8_2")) == 1043
    assert "T8_3" not in contents["Points"].data


def test_read_c3d_force_platforms():
    # Non-regression tests based on visually inspected force platform data
    contents = ktk.read_c3d(
        ktk.doc.download("c3d_test_suite/ezc3d/BTS.c3d"),
        convert_point_unit=True,
    )
    assert np.allclose(
        contents["ForcePlatforms"].data["FP0_Corner1"],
        [0.00361895, 0.00218622, 0.45745718, 1.0],
    )
    assert np.allclose(
        contents["ForcePlatforms"].data["FP0_Corner2"],
        [0.00475062, 0.00153797, 0.0574593, 1.0],
    )
    assert np.allclose(
        contents["ForcePlatforms"].data["FP0_Corner3"],
        [0.60474268, -0.0010519, 0.05916098, 1.0],
    )
    assert np.allclose(
        contents["ForcePlatforms"].data["FP0_Corner4"],
        [0.60361096, -0.00040365, 0.45915887, 1.0],
    )
    assert np.allclose(
        contents["ForcePlatforms"].data["FP5_Corner1"],
        [1.80758936, -0.00559742, 0.46457355, 1.0],
    )
    assert np.allclose(
        contents["ForcePlatforms"].data["FP5_Corner2"],
        [1.80645776, -0.00494917, 0.86457147, 1.0],
    )
    assert np.allclose(
        contents["ForcePlatforms"].data["FP5_Corner3"],
        [1.2064657, -0.0023593, 0.86286981, 1.0],
    )
    assert np.allclose(
        contents["ForcePlatforms"].data["FP5_Corner4"],
        [1.20759741, -0.00300755, 0.46287192, 1.0],
    )
    assert np.allclose(
        contents["ForcePlatforms"].data["FP0_LCS"][0],
        [
            [-0.00282923, -0.99998666, -0.00432103, 0.3041808],
            [0.00162063, 0.00431645, -0.99998937, 0.00056716],
            [0.99999468, -0.0028362, 0.0016084, 0.25830909],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )
    assert np.allclose(
        np.nanmean(contents["ForcePlatforms"].data["FP0_Force"], axis=0),
        [0.26887549, 41.77700117, 1.17952925, 0.0],
    )
    assert np.allclose(
        np.nanmean(
            contents["ForcePlatforms"].data["FP0_MomentAtCenter"], axis=0
        ),
        [-5.22727694, 0.24464729, -3.25622748, 0.0],
    )
    assert np.allclose(
        np.nanmean(contents["ForcePlatforms"].data["FP0_COP"], axis=0),
        [0.23364972, 0.00107284, 0.38322433, 1.0],
    )


def test_read_write_c3d():
    """
    Test that writing and reading back a c3d file yields the same results.

    Tests twice, once using the original c3d, then saving a new c3d and
    opening again.

    """
    markers = ktk.read_c3d(ktk.doc.download("kinematics_racing_static.c3d"))[
        "Points"
    ]

    assert markers.time_info["Unit"] == "s"
    assert markers.data_info["ForearmL1"]["Unit"] == "m"

    ktk.write_c3d("test.c3d", markers)
    markers2 = ktk.read_c3d("test.c3d")["Points"]

    assert np.allclose(markers.data["ForearmL1"], markers2.data["ForearmL1"])
    assert np.allclose(markers.data["ForearmL1"].mean(), 0.14476261166589602)

    os.remove("test.c3d")


def test_read_write_c3d_with_rotations():
    c3d = ktk.read_c3d(
        ktk.doc.download("c3d_test_suite/others/C3DRotationExample.c3d")
    )
    assert "Points" not in c3d
    assert "Analogs" not in c3d
    assert "Rotations" in c3d

    rotations = c3d["Rotations"]

    # No-regression test
    assert len(rotations.data) == 21
    for key in rotations.data:
        assert key in [
            "worldbody_4X4",
            "pelvis_4X4",
            "l_thigh_4X4",
            "l_shank_4X4",
            "l_foot_4X4",
            "l_toes_4X4",
            "r_thigh_4X4",
            "r_shank_4X4",
            "r_foot_4X4",
            "r_toes_4X4",
            "torso_4X4",
            "l_clavicle_4X4",
            "l_uarm_4X4",
            "l_larm_4X4",
            "l_hand_4X4",
            "r_clavicle_4X4",
            "r_uarm_4X4",
            "r_larm_4X4",
            "r_hand_4X4",
            "neck_4X4",
            "head_4X4",
        ]
    assert len(rotations.time) == 340
    assert np.isclose(rotations.get_sample_rate(), 85.0)

    # Add an event and save back to c3d (to test write_c3d)
    event_time = rotations.time[5]
    rotations.add_event(event_time, "TestEvent", in_place=True)
    ktk.write_c3d("test.c3d", rotations=rotations)

    # Read back to test for equality
    c3d2 = ktk.read_c3d("test.c3d")
    assert np.allclose(
        rotations.data["pelvis_4X4"][:, :3, :],
        c3d2["Rotations"].data["pelvis_4X4"][:, :3, :],
        equal_nan=True,
    )
    assert c3d["Rotations"].events[0].name == c3d2["Rotations"].events[0].name
    assert np.allclose(
        c3d["Rotations"].events[0].time, c3d2["Rotations"].events[0].time
    )

    os.remove("test.c3d")


def test_write_rotations_c3d():
    """Test writing fabricated rotation data to c3d."""
    rotations = ktk.TimeSeries()
    rotations.time = np.linspace(0, 1, 240, endpoint=False)
    rotations.data["pelvis_4X4"] = ktk.geometry.create_transforms(
        "x", np.linspace(0, 2 * np.pi, 240, endpoint=False)
    )
    rotations.add_event(0.5, "TestEvent", in_place=True)

    # add some point data
    points = ktk.TimeSeries()
    points.time = np.linspace(0, 1, 120, endpoint=False)
    points.data["point1"] = np.random.rand(120, 4)
    points.data["point1"][:, 3] = 1

    ktk.write_c3d("test.c3d", rotations=rotations, points=points)
    c3d = ktk.read_c3d("test.c3d")
    assert np.allclose(
        c3d["Rotations"].data["pelvis_4X4"], rotations.data["pelvis_4X4"]
    )
    assert c3d["Points"].events[0].name == "TestEvent"
    assert np.allclose(c3d["Points"].data["point1"], points.data["point1"])
    os.remove("test.c3d")


def test_write_c3d_testsuite8():
    """
    Run the c3d.org test suite 8 and check if every file is equivalent even
    after a round-test (read-write-read).

    For now, tests with analogs are commented until the next release of
    ezc3d

    """
    test = []
    for key in [
        "EB015PI",
        "TESTAPI",
        "TESTBPI",
        "TESTCPI",
        "TESTDPI",
    ]:
        data = ktk.read_c3d(
            ktk.doc.download(f"c3d_test_suite/Sample08/{key}.c3d"),
            convert_point_unit=True,
        )
        ktk.write_c3d(
            "test.c3d",
            points=data["Points"],
            analogs=data["Analogs"],
        )
        test.append(ktk.read_c3d("test.c3d"))
    for i in range(1, 5):
        assert test[i]["Points"]._is_equivalent(test[0]["Points"], equal=False)
        assert test[i]["Analogs"]._is_equivalent(
            test[0]["Analogs"], equal=False
        )

    os.remove("test.c3d")


def test_write_c3d_weirdc3d():
    """
    Test that writing data from a weirdly formatted c3d works. This file has
    weird characters.

    ezc3d has to be patched for this test to pass.
    https://github.com/pyomeca/ezc3d/issues/264

    For now, tests with analogs are commented until the next release of ezc3d

    """
    filename = ktk.doc.download("walk.c3d")
    c3d = ktk.read_c3d(filename, convert_point_unit=True)
    ktk.write_c3d(
        "test.c3d",
        points=c3d["Points"],
        analogs=c3d["Analogs"],
    )
    c3d = ktk.read_c3d("test.c3d")

    assert len(c3d["Points"].time) == 221
    assert len(c3d["Points"].data) == 96
    assert len(c3d["Points"].events) == 8
    assert len(c3d["Analogs"].time) == 4420
    assert len(c3d["Analogs"].data) == 248
    assert len(c3d["Analogs"].events) == 8

    assert (
        c3d["Points"].get_index_at_time(
            c3d["Points"]
            .events[c3d["Points"]._get_event_index("Foot Strike", 0)]
            .time
        )
        == 14
    )
    os.remove("test.c3d")


def test_write_c3d_analogs():
    """Test the creation of a c3d file with points and analogs."""
    # When everything is clean
    points = ktk.TimeSeries(time=np.linspace(0, 1, 1 * 240, endpoint=False))
    points.data["point1"] = np.random.rand(points.time.shape[0], 4)
    points.data["point2"] = np.random.rand(points.time.shape[0], 4)
    points.data["point1"][:, 3] = 1
    points.data["point2"][:, 3] = 1

    analogs = ktk.TimeSeries(time=np.linspace(0, 1, 1 * 2400, endpoint=False))
    analogs.data["emg1"] = np.random.rand(analogs.time.shape[0])
    analogs.data["forces"] = np.random.rand(analogs.time.shape[0], 3)

    ktk.write_c3d("test.c3d", points=points, analogs=analogs)
    data = ktk.read_c3d("test.c3d")
    assert np.allclose(points.data["point1"], data["Points"].data["point1"])
    assert np.allclose(points.data["point2"], data["Points"].data["point2"])
    assert np.allclose(analogs.data["emg1"], data["Analogs"].data["emg1"])
    assert np.allclose(
        analogs.data["forces"][:, 0], data["Analogs"].data["forces[0]"]
    )
    assert np.allclose(
        analogs.data["forces"][:, 1], data["Analogs"].data["forces[1]"]
    )
    assert np.allclose(
        analogs.data["forces"][:, 2], data["Analogs"].data["forces[2]"]
    )
    os.remove("test.c3d")

    # When time vectors do not match
    points = ktk.TimeSeries(time=np.linspace(0, 10, 10 * 240, endpoint=False))
    points.data["point1"] = np.ones((points.time.shape[0], 4))
    points.data["point2"] = np.ones((points.time.shape[0], 4))

    analogs = ktk.TimeSeries(
        time=np.linspace(1, 11, 10 * 2400, endpoint=False)
    )
    analogs.data["emg1"] = np.zeros(analogs.time.shape[0])
    analogs.data["forces1"] = np.zeros((analogs.time.shape[0], 4))

    try:
        ktk.write_c3d("test.c3d", points, analogs)
        raise ValueError("This should fail.")
    except ValueError:
        pass

    # When sample rate is invalid
    points = ktk.TimeSeries(time=np.linspace(0, 1, 1 * 240, endpoint=False))
    points.data["point1"] = np.ones((points.time.shape[0], 4))
    points.data["point2"] = np.ones((points.time.shape[0], 4))

    analogs = ktk.TimeSeries(time=np.linspace(0, 1, 1 * 2401, endpoint=False))
    analogs.data["emg1"] = np.zeros(analogs.time.shape[0])
    analogs.data["forces1"] = np.zeros((analogs.time.shape[0], 4))

    try:
        ktk.write_c3d("test.c3d", points, analogs)
        raise ValueError("This should fail.")
    except ValueError:
        pass


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
