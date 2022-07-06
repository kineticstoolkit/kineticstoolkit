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

"""Unit tests for load and save functions."""


import kineticstoolkit as ktk
import numpy as np
import pandas as pd
import os


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
    # os.remove('test.mat')

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

    c3d = ktk.read_c3d(ktk.doc.download("kinematics_racing_static.c3d"))

    markers = c3d["Points"]
    assert c3d["Analogs"] is None

    assert markers.time_info["Unit"] == "s"
    assert markers.data_info["ForearmL1"]["Unit"] == "m"

    ktk.kinematics.write_c3d_file("test.c3d", markers)
    markers2 = ktk.kinematics.read_c3d_file("test.c3d")

    assert np.allclose(markers.data["ForearmL1"], markers2.data["ForearmL1"])
    assert np.allclose(markers.data["ForearmL1"].mean(), 0.14476261166589602)

    # Test a file with more data in it (analogs, events)
    filename = ktk.doc.download("walk.c3d")
    c3d = ktk.read_c3d(filename)

    assert len(c3d["Points"].time) == 221
    assert len(c3d["Points"].data) == 96
    assert len(c3d["Points"].events) == 8
    assert len(c3d["Analogs"].time) == 4420
    assert len(c3d["Analogs"].data) == 248
    assert len(c3d["Analogs"].events) == 8

    assert (
        c3d["Points"].get_index_at_time(
            c3d["Points"].get_event_time("Foot Strike", 0)
        )
        == 14
    )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
