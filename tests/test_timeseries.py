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
These are the unit tests for the TimeSeries class.
"""
import kineticstoolkit as ktk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from kineticstoolkit.exceptions import (
    TimeSeriesRangeError,
    TimeSeriesEventNotFoundError,
)

# %% TimeSeriesEvent


def test_TimeSeriesEvent():
    """Test TimeSeriesEvent."""

    # Test basic getters and setters
    event = ktk.TimeSeriesEvent()
    event.time = 1
    event.name = "one"
    assert event.time == 1
    assert event.name == "one"

    # Test ordering
    event1 = ktk.TimeSeriesEvent(time=1, name="event1")
    event2 = ktk.TimeSeriesEvent(time=2, name="event2")
    event3 = ktk.TimeSeriesEvent(time=2, name="event3")

    assert event1 < event2
    assert event1 <= event2
    assert not event2 < event3
    assert event2 <= event3

    assert event2 > event1
    assert event2 >= event1
    assert not event3 > event2
    assert event3 >= event2

    # Test _to_list and _to_dict
    event = ktk.TimeSeriesEvent(time=1.5, name="event_name")
    the_list = event._to_list()
    assert the_list[0] == 1.5
    assert the_list[1] == "event_name"

    the_dict = event._to_dict()
    assert the_dict["Time"] == 1.5
    assert the_dict["Name"] == "event_name"


# %% TimeSeries constructo, copy and checks


def test_TimeSeries():
    """Test basic functions."""
    ts = ktk.TimeSeries()
    assert isinstance(dir(ts), list)
    assert isinstance(repr(ts), str)
    assert isinstance(str(ts), str)

    # Test equality
    ts2 = ktk.TimeSeries()
    assert ts == ts2

    ts2.time = np.arange(10)
    assert ts != ts2
    ts.time = np.arange(10)
    assert ts == ts2

    ts2.data["test"] = np.arange(10)
    assert ts != ts2
    ts.data["test"] = np.arange(10) + 1  # Different but same size
    assert ts != ts2
    ts.data["test"] = np.arange(20)  # Different sizes
    assert ts != ts2
    ts.data["test"] = np.arange(10)  # Now the samething
    assert ts == ts2

    ts2.time_info["info"] = "test"
    assert ts != ts2
    ts.time_info["info"] = "test"
    assert ts == ts2

    ts2 = ts2.add_data_info("test", "info", "test")
    assert ts != ts2
    ts = ts.add_data_info("test", "info", "test")
    assert ts == ts2

    ts2 = ts2.add_event(1, "one")
    assert ts != ts2
    ts = ts.add_event(1, "one")
    assert ts == ts2


def test_empty_constructor():
    ts = ktk.TimeSeries()
    assert isinstance(ts.time, np.ndarray)
    assert isinstance(ts.data, dict)
    assert isinstance(ts.time_info, dict)
    assert isinstance(ts.data_info, dict)
    assert isinstance(ts.events, list)
    assert ts.time_info["Unit"] == "s"


def test_copy():
    """Test the copy method."""
    ts1 = ktk.TimeSeries()
    ts1.time = np.linspace(0, 99, 100, endpoint=False)
    ts1.data["signal1"] = np.random.rand(100, 2)
    ts1.data["signal2"] = np.random.rand(100, 2)
    ts1.data["signal3"] = np.random.rand(100, 2)
    ts1.add_data_info("signal1", "Unit", "Unit1", in_place=True)
    ts1.add_data_info("signal2", "Unit", "Unit2", in_place=True)
    ts1.add_data_info("signal3", "Unit", "Unit3", in_place=True)
    ts1.add_event(1.54, "test_event1", in_place=True)
    ts1.add_event(10.2, "test_event2", in_place=True)
    ts1.add_event(100, "test_event3", in_place=True)

    # Standard deep copy
    ts2 = ts1.copy()
    assert ts1 == ts2

    # A deep copy without data
    ts2 = ts1.copy(copy_data=False)
    assert ts1 != ts2
    assert np.all(ts2.time == ts1.time)
    assert ts2.data_info["signal1"]["Unit"] == "Unit1"
    assert ts2.data_info["signal2"]["Unit"] == "Unit2"
    assert ts2.data_info["signal3"]["Unit"] == "Unit3"
    assert ts2.events[0].time == 1.54
    assert ts2.events[1].time == 10.2
    assert ts2.events[2].time == 100

    # A deep copy without data_info
    ts2 = ts1.copy(copy_data_info=False)
    assert ts1 != ts2
    assert np.all(ts2.time == ts1.time)
    assert np.all(ts2.data["signal1"] == ts1.data["signal1"])
    assert np.all(ts2.data["signal2"] == ts1.data["signal2"])
    assert np.all(ts2.data["signal3"] == ts1.data["signal3"])
    assert ts2.events[0].time == 1.54
    assert ts2.events[1].time == 10.2
    assert ts2.events[2].time == 100


def test_check_well_typed():
    ts = ktk.TimeSeries()  # Should pass
    ts._check_well_typed()

    ts.time = [1, 2, 3]  # Should fail
    try:
        ts._check_well_typed()
        raise Exception("This should fail.")
    except TypeError:
        pass

    ts.time = np.array([1.0, 2.0, 3.0])  # Should pass
    ts._check_well_typed()

    ts.data["test1"] = np.array([1, 2, 3])
    ts.data["test2"] = [1, 2, 3]  # Should fail
    try:
        ts._check_well_typed()
        raise Exception("This should fail.")
    except TypeError:
        pass

    ts.data["test2"] = np.array([1, 2, 3])  # Should pass
    ts._check_well_typed()

    ts.time[1] = np.nan  # Should fail
    try:
        ts._check_well_typed()
        raise Exception("This should fail.")
    except TypeError:
        pass

    ts.time = np.array([1.0, 2.0, 2.0])  # Should fail
    try:
        ts._check_well_typed()
        raise Exception("This should fail.")
    except TypeError:
        pass

    ts.time = np.array([1.0, 2.0, 3.0])  # Should pass
    ts.add_data_info("test1", "Unit", "N", in_place=True)
    ts.add_data_info("test2", "Unit", "N", in_place=True)
    ts._check_well_typed()

    ts.data_info["test2"] = "string"  # Should fail
    try:
        ts._check_well_typed()
        raise Exception("This should fail.")
    except TypeError:
        pass

    ts.data_info = "string"  # Should fail
    try:
        ts._check_well_typed()
        raise Exception("This should fail.")
    except TypeError:
        pass

    ts.data_info = {}
    ts.time_info = "string"  # Should fail
    try:
        ts._check_well_typed()
        raise Exception("This should fail.")
    except TypeError:
        pass

    ts = ktk.TimeSeries()
    ts.add_event(0, in_place=True)  # Should pass
    ts._check_well_typed()

    ts.events = "test"  # Should fail
    try:
        ts._check_well_typed()
        raise Exception("This should fail.")
    except TypeError:
        pass

    ts.events = [ktk.TimeSeriesEvent(0), "test"]  # Should fail
    try:
        ts._check_well_typed()
        raise Exception("This should fail.")
    except TypeError:
        pass


def test_check_well_shaped():
    ts = ktk.TimeSeries()  # Should pass
    ts._check_well_shaped()

    ts.time = np.array([1.0, 2.0, 3.0])  # Should pass
    ts._check_well_shaped()

    ts.data["test1"] = np.array([1, 2, 3])
    ts.data["test2"] = np.array([1, 2, 3])  # Should pass
    ts._check_well_shaped()

    ts.data["test2"] = np.array([1, 2])  # Should fail
    try:
        ts._check_well_shaped()
        raise Exception("This should fail.")
    except ValueError:
        pass

    ts.data["test2"] = np.array([1, 2, 3])  # Should pass
    ts._check_well_shaped()


def test_check_not_empty_time():
    ts = ktk.TimeSeries()  # Should fail
    try:
        ts._check_not_empty_time()
        raise Exception("This should fail.")
    except ValueError:
        pass
    ts.time = np.array([1.0, 2.0, 3.0])  # Should pass
    ts._check_not_empty_time()


def test_check_increasing_time():
    ts = ktk.TimeSeries(time=np.array([0.0, 2.0, 1.0]))  # Should fail
    try:
        ts._check_increasing_time()
        raise Exception("This should fail.")
    except ValueError:
        pass
    ts.time = np.array([1.0, 2.0, 3.0])  # Should pass
    ts._check_increasing_time


def test_check_constant_sample_rate():
    ts = ktk.TimeSeries(time=np.array([0.0, 1.0, 3.0]))  # Should fail
    try:
        ts._check_constant_sample_rate()
        raise Exception("This should fail.")
    except ValueError:
        pass
    ts.time = np.array([1.0, 2.0, 3.0])  # Should pass
    ts._check_constant_sample_rate()


def test_check_not_empty_data():
    ts = ktk.TimeSeries()
    ts.time = np.array([1.0, 2.0, 3.0])  # Should fail
    try:
        ts._check_not_empty_data()
        raise Exception("This should fail.")
    except ValueError:
        pass
    ts.data["test1"] = np.array([1.0, 2.0, 3.0])  # Should pass
    ts._check_not_empty_data()


# %% From and to dataframe


def test_from_to_dataframe():
    # from_dataframe
    df = pd.DataFrame(
        columns=[
            "Data0",
            "Data1[0,0]",
            "Data1[0,1]",
            "Data1[1,0]",
            "Data1[1,1]",
        ]
    )
    df["Data0"] = np.arange(2)
    df["Data1[0,0]"] = np.arange(2) + 1
    df["Data1[0,1]"] = np.arange(2) + 2
    df["Data1[1,0]"] = np.arange(2) + 3
    df["Data1[1,1]"] = np.arange(2) + 4
    ts = ktk.TimeSeries.from_dataframe(df)
    assert np.allclose(ts.data["Data0"], [0, 1])
    assert np.allclose(ts.data["Data1"], [[[1, 2], [3, 4]], [[2, 3], [4, 5]]])

    # to_dataframe
    df2 = ts.to_dataframe()
    assert np.all(df2 == df)

    # Do the same with empty data
    df = pd.DataFrame(
        columns=[
            "Data0",
            "Data1[0,0]",
            "Data1[0,1]",
            "Data1[1,0]",
            "Data1[1,1]",
        ]
    )
    df["Data0"] = np.array([])
    df["Data1[0,0]"] = np.array([])
    df["Data1[0,1]"] = np.array([])
    df["Data1[1,0]"] = np.array([])
    df["Data1[1,1]"] = np.array([])

    ts = ktk.TimeSeries.from_dataframe(df)
    assert ts.data["Data0"].shape == (0,)
    assert ts.data["Data1"].shape == (0, 2, 2)

    # # This test should pass after solving issue #59
    # df2 = ts.to_dataframe()
    # assert np.all(df == df2)


# def test_bugfix_59():
#     """
#     Test bugfix #59.

#     TimeSeries.to_dataframe(): Bad columns in output DataFrame when data is
#     empty
#     """

#     df = pd.DataFrame(
#         columns=[
#             "Data0",
#             "Data1[0,0]",
#             "Data1[0,1]",
#             "Data1[1,0]",
#             "Data1[1,1]",
#         ]
#     )
#     df["Data0"] = np.array([])
#     df["Data1[0,0]"] = np.array([])
#     df["Data1[0,1]"] = np.array([])
#     df["Data1[1,0]"] = np.array([])
#     df["Data1[1,1]"] = np.array([])

#     ts = ktk.TimeSeries.from_dataframe(df)
#     assert ts.data["Data0"].shape == (0,)
#     assert ts.data["Data1"].shape == (0, 2, 2)

#     df2 = ts.to_dataframe()
#     assert np.all(df == df2)


# %% Metadata


def test_add_remove_data_info():
    ts = ktk.TimeSeries()
    ts = ts.add_data_info("Force", "Unit", "N")
    ts.add_data_info("Force", "Other", "hello", in_place=True)

    # Check that data_info was added
    assert ts.data_info["Force"]["Unit"] == "N"
    assert ts.data_info["Force"]["Other"] == "hello"

    # Test removing non-existing data_info (should not work)
    try:
        ts = ts.remove_data_info("Nonexisting", "Other")
        raise Exception("This should fail.")
    except KeyError:
        pass

    # Test removing existing data_info
    ts.remove_data_info("Force", "Other", in_place=True)
    assert ts.data_info["Force"]["Unit"] == "N"
    assert len(ts.data_info["Force"]) == 1


def test_rename_remove_data():
    ts = ktk.TimeSeries(time=np.arange(10))
    ts.data["Force"] = np.arange(10)
    ts = ts.rename_data("Force", "Moment")
    assert np.allclose(ts.data["Moment"], np.arange(10))

    # Same test but with data info
    ts.add_data_info("Moment", "Unit", "Nm", in_place=True)
    ts = ts.rename_data("Moment", "Power")
    assert np.allclose(ts.data["Power"], np.arange(10))
    assert ts.data_info["Power"]["Unit"] == "Nm"

    # Rename inexistent data (should fail)
    try:
        ts = ts.rename_data("NoKey", "Anything")
        raise Exception("This should fail.")
    except KeyError:
        pass

    # Remove inexistent data (should fail)
    try:
        ts.remove_data("NoKey", in_place=True)
        raise Exception("This should fail.")
    except KeyError:
        pass

    # Those fail should not have modified the TimeSeries
    assert np.allclose(ts.data["Power"], np.arange(10))
    assert len(ts.data) == 1
    assert len(ts.data_info) == 1

    # Remove data
    ts = ts.remove_data("Power")
    assert len(ts.data) == 0
    assert len(ts.data_info) == 0


# %% Events


def test_add_event():
    # Add event with unique=True and False
    ts = ktk.TimeSeries()
    ts = ts.add_event(5.5, "event1")
    ts.add_event(5.5, "event1", in_place=True, unique=True)
    ts.add_event(5.5, "event2", in_place=True, unique=True)
    assert len(ts.events) == 2
    ts.add_event(5.5, "event2", in_place=True)
    assert len(ts.events) == 3


def test_rename_event():
    # Original doctest
    ts = ktk.TimeSeries()
    ts = ts.add_event(5.5, "event1")
    ts.add_event(10.8, "event2", in_place=True)
    ts.add_event(2.3, "event2", in_place=True)

    assert str(ts.events) == (
        "[TimeSeriesEvent(time=5.5, name='event1'), "
        "TimeSeriesEvent(time=10.8, name='event2'), "
        "TimeSeriesEvent(time=2.3, name='event2')]"
    )

    ts = ts.rename_event("event2", "event3")

    assert str(ts.events) == (
        "[TimeSeriesEvent(time=5.5, name='event1'), "
        "TimeSeriesEvent(time=10.8, name='event3'), "
        "TimeSeriesEvent(time=2.3, name='event3')]"
    )

    ts.rename_event("event3", "event4", 0, in_place=True)

    assert str(ts.events) == (
        "[TimeSeriesEvent(time=5.5, name='event1'), "
        "TimeSeriesEvent(time=10.8, name='event3'), "
        "TimeSeriesEvent(time=2.3, name='event4')]"
    )

    # Test renaming an event to a same name (should pass)
    ts = ts.rename_event("event4", "event4")

    assert str(ts.events) == (
        "[TimeSeriesEvent(time=5.5, name='event1'), "
        "TimeSeriesEvent(time=10.8, name='event3'), "
        "TimeSeriesEvent(time=2.3, name='event4')]"
    )

    # Test renaming invalid occurrence (should fail)
    try:
        ts = ts.rename_event("event4", "event5", 10)
        raise Exception("This should fail.")
    except TimeSeriesEventNotFoundError:
        pass

    assert str(ts.events) == (
        "[TimeSeriesEvent(time=5.5, name='event1'), "
        "TimeSeriesEvent(time=10.8, name='event3'), "
        "TimeSeriesEvent(time=2.3, name='event4')]"
    )


def test_remove_event():
    # Original doctest
    ts = ktk.TimeSeries()
    ts = ts.add_event(5.5, "event1")
    ts = ts.add_event(10.8, "event2")
    ts = ts.add_event(2.3, "event2")
    assert str(ts.events) == (
        "[TimeSeriesEvent(time=5.5, name='event1'), "
        "TimeSeriesEvent(time=10.8, name='event2'), "
        "TimeSeriesEvent(time=2.3, name='event2')]"
    )

    ts = ts.remove_event("event1")
    assert str(ts.events) == (
        "[TimeSeriesEvent(time=10.8, name='event2'), "
        "TimeSeriesEvent(time=2.3, name='event2')]"
    )

    ts.remove_event("event2", 1, in_place=True)
    assert str(ts.events) == "[TimeSeriesEvent(time=2.3, name='event2')]"

    # Test remove bad occurrence (should fail)
    try:
        ts = ts.remove_event("event2", 10)
        raise Exception("This should fail.")
    except TimeSeriesEventNotFoundError:
        pass

    assert str(ts.events) == "[TimeSeriesEvent(time=2.3, name='event2')]"


def test_remove_duplicate_events():
    ts = ktk.TimeSeries()
    # Three occurrences of event1 (third after event3)
    ts = ts.add_event(0.0, "event1")
    ts = ts.add_event(1e-12, "event1")

    # One occurrence of event2, but also at 0.0 second
    ts = ts.add_event(0.0, "event2")

    # Two occurrences of event3
    ts = ts.add_event(2.0, "event3")
    ts = ts.add_event(2.0, "event3")
    ts = ts.add_event(0.0, "event1")

    assert str(ts._get_duplicate_event_indexes()) == "[1, 4, 5]"

    ts2 = ts.remove_duplicate_events()
    assert ts2.events[0].time == 0.0
    assert ts2.events[0].name == "event1"
    assert ts2.events[1].time == 0.0
    assert ts2.events[1].name == "event2"
    assert ts2.events[2].time == 2.0
    assert ts2.events[2].name == "event3"


def test_sort_events():
    # Original doctest
    ts = ktk.TimeSeries(time=np.arange(100) / 10)
    ts = ts.add_event(2, "two")
    ts = ts.add_event(1, "one")
    ts = ts.add_event(3, "three")
    ts = ts.add_event(3, "three")
    ts = ts.sort_events()
    assert str(ts.events) == (
        "[TimeSeriesEvent(time=1, name='one'), "
        "TimeSeriesEvent(time=2, name='two'), "
        "TimeSeriesEvent(time=3, name='three'), "
        "TimeSeriesEvent(time=3, name='three')]"
    )
    ts.sort_events(in_place=True, unique=True)
    assert str(ts.events) == (
        "[TimeSeriesEvent(time=1, name='one'), "
        "TimeSeriesEvent(time=2, name='two'), "
        "TimeSeriesEvent(time=3, name='three')]"
    )


def test_get_event_indexes_count_index_time():
    ts = ktk.TimeSeries()
    ts = ts.add_event(5.5, "event1")
    ts = ts.add_event(10.8, "event2")
    ts = ts.add_event(2.3, "event2")
    assert ts.count_events("event1") == 1
    assert ts.count_events("event2") == 2
    assert ts._get_event_indexes("event0") == []
    assert ts._get_event_indexes("event1") == [0]
    assert ts._get_event_indexes("event2") == [2, 1]
    assert ts._get_event_index("event2", 1) == 1
    try:
        ts._get_event_index("event0", 0)
        raise Exception("This should fail.")
    except TimeSeriesEventNotFoundError:
        pass
    try:
        ts._get_event_index("event1", 1)
        raise Exception("This should fail.")
    except TimeSeriesEventNotFoundError:
        pass

    # Deprecated
    assert ts.get_event_time("event1") == 5.5
    assert ts.get_event_time("event2", 0) == 2.3
    assert ts.get_event_time("event2", 1) == 10.8


# %% Sample rate, merge, resample


def test_get_sample_rate():
    ts = ktk.TimeSeries()
    assert np.isnan(ts.get_sample_rate())

    ts.time = np.array([0.0, 0.1, 0.2, 0.3])
    assert ts.get_sample_rate() == 10.0

    ts.time = np.array([0, 0.1, 0.3, 0.4])
    assert np.isnan(ts.get_sample_rate())

    ts.time = np.array([0])
    assert np.isnan(ts.get_sample_rate())

    ts.time = np.array([0.0, 0.2, 0.1])
    assert np.isnan(ts.get_sample_rate())

    ts.time = np.array([0.0, 0.1])
    assert ts.get_sample_rate() == 10.0


def test_merge_and_resample():
    # Begin with two timeseries with identical times
    ts1 = ktk.TimeSeries()
    ts1.time = np.linspace(0, 99, 100)
    ts1.data["signal1"] = np.random.rand(100, 2)
    ts1.data["signal2"] = np.random.rand(100, 2)
    ts1.data["signal3"] = np.random.rand(100, 2)
    ts1.add_data_info("signal1", "Unit", "Unit1", in_place=True)
    ts1.add_data_info("signal2", "Unit", "Unit2", in_place=True)
    ts1.add_data_info("signal3", "Unit", "Unit3", in_place=True)
    ts1.add_event(1.54, "test_event1", in_place=True)
    ts1.add_event(10.2, "test_event2", in_place=True)
    ts1.add_event(100, "test_event3", in_place=True)

    ts2 = ktk.TimeSeries()
    ts2.time = np.linspace(0, 99, 100)
    ts2.data["signal4"] = np.random.rand(100, 2)
    ts2.data["signal5"] = np.random.rand(100, 2)
    ts2.data["signal6"] = np.random.rand(100, 2)
    ts2.add_data_info("signal4", "Unit", "Unit1", in_place=True)
    ts2.add_data_info("signal5", "Unit", "Unit2", in_place=True)
    ts2.add_data_info("signal6", "Unit", "Unit3", in_place=True)
    ts2.add_event(1.54, "test_event1", in_place=True)
    ts2.add_event(10.2, "test_event2", in_place=True)
    ts2.add_event(100, "test_event4", in_place=True)  # This one is named diff.

    ts1 = ts1.merge(ts2)

    assert np.all(ts1.data["signal4"] == ts2.data["signal4"])
    assert np.all(ts1.data["signal5"] == ts2.data["signal5"])
    assert np.all(ts1.data["signal6"] == ts2.data["signal6"])
    assert np.all(
        ts1.data_info["signal4"]["Unit"] == ts2.data_info["signal4"]["Unit"]
    )
    assert np.all(
        ts1.data_info["signal5"]["Unit"] == ts2.data_info["signal5"]["Unit"]
    )
    assert np.all(
        ts1.data_info["signal6"]["Unit"] == ts2.data_info["signal6"]["Unit"]
    )

    assert len(ts1.events) == 4

    # Try with two timeseries that don't fit in time. It must generate an
    # exception.
    ts1 = ktk.TimeSeries()
    ts1.time = np.linspace(0, 99, 100, endpoint=False)
    ts1.data["signal1"] = np.random.rand(100, 2)
    ts1.data["signal2"] = np.random.rand(100, 2)
    ts1.data["signal3"] = np.random.rand(100, 2)
    ts1.add_data_info("signal1", "Unit", "Unit1", in_place=True)
    ts1.add_data_info("signal2", "Unit", "Unit2", in_place=True)
    ts1.add_data_info("signal3", "Unit", "Unit3", in_place=True)
    ts1.add_event(1.54, "test_event1", in_place=True)
    ts1.add_event(10.2, "test_event2", in_place=True)
    ts1.add_event(100, "test_event3", in_place=True)

    ts2 = ktk.TimeSeries()
    ts2.time = np.linspace(0, 99, 300, endpoint=False)
    ts2.data["signal4"] = ts2.time**2
    ts2.data["signal5"] = np.random.rand(300, 2)
    ts2.data["signal6"] = np.random.rand(300, 2)
    ts2.add_data_info("signal4", "Unit", "Unit1", in_place=True)
    ts2.add_data_info("signal5", "Unit", "Unit2", in_place=True)
    ts2.add_data_info("signal6", "Unit", "Unit3", in_place=True)
    ts2.add_event(1.54, "test_event4", in_place=True)
    ts2.add_event(10.2, "test_event5", in_place=True)
    ts2.add_event(100, "test_event6", in_place=True)

    try:
        ts1.merge(ts2)
        raise Exception("This command should have raised a ValueError.")
    except ValueError:
        pass

    # Try the same thing but with linear resampling
    ts1 = ts1.merge(ts2, resample=True)

    def _assert_almost_equal(one, two):
        assert np.max(np.abs(one - two)) < 1e-6

    _assert_almost_equal(ts1.data["signal4"], ts2.data["signal4"][0::3])
    _assert_almost_equal(ts1.data["signal5"], ts2.data["signal5"][0::3])
    _assert_almost_equal(ts1.data["signal6"], ts2.data["signal6"][0::3])
    assert ts1.data_info["signal4"]["Unit"] == ts2.data_info["signal4"]["Unit"]
    assert ts1.data_info["signal5"]["Unit"] == ts2.data_info["signal5"]["Unit"]
    assert ts1.data_info["signal6"]["Unit"] == ts2.data_info["signal6"]["Unit"]


def test_resample_using_frequency():
    """Test resample using a frequency instead of a new time."""
    ts1 = ktk.TimeSeries()
    ts1.time = np.linspace(0, 100, 10000, endpoint=False)  # Sampled at 100Hz
    ts1.data["signal1"] = np.sin(ts1.time)

    # Resample at 50Hz
    ts2 = ts1.resample(50)
    assert np.allclose(ts1.time[0::2], ts2.time)

    # Resample at 25Hz
    ts2 = ts1.resample(25.0)
    assert np.allclose(ts1.time[0::4], ts2.time)

    # Resample at an uneven number
    ts2 = ts1.resample(10.468)
    assert np.abs(ts2.time[-1] - ts1.time[-1]) < 1 / 10.468


def test_resample_with_nans():
    ts = ktk.TimeSeries(time=np.arange(10.0))
    ts.data["data"] = ts.time**2
    ts.data["data"][[0, 1, 5, 8, 9]] = np.nan
    ts1 = ts.resample(2.0)
    assert np.allclose(
        np.isnan(ts1.data["data"]),
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
    )


def test_fill_missing_samples():
    ts = ktk.TimeSeries(time=np.arange(10))
    ts.data["data"] = np.sin(ts.time / 5)

    # Test nothing to do
    ts2 = ts.fill_missing_samples(0)
    assert ts2._is_equivalent(ts)

    # Test fill everything
    ts2 = ts.copy()
    ts2.data["data"][3:6] = np.nan
    ts2.data["data"][-1] = np.nan
    ts3 = ts2.fill_missing_samples(0)
    assert np.all(
        np.isclose(ts2.data["data"], ts3.data["data"])
        == [True, True, True, False, False, False, True, True, True, False]
    )
    assert np.all(
        ts3.isnan("data")
        == [
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]
    )

    # Test fill only small holes
    ts3 = ts2.fill_missing_samples(1)
    assert np.all(
        np.isclose(ts2.data["data"], ts3.data["data"])
        == [True, True, True, False, False, False, True, True, True, False]
    )

    assert np.all(
        ts3.isnan("data")
        == [False, False, False, True, True, True, False, False, False, False]
    )

    # Test fill up to exactly the largest hole
    ts3 = ts2.fill_missing_samples(3)
    assert np.all(
        np.isclose(ts2.data["data"], ts3.data["data"])
        == [True, True, True, False, False, False, True, True, True, False]
    )
    assert np.all(
        ts3.isnan("data")
        == [
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]
    )

    # Test fill with the first one missing
    ts2.data["data"][0:2] = np.nan
    ts3 = ts2.fill_missing_samples(1)
    assert np.all(
        np.isclose(ts2.data["data"], ts3.data["data"])
        == [False, False, True, False, False, False, True, True, True, False]
    )
    assert np.all(
        ts3.isnan("data")
        == [
            False,
            False,
            False,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
        ]
    )


# %% get_index


def test_get_index_at_time():
    ts = ktk.TimeSeries(time=np.array([0.2, 0.5, 1, 1.5, 2]))
    assert ts.get_index_at_time(0) == 0
    assert ts.get_index_at_time(0.2) == 0
    assert ts.get_index_at_time(0.9) == 2
    assert ts.get_index_at_time(1) == 2
    assert ts.get_index_at_time(1.1) == 2
    assert ts.get_index_at_time(2.1) == 4


def test_get_index_before_time():
    ts = ktk.TimeSeries(time=np.array([0.2, 0.5, 1, 1.5, 2]))
    try:
        ts.get_index_before_time(0)
        raise AssertionError("This should fail.")
    except TimeSeriesRangeError:
        pass
    assert ts.get_index_before_time(0.2, inclusive=True) == 0
    assert ts.get_index_before_time(0.9) == 1
    assert ts.get_index_before_time(1) == 1
    assert ts.get_index_before_time(1.1) == 2
    assert ts.get_index_before_time(1.0, inclusive=True) == 2
    assert ts.get_index_before_time(10) == 4


def test_get_index_after_time():
    # doctests
    ts = ktk.TimeSeries(time=np.array([0.2, 0.5, 1, 1.5, 2]))
    assert ts.get_index_after_time(-10) == 0
    assert ts.get_index_after_time(0.9) == 2
    assert ts.get_index_after_time(1) == 3
    assert ts.get_index_after_time(1, inclusive=True) == 2
    try:
        ts.get_index_after_time(2)
        raise AssertionError("This should fail.")
    except TimeSeriesRangeError:
        pass
    assert ts.get_index_after_time(2, inclusive=True) == 4


def test_get_index_at_event():
    ts = ktk.TimeSeries(time=np.arange(10) / 10)
    ts.add_event(0.2, "event", in_place=True)
    ts.add_event(0.36, "event", in_place=True)
    assert ts.get_index_at_event("event") == 2
    assert ts.get_index_at_event("event", occurrence=1) == 4


def test_get_index_before_event():
    ts = ktk.TimeSeries(time=np.arange(10) / 10)
    ts.add_event(0.2, "event", in_place=True)
    ts.add_event(0.36, "event", in_place=True)
    assert ts.get_index_before_event("event") == 1
    assert (
        ts.get_index_before_event("event", occurrence=0, inclusive=True) == 2
    )
    assert ts.get_index_before_event("event", occurrence=1) == 3
    assert (
        ts.get_index_before_event("event", occurrence=1, inclusive=True) == 4
    )
    try:
        ts.get_index_before_event("event", occurrence=2)
        raise AssertionError("This should fail.")
    except TimeSeriesEventNotFoundError:
        pass


def test_get_index_after_event():
    ts = ktk.TimeSeries(time=np.arange(10) / 10)
    ts.add_event(0.2, "event", in_place=True)
    ts.add_event(0.36, "event", in_place=True)
    assert ts.get_index_after_event("event") == 3
    assert ts.get_index_after_event("event", inclusive=True) == 2
    assert ts.get_index_after_event("event", occurrence=1) == 4
    assert ts.get_index_after_event("event", occurrence=1, inclusive=True) == 3
    try:
        ts.get_index_before_event("event", occurrence=2)
        raise AssertionError("This should fail.")
    except TimeSeriesEventNotFoundError:
        pass


# %% get_ts using index(es)
def test_get_ts_before_index():
    ts = ktk.TimeSeries(time=np.arange(10) / 10)
    try:
        ts.get_ts_before_index(0)
        raise AssertionError("This should raise a TimeSeriesRangeError.")
    except TimeSeriesRangeError:
        pass

    assert np.array_equal(
        ts.get_ts_before_index(0, inclusive=True).time, np.array([0.0])
    )
    assert np.array_equal(ts.get_ts_before_index(1).time, np.array([0.0]))
    assert np.array_equal(
        ts.get_ts_before_index(1, inclusive=True).time, np.array([0.0, 0.1])
    )
    assert np.array_equal(
        ts.get_ts_before_index(9, inclusive=True).time, ts.time
    )


def test_get_ts_after_index():
    ts = ktk.TimeSeries(time=np.arange(10) / 10)
    try:
        ts.get_ts_after_index(9)
        raise AssertionError("This should raise a TimeSeriesRangeError.")
    except TimeSeriesRangeError:
        pass

    assert np.array_equal(
        ts.get_ts_after_index(9, inclusive=True).time, np.array([0.9])
    )
    assert np.array_equal(ts.get_ts_after_index(8).time, np.array([0.9]))
    assert np.array_equal(
        ts.get_ts_after_index(8, inclusive=True).time, np.array([0.8, 0.9])
    )
    assert np.array_equal(
        ts.get_ts_after_index(0, inclusive=True).time, ts.time
    )


def test_get_ts_between_indexes():
    ts = ktk.TimeSeries(time=np.arange(10) / 10)
    assert np.array_equal(
        ts.get_ts_between_indexes(2, 5).time, np.array([0.3, 0.4])
    )
    assert np.array_equal(
        ts.get_ts_between_indexes(2, 5, inclusive=True).time,
        np.array([0.2, 0.3, 0.4, 0.5]),
    )
    assert np.array_equal(
        ts.get_ts_between_indexes(2, 5, inclusive=[True, False]).time,
        np.array([0.2, 0.3, 0.4]),
    )
    assert np.array_equal(
        ts.get_ts_between_indexes(2, 5, inclusive=[False, True]).time,
        np.array([0.3, 0.4, 0.5]),
    )
    try:
        ts.get_ts_between_indexes(5, 2)  # Bad order
        raise AssertionError("This should fail.")
    except ValueError:
        pass


# %% get_ts using time(s)


def test_get_ts_before_time():
    ts = ktk.TimeSeries(
        time=np.linspace(0, 9, 10),
        data={"data": np.linspace(0, 0.9, 10)},
    )

    try:
        ts.get_ts_before_time(0)
        raise AssertionError("This should fail.")
    except TimeSeriesRangeError:
        pass

    newts = ts.get_ts_before_time(0, inclusive=True)
    assert np.allclose(newts.time, [0.0])
    assert np.allclose(newts.data["data"], [0.0])

    newts = ts.get_ts_before_time(5)
    assert np.allclose(newts.time, [0.0, 1.0, 2.0, 3.0, 4.0])
    assert np.allclose(newts.data["data"], [0.0, 0.1, 0.2, 0.3, 0.4])

    newts = ts.get_ts_before_time(np.inf)
    assert newts == ts


def test_get_ts_after_time():
    ts = ktk.TimeSeries(
        time=np.linspace(0, 9, 10),
        data={"data": np.linspace(0, 0.9, 10)},
    )

    try:
        ts.get_ts_after_time(9)
        raise AssertionError("This should fail.")
    except TimeSeriesRangeError:
        pass

    newts = ts.get_ts_after_time(9, inclusive=True)
    assert np.allclose(newts.time, [9.0])
    assert np.allclose(newts.data["data"], [0.9])

    newts = ts.get_ts_after_time(5)
    assert np.allclose(newts.time, [6.0, 7.0, 8.0, 9.0])
    assert np.allclose(newts.data["data"], [0.6, 0.7, 0.8, 0.9])

    newts = ts.get_ts_after_time(-np.Inf)
    assert newts == ts


def test_get_ts_between_times():
    ts = ktk.TimeSeries(
        time=np.linspace(0, 9, 10),
        data={"data": np.linspace(0, 0.9, 10)},
    )

    new_ts = ts.get_ts_between_times(-2, 13)
    assert new_ts == ts

    # Test that inclusive works
    new_ts = ts.get_ts_between_times(4.5, 8.0, inclusive=True)
    assert np.allclose(new_ts.time, [5.0, 6.0, 7.0, 8.0])
    assert np.allclose(new_ts.data["data"], [0.5, 0.6, 0.7, 0.8])

    # Test that inclusive works (2)
    new_ts = ts.get_ts_between_times(4.0, 8.0, inclusive=True)
    assert np.allclose(new_ts.time, [4.0, 5.0, 6.0, 7.0, 8.0])
    assert np.allclose(new_ts.data["data"], [0.4, 0.5, 0.6, 0.7, 0.8])

    # Test that double inclusive works
    new_ts = ts.get_ts_between_times(4.0, 8.0, inclusive=[True, False])
    assert np.allclose(new_ts.time, [4.0, 5.0, 6.0, 7.0])
    assert np.allclose(new_ts.data["data"], [0.4, 0.5, 0.6, 0.7])

    # Test that double inclusive works (2)
    new_ts = ts.get_ts_between_times(4.0, 8.0, inclusive=[False, True])
    assert np.allclose(new_ts.time, [5.0, 6.0, 7.0, 8.0])
    assert np.allclose(new_ts.data["data"], [0.5, 0.6, 0.7, 0.8])

    # Check that interverting times fails
    try:
        ts.get_ts_between_times(7.5, 4.5)
        raise Exception("This should fail.")
    except ValueError:
        pass
    # Check that data outside span fails
    try:
        ts.get_ts_between_times(-2, -1)
        raise Exception("This should fail.")
    except TimeSeriesRangeError:
        pass


# %% get_ts using event(s)


def test_get_ts_before_event():
    ts = ktk.TimeSeries(
        time=np.linspace(0, 9, 10),
        data={"data": np.linspace(0, 0.9, 10)},
    )
    ts.add_event(-50, "event", in_place=True)
    ts.add_event(0, "event", in_place=True)
    ts.add_event(0.2, "event", in_place=True)
    ts.add_event(50, "event", in_place=True)

    try:
        ts.get_ts_before_event("event", 0)
    except TimeSeriesRangeError:
        pass

    try:
        ts.get_ts_before_event("event", 1)
    except TimeSeriesRangeError:
        pass

    new_ts = ts.get_ts_before_event("event", 1, inclusive=True)
    assert np.allclose(new_ts.time, [0.0])
    assert np.allclose(new_ts.data["data"], [0.0])

    new_ts = ts.get_ts_before_event("event", 2)
    assert np.allclose(new_ts.time, [0.0])
    assert np.allclose(new_ts.data["data"], [0.0])

    new_ts = ts.get_ts_before_event("event", 2, inclusive=True)
    assert np.allclose(new_ts.time, [0.0, 1.0])
    assert np.allclose(new_ts.data["data"], [0.0, 0.1])

    new_ts = ts.get_ts_before_event("event", 3)
    assert new_ts == ts


def test_get_ts_after_event():
    ts = ktk.TimeSeries(
        time=np.linspace(0, 9, 10),
        data={"data": np.linspace(0, 0.9, 10)},
    )
    ts.add_event(-50, "event", in_place=True)
    ts.add_event(9.0 - 0.2, "event", in_place=True)  # 8.8 with float pt error
    ts.add_event(9.0, "event", in_place=True)
    ts.add_event(50, "event", in_place=True)

    try:
        ts.get_ts_after_event("event", 3)
    except TimeSeriesRangeError:
        pass

    try:
        ts.get_ts_after_event("event", 2)
    except TimeSeriesRangeError:
        pass

    new_ts = ts.get_ts_after_event("event", 2, inclusive=True)
    assert np.allclose(new_ts.time, [9.0])
    assert np.allclose(new_ts.data["data"], [0.9])

    new_ts = ts.get_ts_after_event("event", 1)
    assert np.allclose(new_ts.time, [9.0])
    assert np.allclose(new_ts.data["data"], [0.9])

    new_ts = ts.get_ts_after_event("event", 1, inclusive=True)
    assert np.allclose(new_ts.time, [8.0, 9.0])
    assert np.allclose(new_ts.data["data"], [0.8, 0.9])

    new_ts = ts.get_ts_after_event("event", 0)
    assert new_ts == ts


def test_get_ts_between_events():
    ts = ktk.TimeSeries(
        time=np.linspace(0, 9, 10),
        data={"data": np.linspace(0, 0.9, 10)},
    )

    ts.add_event(-2, "event1", in_place=True)
    ts.add_event(-1, "event1", in_place=True)
    ts.add_event(3, "event1", in_place=True)
    ts.add_event(4.5, "event2", in_place=True)
    ts.add_event(5, "event2", in_place=True)
    ts.add_event(7.5, "event2", in_place=True)
    ts.add_event(13, "event2", in_place=True)

    new_ts = ts.get_ts_between_events("event1", "event2", 0, 3)
    assert new_ts == ts

    # Test that inclusive works
    new_ts = ts.get_ts_between_events("event2", "event2", 0, 2, inclusive=True)
    assert np.allclose(new_ts.time, [4.0, 5.0, 6.0, 7.0, 8.0])
    assert np.allclose(new_ts.data["data"], [0.4, 0.5, 0.6, 0.7, 0.8])

    # Test that multiple inclusive work
    new_ts = ts.get_ts_between_events(
        "event2", "event2", 0, 2, inclusive=[True, False]
    )
    assert np.allclose(new_ts.time, [4.0, 5.0, 6.0, 7.0])
    assert np.allclose(new_ts.data["data"], [0.4, 0.5, 0.6, 0.7])

    # Test that multiple inclusive work (2)
    new_ts = ts.get_ts_between_events(
        "event2", "event2", 0, 2, inclusive=[False, True]
    )
    assert np.allclose(new_ts.time, [5.0, 6.0, 7.0, 8.0])
    assert np.allclose(new_ts.data["data"], [0.5, 0.6, 0.7, 0.8])

    # Check that interverting times fails
    try:
        ts.get_ts_between_events("event2", "event1", 0, 1)
        raise Exception("This should fail")
    except ValueError:
        pass
    # Check that data outside span fails
    try:
        ts.get_ts_between_events("event1", "event1", 0, 1)
        raise Exception("This should fail")
    except TimeSeriesRangeError:
        pass


# %% plot


def test_plot():
    """Test that many parameter combinations doesn't crash."""
    ts = ktk.TimeSeries(time=np.arange(100))

    # Test a plot with no data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts.plot()

    # Test a plot with empty data
    ts.data["data1"] = np.array([])

    # Add some data
    ts.data["data1"] = ts.time.copy()
    fig = plt.figure()
    ts.plot()
    plt.close(fig)

    # Add another data
    ts.data["data2"] = np.hstack(
        [ts.time[:, np.newaxis], ts.time[:, np.newaxis]]
    )
    fig = plt.figure()
    ts.plot()
    plt.close(fig)

    # Add units
    ts.add_data_info("data1", "Unit", "m", in_place=True)
    fig = plt.figure()
    ts.plot()
    plt.close(fig)

    # Add another unit
    ts.add_data_info("data2", "Unit", "mm", in_place=True)
    fig = plt.figure()
    ts.plot()
    plt.close(fig)

    # Plot only one signal
    fig = plt.figure()
    ts.plot(["data1"])
    plt.close(fig)

    # Add events
    ts.add_event(0, "event", in_place=True)
    ts.add_event(2, "_", in_place=True)
    ts.add_event(3, in_place=True)
    fig = plt.figure()
    ts.plot()
    plt.close(fig)

    # Plot without event_names and iwhtout legend
    fig = plt.figure()
    ts.plot(["data1"], event_names=False, legend=False)
    plt.close(fig)


# %% Deprecated
def test_get_ts_at_event___get_ts_at_time():
    ts = ktk.TimeSeries()
    ts.time = np.linspace(0, 99, 100)
    time_as_column = np.reshape(ts.time, (-1, 1))
    ts.data["Forces"] = np.block(
        [time_as_column, time_as_column**2, time_as_column**3]
    )
    ts.data["Moments"] = np.block(
        [time_as_column**2, time_as_column**3, time_as_column**4]
    )
    ts = ts.add_event(5.5, "event1")
    ts = ts.add_event(10.8, "event2")
    ts = ts.add_event(2.3, "event2")
    new_ts = ts.get_ts_at_event("event1")
    assert np.array_equal(new_ts.time, np.array([5]))
    new_ts = ts.get_ts_at_event("event2")
    assert np.array_equal(new_ts.time, np.array([2]))
    new_ts = ts.get_ts_at_event("event2", 1)
    assert np.array_equal(new_ts.time, np.array([11]))


# %% Main
if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
