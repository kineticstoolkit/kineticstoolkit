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
Unit tests for the cycles module.
"""
import kineticstoolkit as ktk
import numpy as np
import warnings


def test_detect_cycles():
    # Test min and max lenghts
    # Create a timeseries with one frame per seconds.
    t = np.arange(40)
    # fmt: off
    d = np.array(
        [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=float
    )
    # fmt: on
    ts = ktk.TimeSeries(time=t, data={"data": d})

    # Detect all cycles
    ts2 = ktk.cycles.detect_cycles(
        ts, "data", event_names=["start", "stop"], thresholds=[0.5, 0.5]
    )
    assert ts2.events[0].name == "start"
    assert ts2.events[0].time == 2
    assert ts2.events[1].name == "stop"
    assert ts2.events[1].time == 6
    assert ts2.events[3].name == "start"
    assert ts2.events[3].time == 10
    assert ts2.events[4].name == "stop"
    assert ts2.events[4].time == 13
    assert ts2.events[6].name == "start"
    assert ts2.events[6].time == 18
    assert ts2.events[7].name == "stop"
    assert ts2.events[7].time == 19
    assert ts2.events[9].name == "start"
    assert ts2.events[9].time == 22
    assert ts2.events[10].name == "stop"
    assert ts2.events[10].time == 31
    assert ts2.events[11].name == "_"
    assert ts2.events[11].time == 39

    # With falling direction
    ts2 = ktk.cycles.detect_cycles(
        ts,
        "data",
        event_names=["start", "stop"],
        thresholds=[0.5, 0.5],
        directions=["falling"],
    )
    assert ts2.events[0].name == "start"
    assert ts2.events[0].time == 6
    assert ts2.events[1].name == "stop"
    assert ts2.events[1].time == 10
    assert ts2.events[2].name == "_"
    assert ts2.events[2].time == 13
    assert ts2.events[3].name == "start"
    assert ts2.events[3].time == 13
    assert ts2.events[4].name == "stop"
    assert ts2.events[4].time == 18
    assert ts2.events[5].name == "_"
    assert ts2.events[5].time == 19
    assert ts2.events[6].name == "start"
    assert ts2.events[6].time == 19
    assert ts2.events[7].name == "stop"
    assert ts2.events[7].time == 22
    assert ts2.events[8].name == "_"
    assert ts2.events[8].time == 31

    # With minimal cycles
    ts3 = ktk.cycles.detect_cycles(
        ts,
        "data",
        event_names=["start", "stop"],
        thresholds=[0.5, 0.5],
        min_durations=[2, 4],
    )
    assert ts3.events[0].name == "start"
    assert ts3.events[0].time == 2
    assert ts3.events[1].name == "stop"
    assert ts3.events[1].time == 6
    assert ts3.events[2].name == "_"
    assert ts3.events[2].time == 10
    assert ts3.events[3].name == "start"
    assert ts3.events[3].time == 10
    assert ts3.events[4].name == "stop"
    assert ts3.events[4].time == 13
    assert ts3.events[5].name == "_"
    assert ts3.events[5].time == 18
    assert ts3.events[6].name == "start"
    assert ts3.events[6].time == 22
    assert ts3.events[7].name == "stop"
    assert ts3.events[7].time == 31
    assert ts3.events[8].name == "_"
    assert ts3.events[8].time == 39

    # With both minimal and maximal cycles
    ts4 = ktk.cycles.detect_cycles(
        ts,
        "data",
        event_names=["start", "stop"],
        thresholds=[0.5, 0.5],
        min_durations=[4, 3],
        max_durations=[8, 7],
    )
    assert ts4.events[0].name == "start"
    assert ts4.events[0].time == 2
    assert ts4.events[1].name == "stop"
    assert ts4.events[1].time == 6
    assert ts4.events[2].name == "_"
    assert ts4.events[2].time == 10
    assert len(ts4.events) == 3

    # With target heights
    t = np.arange(40)
    # fmt: off
    d = np.array(
        [0, 0, 1, 1, 2, 1, 0, 0, -1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        dtype=float,
    )
    # fmt: on
    ts = ktk.TimeSeries(time=t, data={"data": d})

    ts5 = ktk.cycles.detect_cycles(
        ts,
        "data",
        event_names=["start", "stop"],
        thresholds=[0.5, 0.5],
        min_peak_heights=[2, -np.Inf],
        max_peak_heights=[np.Inf, -1],
    )
    assert ts5.events[0].name == "start"
    assert ts5.events[0].time == 2
    assert ts5.events[1].name == "stop"
    assert ts5.events[1].time == 6
    assert ts5.events[2].name == "_"
    assert ts5.events[2].time == 10
    assert len(ts5.events) == 3


def test_time_normalize():
    # Create a TimeSeries with some events directly synced with the data and
    # some other that aren't.
    ts = ktk.TimeSeries()
    ts.time = np.linspace(0, 20, 201)  # 0 to 20 seconds by 0.1s
    ts.data["test"] = ts.time**2
    ts = ts.add_event(1.0, "push")
    ts = ts.add_event(2.0, "recovery")
    ts = ts.add_event(3.0, "push")
    ts = ts.add_event(3.95, "recovery")
    ts = ts.add_event(5.0, "push")
    ts = ts.add_event(6.05, "recovery")
    ts = ts.add_event(7.05, "push")
    ts = ts.add_event(8.0, "recovery")
    ts = ts.add_event(8.95, "push")
    ts = ts.add_event(10.05, "recovery")

    ts1 = ktk.cycles.time_normalize(ts, "push", "recovery")
    assert len(ts1.events) == 10  # We got all events

    # Test that if we re-time-normalize, we obtain the same TimeSeries
    ts2 = ktk.cycles.time_normalize(ts1, "push", "_")

    # Samething but with push to next push
    ts3 = ktk.cycles.time_normalize(ts, "push", "push")
    assert len(ts3.events) == 12  # No missing events
    assert ts3.events[0].name == "push"
    assert np.allclose(ts3.events[0].time, 0)
    assert ts3.events[1].name == "recovery"
    assert np.allclose(ts3.events[1].time, 50)
    assert ts3.events[2].name == "_"
    assert np.allclose(ts3.events[2].time, 100)
    assert ts3.events[3].name == "push"
    assert np.allclose(ts3.events[3].time, 100)

    # Test that if we re-time-normalize, we obtain the same TimeSeries
    ts4 = ktk.cycles.time_normalize(ts3, "push", "_")
    assert ts3 == ts4

    # There should be no nan in ts2
    assert ~ts2.isnan("test").all()

    # Test with a wider range
    ts5 = ktk.cycles.time_normalize(
        ts, "push", "push", n_points=100, span=[-25, 125]
    )
    # plt.subplot(2, 1, 1)
    # ts5.plot([], '.r')
    # plt.grid(True)
    # plt.subplot(2, 1, 2)
    # ts3.plot([], '.b')

    # We use a quite high rtol because some differences may be seen in the
    # bounds, due to the choice of resampling. Since the event times do not
    # necessarily correspond to the sample times, then the boundaries of ts3
    # may be calculated based on extrapolation, while the boundaries of ts5
    # are not since the time span is larger.
    assert np.allclose(
        ts5.data["test"][25:125], ts3.data["test"][0:100], rtol=0.03
    )
    assert np.allclose(
        ts5.data["test"][175:275], ts3.data["test"][100:200], rtol=0.03
    )
    assert np.allclose(
        ts5.data["test"][325:425], ts3.data["test"][200:300], rtol=0.03
    )
    assert np.allclose(
        ts5.data["test"][475:575], ts3.data["test"][300:400], rtol=0.03
    )

    assert ts5.events[0].name == "push"
    assert np.allclose(ts5.events[0].time, 25)
    assert ts5.events[1].name == "recovery"
    assert np.allclose(ts5.events[1].time, 0.5 * 100 + 25)
    assert ts5.events[2].name == "_"
    assert np.allclose(ts5.events[2].time, 100 + 25)

    assert ts5.events[3].name == "push"
    assert np.allclose(ts5.events[3].time, 175)
    assert ts5.events[4].name == "recovery"
    assert np.allclose(ts5.events[4].time, (0.95 / 2) * 100 + 175)
    assert ts5.events[5].name == "_"
    assert np.allclose(ts5.events[5].time, 100 + 175)

    assert ts5.events[6].name == "push"
    assert np.allclose(ts5.events[6].time, 325)
    assert ts5.events[7].name == "recovery"
    assert np.allclose(ts5.events[7].time, (1.05 / 2.05) * 100 + 325)
    assert ts5.events[8].name == "_"
    assert np.allclose(ts5.events[8].time, 100 + 325)

    # Test time_normalize with nonvalid event names - ValueError
    # (issue #156)
    try:
        ktk.cycles.time_normalize(ts, "begin", "end")
        raise ValueError("Should raise a ValueError.")
    except ValueError:
        pass


# def test_normalize_extended():
#     """
#     Test normalize with extended_span.
#     Commented because a more exhaustive test is done in test_timenormalize
#     ts = ktk.TimeSeries(time=np.arange(10))
#     ts.data['test'] = np.arange(10) ** 2
#     ts = ts.add_event(2, 'push')
#     ts = ts.add_event(4, 'recovery')
#     ts = ts.add_event(6, 'push')

#     ts1 = ktk.cycles.time_normalize(ts, 'push', '_', n_points=10)
#     ts2 = ktk.cycles.time_normalize(ts, 'push', '_', n_points=10,
#                                     span=[0, 11])
#     ts3 = ktk.cycles.time_normalize(ts, 'push', '_', n_points=10,
#                                     span=[-1, 10])

#     plt.subplot(3,1,1)
#     ts3.plot([], '.r')
#     plt.grid(True)
#     plt.subplot(3,1,2)
#     ts1.plot([], '.b')
#     plt.grid(True)
#     plt.subplot(3,1,3)
#     ts2.plot([], '.g')
#     plt.grid(True)


def test_most_repeatable_cycles():
    # Create a TimeSeries with 5 cycles, one of those is different from
    # the others
    data = np.array(
        [
            np.sin(np.arange(0, 10, 0.1)) + 0.00,  # 0 - most diff. 3rd removed
            np.sin(np.arange(0, 10, 0.1))
            + 0.10,  # 1 - 1st of remain. 5th removed
            np.cos(np.arange(0, 10, 0.1)) + 0.15,  # 2 - cos. 2nd removed
            np.sin(np.arange(0, 10, 0.1)) + 0.15,  # 3 - with nans. 1st removed
            np.sin(np.arange(0, 10, 0.1)) + 0.12,
        ]
    )  # 4 - 2nd of remn. 4th removed
    # Put some nans in the fourth cycle (which should be discarted entirely)
    data[3, 30] = np.nan

    test = ktk.cycles.most_repeatable_cycles(data)

    assert test == [1, 4, 0, 2]


def test_stack_unstack():
    # Create a periodic TimeSeries, time-normalize and stack
    ts = ktk.TimeSeries()
    ts.time = np.arange(1000)
    ts.data["sin"] = np.sin(ts.time / 100 * 2 * np.pi)
    ts.data["cos"] = np.cos(ts.time / 100 * 2 * np.pi)

    data = ktk.cycles.stack(ts)

    for i_cycle in range(10):
        assert np.all(
            np.abs(
                data["sin"][i_cycle] - np.sin(np.arange(100) / 100 * 2 * np.pi)
            )
            < 1e-10
        )

        assert np.all(
            np.abs(
                data["cos"][i_cycle] - np.cos(np.arange(100) / 100 * 2 * np.pi)
            )
            < 1e-10
        )

    # Test unstack
    ts2 = ktk.cycles.unstack(data)
    assert np.all(np.abs(ts2.data["sin"] - ts.data["sin"]) < 1e-10)
    assert np.all(np.abs(ts2.data["cos"] - ts.data["cos"]) < 1e-10)
    assert np.all(np.abs(ts2.time - ts.time) < 1e-10)


# Commented since stack_events is also commented for now.
#
# def test_stack_events():
#     # Create a TimeSeries with different time-normalized events
#     ts = ktk.TimeSeries(time=np.arange(400))  # 4 cycles of 100%
#     ts = ts.add_event(9, 'event1')    # event1 at 9% of cycle 0
#     ts = ts.add_event(110, 'event1')  # event1 at 10% of cycle 1
#     ts = ts.add_event(312, 'event1')  # event1 at 12% of cycle 3
#     ts = ts.add_event(382, 'event1')  # 2nd occurr. event1 at 82% of cycle 3
#     ts = ts.add_event(1, 'event2')  # event2 at 1% of cycle 0
#     ts = ts.add_event(5, 'event2')  # event2 at 5% of cycle 0

#     # Stack these events
#     events = ktk.cycles.stack_events(ts)
#     assert events['event1'] == [[9.0], [10.0], [], [12.0, 82.0]]
#     assert events['event2'] == [[1.0, 5.0], [], [], []]

if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
