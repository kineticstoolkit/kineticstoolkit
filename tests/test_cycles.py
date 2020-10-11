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
import ktk
import numpy as np


def test_detect_cycles():
    # Test min and max lenghts
    # Create a timeseries with one frame per seconds.
    t = np.arange(40)
    d = np.array([0, 0,
                 1, 1, 1, 1,
                 0, 0, 0, 0,
                 1, 1, 1,
                 0, 0, 0, 0, 0,
                 1,
                 0, 0, 0,
                 1, 1, 1, 1, 1, 1, 1, 1, 1,
                 0, 0, 0, 0, 0, 0, 0, 0,
                 1])
    ts = ktk.TimeSeries(time=t, data={'data': d})

    # Detect all cycles
    ts2 = ktk.cycles.detect_cycles(ts, 'data', 'start', 'stop',
                                   0.5, 0.5)
    assert ts2.events[0].name == 'start'
    assert ts2.events[0].time == 2
    assert ts2.events[1].name == 'stop'
    assert ts2.events[1].time == 6
    assert ts2.events[3].name == 'start'
    assert ts2.events[3].time == 10
    assert ts2.events[4].name == 'stop'
    assert ts2.events[4].time == 13
    assert ts2.events[6].name == 'start'
    assert ts2.events[6].time == 18
    assert ts2.events[7].name == 'stop'
    assert ts2.events[7].time == 19
    assert ts2.events[9].name == 'start'
    assert ts2.events[9].time == 22
    assert ts2.events[10].name == 'stop'
    assert ts2.events[10].time == 31
    assert ts2.events[11].name == '_'
    assert ts2.events[11].time == 39

    # With falling direction
    ts2 = ktk.cycles.detect_cycles(ts, 'data', 'start', 'stop',
                                   0.5, 0.5, direction1='falling')
    assert ts2.events[0].name == 'start'
    assert ts2.events[0].time == 6
    assert ts2.events[1].name == 'stop'
    assert ts2.events[1].time == 10
    assert ts2.events[3].name == 'start'
    assert ts2.events[3].time == 13
    assert ts2.events[4].name == 'stop'
    assert ts2.events[4].time == 18
    assert ts2.events[6].name == 'start'
    assert ts2.events[6].time == 19
    assert ts2.events[7].name == 'stop'
    assert ts2.events[7].time == 22
    assert ts2.events[9].name == 'start'
    assert ts2.events[9].time == 31
    assert ts2.events[10].name == 'stop'
    assert ts2.events[10].time == 39


    # With minimal cycles
    ts3 = ktk.cycles.detect_cycles(ts, 'data', 'start', 'stop',
                                   0.5, 0.5,
                                   min_length1=2,
                                   min_length2=4)
    assert ts3.events[0].name == 'start'
    assert ts3.events[0].time == 2
    assert ts3.events[1].name == 'stop'
    assert ts3.events[1].time == 6
    assert ts3.events[2].name == '_'
    assert ts3.events[2].time == 10
    assert ts3.events[3].name == 'start'
    assert ts3.events[3].time == 10
    assert ts3.events[4].name == 'stop'
    assert ts3.events[4].time == 13
    assert ts3.events[5].name == '_'
    assert ts3.events[5].time == 18
    assert ts3.events[6].name == 'start'
    assert ts3.events[6].time == 22
    assert ts3.events[7].name == 'stop'
    assert ts3.events[7].time == 31
    assert ts3.events[8].name == '_'
    assert ts3.events[8].time == 39

    # With both minimal and maximal cycles
    ts4 = ktk.cycles.detect_cycles(ts, 'data', 'start', 'stop',
                                   0.5, 0.5,
                                   min_length1=4,
                                   max_length1=8,
                                   min_length2=3,
                                   max_length2=7)
    assert ts4.events[0].name == 'start'
    assert ts4.events[0].time == 2
    assert ts4.events[1].name == 'stop'
    assert ts4.events[1].time == 6
    assert ts4.events[2].name == '_'
    assert ts4.events[2].time == 10
    assert len(ts4.events) == 3

    # With target heights
    t = np.arange(40)
    d = np.array([0, 0,
                 1, 1, 2, 1,
                 0, 0, -1, 0,
                 1, 1, 1,
                 0, 0, 0, 0, 0,
                 1,
                 0, 0, 0,
                 1, 1, 1, 1, 1, 1, 1, 1, 1,
                 0, 0, 0, 0, 0, 0, 0, 0,
                 1])
    ts = ktk.TimeSeries(time=t, data={'data': d})

    ts5 = ktk.cycles.detect_cycles(ts, 'data', 'start', 'stop',
                                   0.5, 0.5,
                                   target_height1=2,
                                   target_height2=-1)
    assert ts5.events[0].name == 'start'
    assert ts5.events[0].time == 2
    assert ts5.events[1].name == 'stop'
    assert ts5.events[1].time == 6
    assert ts5.events[2].name == '_'
    assert ts5.events[2].time == 10
    assert len(ts5.events) == 3


def test_normalize():
    # Create a TimeSeries with some events directly synced with the data and
    # some other that aren't.
    ts = ktk.TimeSeries()
    ts.time = np.linspace(0, 20, 201)  # 0 to 20 seconds by 0.1s
    ts.data['test'] = ts.time ** 2
    ts.add_event(1.0, 'push')
    ts.add_event(2.0, 'recovery')
    ts.add_event(3.0, 'push')
    ts.add_event(3.95, 'recovery')
    ts.add_event(5.0, 'push')
    ts.add_event(6.05, 'recovery')
    ts.add_event(7.05, 'push')
    ts.add_event(8.0, 'recovery')
    ts.add_event(8.95, 'push')
    ts.add_event(10.05, 'recovery')

    ts1 = ktk.cycles.time_normalize(ts, 'push', 'recovery')
    assert len(ts1.events) == 10  # No missing events

    # Test that if we re-time-normalize, we obtain the same TimeSeries
    ts2 = ktk.cycles.time_normalize(ts1, 'push', '_')
    assert ts1 == ts2

    # Samething but with push to next push
    ts1 = ktk.cycles.time_normalize(ts, 'push', '_')
    assert len(ts1.events) == 12  # No missing events

    # Test that if we re-time-normalize, we obtain the same TimeSeries
    ts2 = ktk.cycles.time_normalize(ts1, 'push', '_')
    assert ts1 == ts2

    # There should be no nan in ts2
    assert ~ts2.isnan('test').all()


def test_most_repeatable_cycles():
    # Create a TimeSeries with 5 cycles, one of those is different from
    # the others
    data = np.array([
        np.sin(np.arange(0, 10, 0.1)) + 0.00,  # 0 - most diff. 3rd removed
        np.sin(np.arange(0, 10, 0.1)) + 0.10,  # 1 - 1st of remain. 5th removed
        np.cos(np.arange(0, 10, 0.1)) + 0.15,  # 2 - cos. 2nd removed
        np.sin(np.arange(0, 10, 0.1)) + 0.15,  # 3 - with nans. 1st removed
        np.sin(np.arange(0, 10, 0.1)) + 0.12])  # 4 - 2nd of remn. 4th removed
    # Put some nans in the fourth cycle (which should be discarted entirely)
    data[3, 30] = np.nan

    test = ktk.cycles.most_repeatable_cycles(data)

    assert test == [1, 4, 0, 2]


def test_stack_unstack():
    # Create a periodic TimeSeries, time-normalize and stack
    ts = ktk.TimeSeries()
    ts.time = np.arange(1000)
    ts.data['sin'] = np.sin(ts.time / 100 * 2 * np.pi)
    ts.data['cos'] = np.cos(ts.time / 100 * 2 * np.pi)

    data = ktk.cycles.stack(ts)

    for i_cycle in range(10):

        assert np.all(np.abs(data['sin'][i_cycle] - np.sin(
            np.arange(100) / 100 * 2 * np.pi)) < 1E-10)

        assert np.all(np.abs(data['cos'][i_cycle] - np.cos(
            np.arange(100) / 100 * 2 * np.pi)) < 1E-10)

    # Test unstack
    ts2 = ktk.cycles.unstack(data)
    assert np.all(np.abs(ts2.data['sin'] - ts.data['sin']) < 1E-10)
    assert np.all(np.abs(ts2.data['cos'] - ts.data['cos']) < 1E-10)
    assert np.all(np.abs(ts2.time - ts.time) < 1E-10)



# Commented since stack_events is also commented for now.
#
# def test_stack_events():
#     # Create a TimeSeries with different time-normalized events
#     ts = ktk.TimeSeries(time=np.arange(400))  # 4 cycles of 100%
#     ts.add_event(9, 'event1')    # event1 at 9% of cycle 0
#     ts.add_event(110, 'event1')  # event1 at 10% of cycle 1
#     ts.add_event(312, 'event1')  # event1 at 12% of cycle 3
#     ts.add_event(382, 'event1')  # 2nd occurr. event1 at 82% of cycle 3
#     ts.add_event(1, 'event2')  # event2 at 1% of cycle 0
#     ts.add_event(5, 'event2')  # event2 at 5% of cycle 0

#     # Stack these events
#     events = ktk.cycles.stack_events(ts)
#     assert events['event1'] == [[9.0], [10.0], [], [12.0, 82.0]]
#     assert events['event2'] == [[1.0, 5.0], [], [], []]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
