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
import ktk
import numpy as np

def test_empty_constructor():
    ts = ktk.TimeSeries()
    assert isinstance(ts.time, np.ndarray)
    assert isinstance(ts.data, dict)
    assert isinstance(ts.time_info, dict)
    assert isinstance(ts.data_info, dict)
    assert isinstance(ts.events, list)
    assert ts.time_info['Unit'] == 's'

def test_add_data_info():
    ts = ktk.TimeSeries()
    ts.add_data_info('Force', 'Unit', 'N')
    assert ts.data_info['Force']['Unit'] == 'N'

def test_get_event_time():
    ts = ktk.TimeSeries()
    ts.add_event(5.5, 'event1')
    ts.add_event(10.8, 'event2')
    ts.add_event(2.3, 'event2')
    assert ts.get_event_time('event1') == 5.5
    assert ts.get_event_time('event2', 0) == 2.3
    assert ts.get_event_time('event2', 1) == 10.8

def test_get_ts_at_event___get_ts_at_time():
    ts = ktk.TimeSeries()
    ts.time = np.linspace(0, 99, 100)
    time_as_column = np.reshape(ts.time, (-1, 1))
    ts.data['Forces'] = np.block(
            [time_as_column, time_as_column**2, time_as_column**3])
    ts.data['Moments'] = np.block(
            [time_as_column**2, time_as_column**3, time_as_column**4])
    ts.add_event(5.5, 'event1')
    ts.add_event(10.8, 'event2')
    ts.add_event(2.3, 'event2')
    new_ts = ts.get_ts_at_event('event1')
    assert new_ts.time == 5
    new_ts = ts.get_ts_at_event('event2')
    assert new_ts.time == 2
    new_ts = ts.get_ts_at_event('event2', 1)
    assert new_ts.time == 11

def tes_get_ts_before_time():
    ts = ktk.TimeSeries(time=np.linspace(0, 9, 10))
    new_ts = ts.get_ts_before_time(-2)
    assert new_ts.time.tolist() == []
    new_ts = ts.get_ts_before_time(13)
    assert new_ts.time.tolist() == [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]

def test_get_ts_after_time():
    ts = ktk.TimeSeries(time=np.linspace(0, 9, 10))
    new_ts = ts.get_ts_after_time(-2)
    assert new_ts.time.tolist() == [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]
    new_ts = ts.get_ts_after_time(13)
    assert new_ts.time.tolist() == []

def test_get_ts_between_times():
    ts = ktk.TimeSeries(time=np.linspace(0, 9, 10))
    new_ts = ts.get_ts_between_times(-2, 13)
    assert new_ts.time.tolist() == [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]
    new_ts = ts.get_ts_between_times(-2, -1)
    assert new_ts.time.tolist() == []

def test_merge_and_resample():
    # Begin with two timeseries with identical times
    ts1 = ktk.TimeSeries()
    ts1.time = np.linspace(0, 99, 100)
    ts1.data['signal1'] = np.random.rand(100, 2)
    ts1.data['signal2'] = np.random.rand(100, 2)
    ts1.data['signal3'] = np.random.rand(100, 2)
    ts1.add_data_info('signal1', 'Unit', 'Unit1')
    ts1.add_data_info('signal2', 'Unit', 'Unit2')
    ts1.add_data_info('signal3', 'Unit', 'Unit3')
    ts1.add_event(1.54, 'test_event1')
    ts1.add_event(10.2, 'test_event2')
    ts1.add_event(100, 'test_event3')

    ts2 = ktk.TimeSeries()
    ts2.time = np.linspace(0, 99, 100)
    ts2.data['signal4'] = np.random.rand(100, 2)
    ts2.data['signal5'] = np.random.rand(100, 2)
    ts2.data['signal6'] = np.random.rand(100, 2)
    ts2.add_data_info('signal4', 'Unit', 'Unit1')
    ts2.add_data_info('signal5', 'Unit', 'Unit2')
    ts2.add_data_info('signal6', 'Unit', 'Unit3')
    ts2.add_event(1.54, 'test_event4')
    ts2.add_event(10.2, 'test_event5')
    ts2.add_event(100, 'test_event6')

    ts1.merge(ts2)

    assert np.all(ts1.data['signal4'] == ts2.data['signal4'])
    assert np.all(ts1.data['signal5'] == ts2.data['signal5'])
    assert np.all(ts1.data['signal6'] == ts2.data['signal6'])
    assert np.all(ts1.data_info['signal4']['Unit'] ==
                  ts2.data_info['signal4']['Unit'])
    assert np.all(ts1.data_info['signal5']['Unit'] ==
                  ts2.data_info['signal5']['Unit'])
    assert np.all(ts1.data_info['signal6']['Unit'] ==
                  ts2.data_info['signal6']['Unit'])

    # Try with two timeseries that don't fit in time. It must generate an
    # exception.
    ts1 = ktk.TimeSeries()
    ts1.time = np.linspace(0, 99, 100, endpoint=False)
    ts1.data['signal1'] = np.random.rand(100, 2)
    ts1.data['signal2'] = np.random.rand(100, 2)
    ts1.data['signal3'] = np.random.rand(100, 2)
    ts1.add_data_info('signal1', 'Unit', 'Unit1')
    ts1.add_data_info('signal2', 'Unit', 'Unit2')
    ts1.add_data_info('signal3', 'Unit', 'Unit3')
    ts1.add_event(1.54, 'test_event1')
    ts1.add_event(10.2, 'test_event2')
    ts1.add_event(100, 'test_event3')

    ts2 = ktk.TimeSeries()
    ts2.time = np.linspace(0, 99, 300, endpoint=False)
    ts2.data['signal4'] = ts2.time ** 2
    ts2.data['signal5'] = np.random.rand(300, 2)
    ts2.data['signal6'] = np.random.rand(300, 2)
    ts2.add_data_info('signal4', 'Unit', 'Unit1')
    ts2.add_data_info('signal5', 'Unit', 'Unit2')
    ts2.add_data_info('signal6', 'Unit', 'Unit3')
    ts2.add_event(1.54, 'test_event4')
    ts2.add_event(10.2, 'test_event5')
    ts2.add_event(100, 'test_event6')

    try:
        ts1.merge(ts2)
        raise Exception('This command should have raised a ValueError.')
    except ValueError:
        pass

    # Try the same thing but with linear resampling
    ts1.merge(ts2, resample=True)

    def _assert_almost_equal(one, two):
        assert np.max(np.abs(one - two)) < 1E-6

    _assert_almost_equal(ts1.data['signal4'], ts2.data['signal4'][0::3])
    _assert_almost_equal(ts1.data['signal5'], ts2.data['signal5'][0::3])
    _assert_almost_equal(ts1.data['signal6'], ts2.data['signal6'][0::3])
    assert ts1.data_info['signal4']['Unit'] == ts2.data_info['signal4']['Unit']
    assert ts1.data_info['signal5']['Unit'] == ts2.data_info['signal5']['Unit']
    assert ts1.data_info['signal6']['Unit'] == ts2.data_info['signal6']['Unit']

def test_rename_data():
    ts = ktk.TimeSeries(time=np.arange(100))
    ts.data['data1'] = ts.time.copy()
    ts.data['data2'] = ts.time.copy()
    ts.add_data_info('data2', 'Unit', 'N')

    ts.rename_data('data2', 'data3')

    assert 'data2' not in ts.data
    assert 'data2' not in ts.data_info
    assert np.all(ts.data['data3'] == ts.time)
    assert ts.data_info['data3']['Unit'] == 'N'


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])

