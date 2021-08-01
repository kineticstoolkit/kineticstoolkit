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


def test_TimeSeriesEvent():
    """Test basic getters and setters."""
    event = ktk.TimeSeriesEvent()
    event.time = 1
    event.name = 'one'
    assert event.time == 1
    assert event.name == 'one'


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

    ts2.data['test'] = np.arange(10)
    assert ts != ts2
    ts.data['test'] = np.arange(10) + 1  # Different but same size
    assert ts != ts2
    ts.data['test'] = np.arange(20)  # Different sizes
    assert ts != ts2
    ts.data['test'] = np.arange(10)  # Now the samething
    assert ts == ts2

    ts2.time_info['info'] = 'test'
    assert ts != ts2
    ts.time_info['info'] = 'test'
    assert ts == ts2

    ts2 = ts2.add_data_info('test', 'info', 'test')
    assert ts != ts2
    ts = ts.add_data_info('test', 'info', 'test')
    assert ts == ts2

    ts2 = ts2.add_event(1, 'one')
    assert ts != ts2
    ts = ts.add_event(1, 'one')
    assert ts == ts2


def test_empty_constructor():
    ts = ktk.TimeSeries()
    assert isinstance(ts.time, np.ndarray)
    assert isinstance(ts.data, dict)
    assert isinstance(ts.time_info, dict)
    assert isinstance(ts.data_info, dict)
    assert isinstance(ts.events, list)
    assert ts.time_info['Unit'] == 's'


def test_from_dataframe():
    df = pd.DataFrame(columns=['Data0', 'Data1[0,0]', 'Data1[0,1]',
                               'Data1[1,0]', 'Data1[1,1]'])
    df['Data0'] = np.arange(2)
    df['Data1[0,0]'] = np.arange(2) + 1
    df['Data1[0,1]'] = np.arange(2) + 2
    df['Data1[1,0]'] = np.arange(2) + 3
    df['Data1[1,1]'] = np.arange(2) + 4
    ts = ktk.TimeSeries.from_dataframe(df)
    assert np.allclose(ts.data['Data0'], [0, 1])
    assert np.allclose(ts.data['Data1'], [[[1, 2], [3, 4]], [[2, 3], [4, 5]]])


def test_add_remove_data_info():
    ts = ktk.TimeSeries()
    ts = ts.add_data_info('Force', 'Unit', 'N')
    ts = ts.add_data_info('Force', 'Other', 'hello')

    # Check that data_info was added
    assert ts.data_info['Force']['Unit'] == 'N'
    assert ts.data_info['Force']['Other'] == 'hello'

    # Test removing non-existing data_info
    ts = ts.remove_data_info('Nonexisting', 'Other')
    assert len(ts.data_info['Force']) == 2

    # Test removing existing data_info
    ts = ts.remove_data_info('Force', 'Other')
    assert ts.data_info['Force']['Unit'] == 'N'
    assert len(ts.data_info['Force']) == 1


def test_rename_remove_data():
    ts = ktk.TimeSeries()
    ts.data['Force'] = np.arange(10)
    ts = ts.rename_data('Force', 'Moment')
    assert np.allclose(ts.data['Moment'], np.arange(10))

    # Same test but with data info
    ts = ts.add_data_info('Moment', 'Unit', 'Nm')
    ts = ts.rename_data('Moment', 'Power')
    assert np.allclose(ts.data['Power'], np.arange(10))
    assert ts.data_info['Power']['Unit'] == 'Nm'

    # Rename inexistent data
    ts = ts.rename_data('NoKey', 'Anything')
    assert np.allclose(ts.data['Power'], np.arange(10))
    assert len(ts.data) == 1
    assert len(ts.data_info) == 1

    # Remove inexistent data
    ts = ts.remove_data('NoKey')
    assert np.allclose(ts.data['Power'], np.arange(10))
    assert len(ts.data) == 1
    assert len(ts.data_info) == 1

    # Remove data
    ts = ts.remove_data('Power')
    assert len(ts.data) == 0
    assert len(ts.data_info) == 0


def test_rename_event():
    # Original doctest
    ts = ktk.TimeSeries()
    ts = ts.add_event(5.5, 'event1')
    ts = ts.add_event(10.8, 'event2')
    ts = ts.add_event(2.3, 'event2')

    assert str(ts.events) == (
        "[TimeSeriesEvent(time=5.5, name='event1'), "
        "TimeSeriesEvent(time=10.8, name='event2'), "
        "TimeSeriesEvent(time=2.3, name='event2')]"
    )

    ts = ts.rename_event('event2', 'event3')

    assert str(ts.events) == (
        "[TimeSeriesEvent(time=5.5, name='event1'), "
        "TimeSeriesEvent(time=10.8, name='event3'), "
        "TimeSeriesEvent(time=2.3, name='event3')]"
    )

    ts = ts.rename_event('event3', 'event4', 0)

    assert str(ts.events) == (
        "[TimeSeriesEvent(time=5.5, name='event1'), "
        "TimeSeriesEvent(time=10.8, name='event3'), "
        "TimeSeriesEvent(time=2.3, name='event4')]"
    )

    # Test renaming an event to a same name (dumb case but should pass)
    ts = ts.rename_event('event4', 'event4')
    assert str(ts.events) == (
        "[TimeSeriesEvent(time=5.5, name='event1'), "
        "TimeSeriesEvent(time=10.8, name='event3'), "
        "TimeSeriesEvent(time=2.3, name='event4')]"
    )

    # Test renaming invalid occurrence
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts = ts.rename_event('event4', 'event5', 10)
    assert str(ts.events) == (
        "[TimeSeriesEvent(time=5.5, name='event1'), "
        "TimeSeriesEvent(time=10.8, name='event3'), "
        "TimeSeriesEvent(time=2.3, name='event4')]"
    )


def test_remove_event():
    # Original doctest
    ts = ktk.TimeSeries()
    ts = ts.add_event(5.5, 'event1')
    ts = ts.add_event(10.8, 'event2')
    ts = ts.add_event(2.3, 'event2')
    assert str(ts.events) == (
        "[TimeSeriesEvent(time=5.5, name='event1'), "
        "TimeSeriesEvent(time=10.8, name='event2'), "
        "TimeSeriesEvent(time=2.3, name='event2')]"
    )

    ts = ts.remove_event('event1')
    assert str(ts.events) == (
        "[TimeSeriesEvent(time=10.8, name='event2'), "
        "TimeSeriesEvent(time=2.3, name='event2')]"
    )

    ts = ts.remove_event('event2', 1)
    assert str(ts.events) == "[TimeSeriesEvent(time=2.3, name='event2')]"

    # Test remove bad occurrence
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts = ts.remove_event('event2', 10)
    assert str(ts.events) == "[TimeSeriesEvent(time=2.3, name='event2')]"


def test_sort_events():
    # Original doctest
    ts = ktk.TimeSeries(time=np.arange(100)/10)
    ts = ts.add_event(2, 'two')
    ts = ts.add_event(1, 'one')
    ts = ts.add_event(3, 'three')
    ts = ts.add_event(3, 'three')
    ts = ts.sort_events(unique=False)
    assert str(ts.events) == (
        "[TimeSeriesEvent(time=1, name='one'), "
        "TimeSeriesEvent(time=2, name='two'), "
        "TimeSeriesEvent(time=3, name='three'), "
        "TimeSeriesEvent(time=3, name='three')]"
    )
    ts = ts.sort_events()
    assert str(ts.events) == (
        "[TimeSeriesEvent(time=1, name='one'), "
        "TimeSeriesEvent(time=2, name='two'), "
        "TimeSeriesEvent(time=3, name='three')]"
    )


def test_get_event_time():
    ts = ktk.TimeSeries()
    ts = ts.add_event(5.5, 'event1')
    ts = ts.add_event(10.8, 'event2')
    ts = ts.add_event(2.3, 'event2')
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
    ts = ts.add_event(5.5, 'event1')
    ts = ts.add_event(10.8, 'event2')
    ts = ts.add_event(2.3, 'event2')
    new_ts = ts.get_ts_at_event('event1')
    assert new_ts.time == 5
    new_ts = ts.get_ts_at_event('event2')
    assert new_ts.time == 2
    new_ts = ts.get_ts_at_event('event2', 1)
    assert new_ts.time == 11


def test_get_ts_before_time():
    ts = ktk.TimeSeries(time=np.linspace(0, 9, 10))
    new_ts = ts.get_ts_before_time(0)
    assert new_ts.time.tolist() == []
    new_ts = ts.get_ts_before_time(5)
    assert new_ts.time.tolist() == [0., 1., 2., 3., 4.]


def test_get_ts_after_time():
    ts = ktk.TimeSeries(time=np.linspace(0, 9, 10))
    new_ts = ts.get_ts_after_time(4)
    assert new_ts.time.tolist() == [5., 6., 7., 8., 9.]
    new_ts = ts.get_ts_after_time(9)
    assert new_ts.time.tolist() == []


def test_get_ts_between_times():
    ts = ktk.TimeSeries(time=np.linspace(0, 9, 10))
    new_ts = ts.get_ts_between_times(-2, 13)
    assert new_ts.time.tolist() == [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]
    new_ts = ts.get_ts_between_times(-2, -1)
    assert new_ts.time.tolist() == []
    # Test that inclusive works
    new_ts = ts.get_ts_between_times(4.5, 7.5, inclusive=True)
    assert new_ts.time.tolist() == [4., 5., 6., 7., 8.]


def test_merge_and_resample():
    # Begin with two timeseries with identical times
    ts1 = ktk.TimeSeries()
    ts1.time = np.linspace(0, 99, 100)
    ts1.data['signal1'] = np.random.rand(100, 2)
    ts1.data['signal2'] = np.random.rand(100, 2)
    ts1.data['signal3'] = np.random.rand(100, 2)
    ts1 = ts1.add_data_info('signal1', 'Unit', 'Unit1')
    ts1 = ts1.add_data_info('signal2', 'Unit', 'Unit2')
    ts1 = ts1.add_data_info('signal3', 'Unit', 'Unit3')
    ts1 = ts1.add_event(1.54, 'test_event1')
    ts1 = ts1.add_event(10.2, 'test_event2')
    ts1 = ts1.add_event(100, 'test_event3')

    ts2 = ktk.TimeSeries()
    ts2.time = np.linspace(0, 99, 100)
    ts2.data['signal4'] = np.random.rand(100, 2)
    ts2.data['signal5'] = np.random.rand(100, 2)
    ts2.data['signal6'] = np.random.rand(100, 2)
    ts2 = ts2.add_data_info('signal4', 'Unit', 'Unit1')
    ts2 = ts2.add_data_info('signal5', 'Unit', 'Unit2')
    ts2 = ts2.add_data_info('signal6', 'Unit', 'Unit3')
    ts2 = ts2.add_event(1.54, 'test_event4')
    ts2 = ts2.add_event(10.2, 'test_event5')
    ts2 = ts2.add_event(100, 'test_event6')

    ts1 = ts1.merge(ts2)

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
    ts1 = ts1.add_data_info('signal1', 'Unit', 'Unit1')
    ts1 = ts1.add_data_info('signal2', 'Unit', 'Unit2')
    ts1 = ts1.add_data_info('signal3', 'Unit', 'Unit3')
    ts1 = ts1.add_event(1.54, 'test_event1')
    ts1 = ts1.add_event(10.2, 'test_event2')
    ts1 = ts1.add_event(100, 'test_event3')

    ts2 = ktk.TimeSeries()
    ts2.time = np.linspace(0, 99, 300, endpoint=False)
    ts2.data['signal4'] = ts2.time ** 2
    ts2.data['signal5'] = np.random.rand(300, 2)
    ts2.data['signal6'] = np.random.rand(300, 2)
    ts2 = ts2.add_data_info('signal4', 'Unit', 'Unit1')
    ts2 = ts2.add_data_info('signal5', 'Unit', 'Unit2')
    ts2 = ts2.add_data_info('signal6', 'Unit', 'Unit3')
    ts2 = ts2.add_event(1.54, 'test_event4')
    ts2 = ts2.add_event(10.2, 'test_event5')
    ts2 = ts2.add_event(100, 'test_event6')

    try:
        ts1.merge(ts2)
        raise Exception('This command should have raised a ValueError.')
    except ValueError:
        pass

    # Try the same thing but with linear resampling
    ts1 = ts1.merge(ts2, resample=True)

    def _assert_almost_equal(one, two):
        assert np.max(np.abs(one - two)) < 1E-6

    _assert_almost_equal(ts1.data['signal4'], ts2.data['signal4'][0::3])
    _assert_almost_equal(ts1.data['signal5'], ts2.data['signal5'][0::3])
    _assert_almost_equal(ts1.data['signal6'], ts2.data['signal6'][0::3])
    assert ts1.data_info['signal4']['Unit'] == ts2.data_info['signal4']['Unit']
    assert ts1.data_info['signal5']['Unit'] == ts2.data_info['signal5']['Unit']
    assert ts1.data_info['signal6']['Unit'] == ts2.data_info['signal6']['Unit']


def test_plot():
    """Test that many parameter combinations doesn't crash."""
    ts = ktk.TimeSeries(time=np.arange(100))

    # Test a plot with no data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts.plot()

    # Test a plot with empty data
    ts.data['data1'] = np.array([])

    # Add some data
    ts.data['data1'] = ts.time.copy()
    fig = plt.figure()
    ts.plot()
    plt.close(fig)

    # Add another data
    ts.data['data2'] = np.hstack([ts.time[:, np.newaxis],
                                  ts.time[:, np.newaxis]])
    fig = plt.figure()
    ts.plot()
    plt.close(fig)

    # Add units
    ts = ts.add_data_info('data1', 'Unit', 'm')
    fig = plt.figure()
    ts.plot()
    plt.close(fig)

    # Add another unit
    ts = ts.add_data_info('data2', 'Unit', 'mm')
    fig = plt.figure()
    ts.plot()
    plt.close(fig)

    # Plot only one signal
    fig = plt.figure()
    ts.plot(['data1'])
    plt.close(fig)

    # Add events
    ts = ts.add_event(0, 'event')
    ts = ts.add_event(2, '_')
    ts = ts.add_event(3)
    fig = plt.figure()
    ts.plot()
    plt.close(fig)

    # Plot without event_names and iwhtout legend
    fig = plt.figure()
    ts.plot(['data1'], event_names=False, legend=False)
    plt.close(fig)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
