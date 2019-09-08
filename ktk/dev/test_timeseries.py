#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the TimeSeries and TimeSeriesEvent class methods.

Author: Félix Chénier
Date: July 2019
"""

import ktk
import numpy as np
import matplotlib.pyplot as plt
import os


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


def test_add_event():
    ts = ktk.TimeSeries()
    ts.add_event(15.34, 'test_event1')
    ts.add_event(99.2, 'test_event2')
    ts.add_event(1, 'test_event3')
    assert ts.events[0].name == 'test_event1'
    assert ts.events[1].name == 'test_event2'
    assert ts.events[2].name == 'test_event3'
    assert ts.events[0].time == 15.34
    assert ts.events[1].time == 99.2
    assert ts.events[2].time == 1


def test_plot():
    ts = ktk.TimeSeries()
    ts.time = np.linspace(0, 99, 100)
    ts.data['signal1'] = np.random.rand(100, 2)
    ts.data['signal2'] = np.random.rand(100, 2)
    ts.data['signal3'] = np.random.rand(100, 2)
    ts.add_data_info('signal1', 'Unit', 'Unit1')
    ts.add_data_info('signal2', 'Unit', 'Unit2')
    ts.add_data_info('signal3', 'Unit', 'Unit3')
    ts.add_event(15.34, 'test_event1')
    ts.add_event(99.2, 'test_event2')
    ts.add_event(1, 'test_event3')
    ts.plot()
    plt.close(plt.gcf())
    # Does not crash ? It's tested.


def test_get_index_before_time():
    ts = ktk.TimeSeries(time=np.array([0, 0.5, 1, 1.5, 2]))
    assert ts.get_index_before_time(0.9) == 1
    assert ts.get_index_before_time(1) == 2
    assert ts.get_index_before_time(1.1) == 2
    assert np.isnan(ts.get_index_before_time(-1))


def test_get_index_at_time():
    ts = ktk.TimeSeries(time=np.array([0, 0.5, 1, 1.5, 2]))
    assert ts.get_index_at_time(0.9) == 2
    assert ts.get_index_at_time(1) == 2
    assert ts.get_index_at_time(1.1) == 2


def test_get_index_after_time():
    ts = ktk.TimeSeries(time=np.array([0, 0.5, 1, 1.5, 2]))
    assert ts.get_index_after_time(0.9) == 2
    assert ts.get_index_after_time(1) == 2
    assert ts.get_index_after_time(1.1) == 3
    assert np.isnan(ts.get_index_after_time(13))


def test_get_event_time():
    ts = ktk.TimeSeries()
    ts.add_event(5.5, 'event1')
    ts.add_event(10.8, 'event2')
    ts.add_event(2.3, 'event2')
    assert ts.get_event_time('event1') == 5.5
    assert ts.get_event_time('event2', 1) == 2.3
    assert ts.get_event_time('event2', 2) == 10.8


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
    new_ts = ts.get_ts_at_event('event2', 2)
    assert new_ts.time == 11


def tes_get_ts_before_time():
    ts = ktk.TimeSeries(time=np.linspace(0, 9, 10))
    new_ts = ts.get_ts_before_time(3)
    assert new_ts.time.tolist() == [0., 1., 2., 3.]
    new_ts = ts.get_ts_before_time(3.5)
    assert new_ts.time.tolist() == [0., 1., 2., 3.]
    new_ts = ts.get_ts_before_time(-2)
    assert new_ts.time.tolist() == []
    new_ts = ts.get_ts_before_time(13)
    assert new_ts.time.tolist() == [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]


def test_get_ts_after_time():
    ts = ktk.TimeSeries(time=np.linspace(0, 9, 10))
    new_ts = ts.get_ts_after_time(3)
    assert new_ts.time.tolist() == [3., 4., 5., 6., 7., 8., 9.]
    new_ts = ts.get_ts_after_time(3.5)
    assert new_ts.time.tolist() == [4., 5., 6., 7., 8., 9.]
    new_ts = ts.get_ts_after_time(-2)
    assert new_ts.time.tolist() == [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]
    new_ts = ts.get_ts_after_time(13)
    assert new_ts.time.tolist() == []


def test_get_ts_between_times():
    ts = ktk.TimeSeries(time=np.linspace(0, 9, 10))
    new_ts = ts.get_ts_between_times(3, 6)
    assert new_ts.time.tolist() == [3., 4., 5., 6.]
    new_ts = ts.get_ts_between_times(3.5, 5.5)
    assert new_ts.time.tolist() == [4., 5.]
    new_ts = ts.get_ts_between_times(-2, 13)
    assert new_ts.time.tolist() == [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]
    new_ts = ts.get_ts_between_times(-2, -1)
    assert new_ts.time.tolist() == []


def test_to_dataframe___from_dataframe___load___save():
    ts = ktk.TimeSeries()
    ts.time = np.linspace(0, 9, 10)
    ts.data['signal1'] = np.random.rand(10)
    ts.data['signal2'] = np.random.rand(10, 3)
    ts.data['signal3'] = np.random.rand(10, 3, 3)
    ts.add_data_info('signal1', 'Unit', 'm/s')
    ts.add_data_info('signal2', 'Unit', 'km/h')
    ts.add_data_info('signal3', 'Unit', 'N')
    ts.add_data_info('signal3', 'signal_type', 'force')
    ts.add_event(1.53, 'test_event1')
    ts.add_event(7.2, 'test_event2')
    ts.add_event(1, 'test_event3')
    df = ts.to_dataframes()
    assert df['data'].time.tolist() == ts.time.tolist()
    assert df['data']['signal1'].tolist() == ts.data['signal1'].tolist()
    assert df['data']['signal2[0]'].tolist() == \
        ts.data['signal2'][:, 0].tolist()
    assert df['data']['signal2[1]'].tolist() == \
        ts.data['signal2'][:, 1].tolist()
    assert df['data']['signal2[2]'].tolist() == \
        ts.data['signal2'][:, 2].tolist()
    assert df['data']['signal3[0,0]'].tolist() == \
        ts.data['signal3'][:, 0, 0].tolist()
    assert df['data']['signal3[0,1]'].tolist() == \
        ts.data['signal3'][:, 0, 1].tolist()
    assert df['data']['signal3[0,2]'].tolist() == \
        ts.data['signal3'][:, 0, 2].tolist()
    assert df['data']['signal3[1,2]'].tolist() == \
        ts.data['signal3'][:, 1, 2].tolist()
    assert df['events']['time'][0] == ts.events[0].time
    assert df['events']['name'][0] == ts.events[0].name
    assert df['events'].time[1] == ts.events[1][0]
    assert df['events'].name[1] == ts.events[1][1]
    assert df['info']['time']['Unit'] == ts.time_info['Unit']
    assert df['info']['signal1']['Unit'] == ts.data_info['signal1']['Unit']
    assert df['info']['signal3']['signal_type'] == \
        ts.data_info['signal3']['signal_type']

    # Form back the original timeseries to test from_dataframes
    ts2 = ktk.TimeSeries.from_dataframes(df['data'],
                                         df['events'],
                                         df['info'])
    assert ts == ts2

    # Test load and save
    try:
        os.remove('test.zip')
    except:
        pass

    ts.save('test.zip')
    ts3 = ktk.TimeSeries.load('test.zip')
    assert ts == ts3
    os.remove('test.zip')
