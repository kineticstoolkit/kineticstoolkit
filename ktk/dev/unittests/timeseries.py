#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the TimeSeries and TimeSeriesEvent class methods.

Author: Félix Chénier
Date: July 2019
"""

import unittest
import ktk
import numpy as np
import matplotlib.pyplot as plt


class timeseriesTest(unittest.TestCase):
    """TimeSeries unit tests."""

    def test_empty_constructor(self):
        """Test the empty constructor."""
        ts = ktk.TimeSeries()
        self.assertIsInstance(ts.time, np.ndarray)
        self.assertIsInstance(ts.data, dict)
        self.assertIsInstance(ts.time_info, dict)
        self.assertIsInstance(ts.data_info, dict)
        self.assertIsInstance(ts.events, list)
        self.assertEqual(ts.time_info['unit'], 's')

    def test_add_data_info(self):
        """Test the add_data_info method."""
        ts = ktk.TimeSeries()
        ts.add_data_info('Force', 'unit', 'N')
        self.assertEqual(ts.data_info['Force']['unit'], 'N')

    def test_add_event(self):
        """Test the add_event method."""
        ts = ktk.TimeSeries()
        ts.add_event(15.34, 'test_event1')
        ts.add_event(99.2, 'test_event2')
        ts.add_event(1, 'test_event3')
        self.assertEqual(ts.events[0].name, 'test_event1')
        self.assertEqual(ts.events[1].name, 'test_event2')
        self.assertEqual(ts.events[2].name, 'test_event3')
        self.assertEqual(ts.events[0].time, 15.34)
        self.assertEqual(ts.events[1].time, 99.2)
        self.assertEqual(ts.events[2].time, 1)

    def test_plot(self):
        """Test the plot method."""
        ts = ktk.TimeSeries()
        ts.time = np.linspace(0, 99, 100)
        ts.data['signal1'] = np.random.rand(100, 2)
        ts.data['signal2'] = np.random.rand(100, 2)
        ts.data['signal3'] = np.random.rand(100, 2)
        ts.add_data_info('signal1', 'unit', 'Unit1')
        ts.add_data_info('signal2', 'unit', 'Unit2')
        ts.add_data_info('signal3', 'unit', 'Unit3')
        ts.add_event(15.34, 'test_event1')
        ts.add_event(99.2, 'test_event2')
        ts.add_event(1, 'test_event3')
        ts.plot()
        plt.close(plt.gcf())

    def test_get_index_before_time(self):
        """Test the get_index_at_time method."""
        ts = ktk.TimeSeries(time=np.array([0, 0.5, 1, 1.5, 2]))
        self.assertEqual(ts.get_index_before_time(0.9), 1)
        self.assertEqual(ts.get_index_before_time(1), 2)
        self.assertEqual(ts.get_index_before_time(1.1), 2)
        self.assertTrue(np.isnan(ts.get_index_before_time(-1)))

    def test_get_index_at_time(self):
        """Test the get_index_at_time method."""
        ts = ktk.TimeSeries(time=np.array([0, 0.5, 1, 1.5, 2]))
        self.assertEqual(ts.get_index_at_time(0.9), 2)
        self.assertEqual(ts.get_index_at_time(1), 2)
        self.assertEqual(ts.get_index_at_time(1.1), 2)

    def test_get_index_after_time(self):
        """Test the get_index_at_time method."""
        ts = ktk.TimeSeries(time=np.array([0, 0.5, 1, 1.5, 2]))
        self.assertEqual(ts.get_index_after_time(0.9), 2)
        self.assertEqual(ts.get_index_after_time(1), 2)
        self.assertEqual(ts.get_index_after_time(1.1), 3)
        self.assertTrue(np.isnan(ts.get_index_after_time(13)))

    def test_get_event_time(self):
        """Test the get_event_time method."""
        ts = ktk.TimeSeries()
        ts.add_event(5.5, 'event1')
        ts.add_event(10.8, 'event2')
        ts.add_event(2.3, 'event2')
        self.assertEqual(ts.get_event_time('event1'), 5.5)
        self.assertEqual(ts.get_event_time('event2', 1), 2.3)
        self.assertEqual(ts.get_event_time('event2', 2), 10.8)

    def test_get_ts_at_event(self):
        """Test the get_ts_at_event and get_ts_at_time methods."""
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
        self.assertEqual(new_ts.time, 5)
        new_ts = ts.get_ts_at_event('event2')
        self.assertEqual(new_ts.time, 2)
        new_ts = ts.get_ts_at_event('event2', 2)
        self.assertEqual(new_ts.time, 11)

    def test_get_ts_before_time(self):
        """Test the get_ts_before_time method."""
        ts = ktk.TimeSeries(time=np.linspace(0, 9, 10))
        new_ts = ts.get_ts_before_time(3)
        self.assertListEqual(new_ts.time.tolist(), [0., 1., 2., 3.])
        new_ts = ts.get_ts_before_time(3.5)
        self.assertListEqual(new_ts.time.tolist(), [0., 1., 2., 3.])
        new_ts = ts.get_ts_before_time(-2)
        self.assertListEqual(new_ts.time.tolist(), [])
        new_ts = ts.get_ts_before_time(13)
        self.assertListEqual(new_ts.time.tolist(),
                             [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])

    def test_get_ts_after_time(self):
        """Test the get_ts_after_time method."""
        ts = ktk.TimeSeries(time=np.linspace(0, 9, 10))
        new_ts = ts.get_ts_after_time(3)
        self.assertListEqual(new_ts.time.tolist(),
                             [3., 4., 5., 6., 7., 8., 9.])
        new_ts = ts.get_ts_after_time(3.5)
        self.assertListEqual(new_ts.time.tolist(),
                             [4., 5., 6., 7., 8., 9.])
        new_ts = ts.get_ts_after_time(-2)
        self.assertListEqual(new_ts.time.tolist(),
                             [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
        new_ts = ts.get_ts_after_time(13)
        self.assertListEqual(new_ts.time.tolist(), [])

    def test_get_ts_between_times(self):
        """Test the get_ts_between_times method."""
        ts = ktk.TimeSeries(time=np.linspace(0, 9, 10))
        new_ts = ts.get_ts_between_times(3, 6)
        self.assertListEqual(new_ts.time.tolist(), [3., 4., 5., 6.])
        new_ts = ts.get_ts_between_times(3.5, 5.5)
        self.assertListEqual(new_ts.time.tolist(), [4., 5.])
        new_ts = ts.get_ts_between_times(-2, 13)
        self.assertListEqual(new_ts.time.tolist(),
                             [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
        new_ts = ts.get_ts_between_times(-2, -1)
        self.assertListEqual(new_ts.time.tolist(), [])

    def test_tofrom_dataframes_loadsave(self):
        """Test the to_dataframes, from_dataframes, load and save methods."""
        ts = ktk.TimeSeries()
        ts.time = np.linspace(0, 9, 10)
        ts.data['signal1'] = np.random.rand(10)
        ts.data['signal2'] = np.random.rand(10, 3)
        ts.data['signal3'] = np.random.rand(10, 3, 3)
        ts.add_data_info('signal1', 'unit', 'm/s')
        ts.add_data_info('signal2', 'unit', 'km/h')
        ts.add_data_info('signal3', 'unit', 'N')
        ts.add_data_info('signal3', 'signal_type', 'force')
        ts.add_event(1.53, 'test_event1')
        ts.add_event(7.2, 'test_event2')
        ts.add_event(1, 'test_event3')
        df = ts.to_dataframes()
        self.assertListEqual(df['data'].time.tolist(), ts.time.tolist())
        self.assertListEqual(df['data']['signal1'].tolist(),
                             ts.data['signal1'].tolist())
        self.assertListEqual(df['data']['signal2[0]'].tolist(),
                             ts.data['signal2'][:, 0].tolist())
        self.assertListEqual(df['data']['signal2[1]'].tolist(),
                             ts.data['signal2'][:, 1].tolist())
        self.assertListEqual(df['data']['signal2[2]'].tolist(),
                             ts.data['signal2'][:, 2].tolist())
        self.assertListEqual(df['data']['signal3[0,0]'].tolist(),
                             ts.data['signal3'][:, 0, 0].tolist())
        self.assertListEqual(df['data']['signal3[0,1]'].tolist(),
                             ts.data['signal3'][:, 0, 1].tolist())
        self.assertListEqual(df['data']['signal3[0,2]'].tolist(),
                             ts.data['signal3'][:, 0, 2].tolist())
        self.assertListEqual(df['data']['signal3[1,2]'].tolist(),
                             ts.data['signal3'][:, 1, 2].tolist())
        self.assertEqual(df['events']['time'][0], ts.events[0].time)
        self.assertEqual(df['events']['name'][0], ts.events[0].name)
        self.assertEqual(df['events'].time[1], ts.events[1][0])
        self.assertEqual(df['events'].name[1], ts.events[1][1])
        self.assertEqual(df['info']['time']['unit'], ts.time_info['unit'])
        self.assertEqual(df['info']['signal1']['unit'],
                         ts.data_info['signal1']['unit'])
        self.assertEqual(df['info']['signal3']['signal_type'],
                         ts.data_info['signal3']['signal_type'])







if __name__ == '__main__':
    unittest.main()
