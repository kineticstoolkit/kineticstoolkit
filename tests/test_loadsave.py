#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:57:41 2019

@author: felix
"""
import ktk
import numpy as np
import os


# test_loadmat():
# This test is implicitly done with test_read_c3d_file in ktk.kinematics


def test_save_load():
    """Test the save and load functions."""
    # Create a test variable with all possible supported combinations
    random_variable = np.random.rand(5, 2, 2)
    ts = ktk.TimeSeries()
    ts.time = np.linspace(0, 9, 10)
    ts.data['signal1'] = np.random.rand(10)
    ts.data['signal2'] = np.random.rand(10, 3)
    ts.data['signal3'] = np.random.rand(10, 3, 3)
    ts.add_data_info('signal1', 'Unit', 'm/s')
    ts.add_data_info('signal2', 'Unit', 'km/h')
    ts.add_data_info('signal3', 'Unit', 'N')
    ts.add_data_info('signal3', 'SignalType', 'force')
    ts.add_event(1.53, 'TestEvent1')
    ts.add_event(7.2, 'TestEvent2')
    ts.add_event(1, 'TestEvent3')

    a = dict()
    a['TestTimeSeries'] = ts
    a['TestInt'] = 10
    a['TestFloat'] = np.pi
    a['TestBool'] = True
    a['TestStr'] = """Test string with 'quotes' and "double quotes"."""
    a['TestComplex'] = (34.05+2j)
    a['TestArray'] = random_variable
    a['TestList'] = [0, 'test', True]
    a['TestTuple'] = (1, 'test2', False)
    a['TestBigList'] = list(np.arange(-1, 1, 1E-4))
    a['TestDict'] = {'key1': 'value1',
                      'key2': 10,
                      'key3': True}
    a['TestComplexDict'] = {'key1': 'value1',
                            'key2': 10,
                            'key3': ts.copy()}
    a['TestComplexList'] = ['value1', 10, ts.copy(),
                            'value2', 12, None,
                            True, np.pi, (34.05+2j),
                            [1, 2, 3, 4], (1, 2, 3, 4),
                            {'key1': 'value1',
                              'key2': 10,
                              'key3': ts.copy()}
                            ]
    a['TestComplexTuple'] = tuple(a['TestComplexList'])

    ktk.save('test.ktk.zip', a)
    b = ktk.load('test.ktk.zip')
    os.remove('test.ktk.zip')

    assert a['TestTimeSeries'] == b['TestTimeSeries']
    assert a['TestInt'] == b['TestInt']
    assert a['TestFloat'] == b['TestFloat']
    assert a['TestBool'] == b['TestBool']
    assert a['TestStr'] == b['TestStr']
    assert a['TestComplex'] == b['TestComplex']
    assert np.sum(np.abs(a['TestArray'] - b['TestArray'])) < 1E-10
    assert a['TestList'] == b['TestList']
    assert a['TestTuple'] == b['TestTuple']
    assert a['TestBigList'] == b['TestBigList']
    assert a['TestDict'] == b['TestDict']
    assert a['TestComplexDict']['key1'] == b['TestComplexDict']['key1']
    assert a['TestComplexDict']['key2'] == b['TestComplexDict']['key2']
    assert a['TestComplexDict']['key3'] == b['TestComplexDict']['key3']
    for i in range(10):
        assert a['TestComplexList'][i] == b['TestComplexList'][i]
    assert a['TestComplexList'][10] == b['TestComplexList'][10]
    for i in range(10):
        assert a['TestComplexTuple'][i] == b['TestComplexTuple'][i]
    assert a['TestComplexTuple'][10] == b['TestComplexTuple'][10]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
