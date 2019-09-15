#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:57:41 2019

@author: felix
"""
import ktk
import numpy as np
import pandas as pd
import os

def test_loadmat():
    if ktk.config['IsMac']:
        """Test the empty constructor."""
        data = ktk.loadsave.loadmat(
                ktk.config['RootFolder'] +
                '/tutorials/data/sample_reconstructed_kinetics_racing_wheelchair.mat')

        assert isinstance(data['config'], dict)
        assert isinstance(data['kinematics'], dict)
        assert isinstance(data['config']['MarkerLabels'], dict)
        assert isinstance(data['config']['Segments'], dict)
        assert isinstance(data['config']['RigidBodies'], dict)
        assert isinstance(data['config']['VirtualMarkers'], dict)
        assert isinstance(data['config']['VirtualRigidBodies'], dict)
        assert isinstance(data['kinematics']['Markers'], dict)
        assert isinstance(data['kinematics']['RigidBodies'], dict)
        assert isinstance(data['kinematics']['VirtualRigidBodies'], dict)

    else:
        print("======================================================")
        print("WARNING - NO UNITTEST FOR ktk.loadsave.loadmat WAS RUN")
        print("BECAUSE THIS FUNCTION IS NOT IMPLEMENTED YET ON WINDOWS.")
        print("======================================================")


def test_save_load():
    """Test the save and load functions."""
    # Create a test variable with all possible supported combinations
    random_variable = np.random.rand(30, 3, 3)
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

    ktk.loadsave.save('test.ktk.zip', a)
    b = ktk.loadsave.load('test.ktk.zip')
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
