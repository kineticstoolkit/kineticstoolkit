#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:57:41 2019

@author: felix
"""
import ktk
import numpy as np
import pandas as pd

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

def test_save():
    """Test the save function."""
    # Create a test variable with all possible supported combinations
    a = dict()
    a['TestInt'] = 10
    a['TestFloat'] = 10.843
    a['TestBool'] = True
    a['TestStr'] = """Test string with 'quotes' and "double quotes"."""
    random_variable = np.random.rand(30, 3, 3)
    a['TestArray'] = random_variable
    a['TestDaraFrame'] = pd.DataFrame(random_variable)
    a['TestSeries'] = pd.Series(random_variable[:, 0, 0])
    a['TestTimeSeries'] = ktk.TimeSeries(time=np.arange(30))
    a['TestTimeSeries'].data = {'data1': random_variable,
                                'data2': random_variable[:, 0, 0]}
