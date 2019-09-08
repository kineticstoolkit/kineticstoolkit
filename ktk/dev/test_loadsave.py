#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:57:41 2019

@author: felix
"""
import ktk

def test_loadsave():
    if ktk._ISMAC:
        """Test the empty constructor."""
        data = ktk.loadsave.loadmat(
                ktk._ROOT_FOLDER +
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
