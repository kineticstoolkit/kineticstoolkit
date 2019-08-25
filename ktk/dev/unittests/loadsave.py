#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:57:41 2019

@author: felix
"""

import unittest
import ktk
import numpy as np


class loadsaveTest(unittest.TestCase):
    """TimeSeries unit tests."""

    def test_loadmat(self):
        
        if ktk._ISMAC:
            """Test the empty constructor."""
            data = ktk.loadsave.loadmat(
                    ktk._ROOT_FOLDER + 
                    '/tutorials/data/sample_reconstructed_kinetics_racing_wheelchair.mat')
            
            self.assertIsInstance(data['config'], dict)
            self.assertIsInstance(data['kinematics'], dict)
            self.assertIsInstance(data['config']['MarkerLabels'], dict)
            self.assertIsInstance(data['config']['Segments'], dict)
            self.assertIsInstance(data['config']['RigidBodies'], dict)
            self.assertIsInstance(data['config']['VirtualMarkers'], dict)
            self.assertIsInstance(data['config']['VirtualRigidBodies'], dict)
            self.assertIsInstance(data['kinematics']['Markers'], dict)
            self.assertIsInstance(data['kinematics']['RigidBodies'], dict)
            self.assertIsInstance(data['kinematics']['VirtualRigidBodies'], dict)
            
        else:
            print("======================================================")
            print("WARNING - NO UNITTEST FOR ktk.loadsave.loadmat WAS RUN")
            print("BECAUSE THIS FUNCTION IS NOT IMPLEMENTED YET ON WINDOWS.")
            print("======================================================")


if __name__ == '__main__':
    unittest.main()
