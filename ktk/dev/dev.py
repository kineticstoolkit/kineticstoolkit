#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KTK development functions.

Author: Félix Chénier
Date: July 2019
"""

import subprocess
from ktk import _ROOT_FOLDER

import unittest
from ktk.dev.unittests.timeseries import timeseriesTest
from ktk.dev.unittests.loadsave import loadsaveTest


def run_tests():
    """Run all unit tests."""
    suite = unittest.TestSuite([
            unittest.TestLoader().loadTestsFromTestCase(timeseriesTest),
            unittest.TestLoader().loadTestsFromTestCase(loadsaveTest)
            ])
    unittest.TextTestRunner(verbosity=2).run(suite)

def generate_tutorials():
    """Update the Jupyter tutorials into their final html form."""
    subprocess.call(['jupyter-nbconvert', '--to=html', '--execute',
                         _ROOT_FOLDER + '/tutorials/*.ipynb'])

def release():
    print("==================")
    print("RUNNING UNIT TESTS")
    print("------------------")
    run_tests()
    
    print("==================")
    print("RUNNING TUTORIALS ")
    print("------------------")
    generate_tutorials()

    print("------------------")
    print("DONE.")
    print("==================")
    