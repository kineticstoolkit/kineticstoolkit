#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KTK development functions.

Author: Félix Chénier
Date: July 2019
"""

import subprocess
from ktk import _ROOT_FOLDER

def run_tests():
    """Run all unit tests."""
    subprocess.call(_ROOT_FOLDER + '/dev/unittests/timeseries.py')

def generate_tutorials():
    """Update the Jupyter tutorials into their final html form."""
    subprocess.call(['jupyter-nbconvert', '--to=html', '--execute',
                         _ROOT_FOLDER + '/../tutorials/*.ipynb'])

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
    