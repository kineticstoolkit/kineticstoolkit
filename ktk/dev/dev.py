#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KTK development functions.

Author: Félix Chénier
Date: July 2019
"""

import subprocess as _subprocess
from ktk import _ROOT_FOLDER
import webbrowser as _webbrowser
import inspect as _inspect

import test_timeseries
import test_loadsave
import test_filters
import test_pushrimkinetics


def run_tests(module=None):
    """Run all unit tests."""

    # If called without arguments, call again with all test modules
    if module is None:
        run_tests(test_timeseries)
        run_tests(test_loadsave)
        run_tests(test_filters)
        run_tests(test_pushrimkinetics)

    # If called with test module, run all functions in this module
    else:
        print('----------------------------------------------')
        print(f'Running tests from {module}')
        tests = _inspect.getmembers(module, _inspect.isfunction)
        for test in tests:
            if test[0][0] != '_':  # Not a private function
                print(test[0])  # Function name
                test[1]()   # Execute function


def generate_tutorials():
    """Update the Jupyter tutorials into their final html form."""
    _subprocess.call(['jupyter-nbconvert', '--to=html', '--execute',
                      _ROOT_FOLDER + '/tutorials/*.ipynb'])
    _webbrowser.open('file:///' + _ROOT_FOLDER + '/tutorials/index.html', new=2)


def release():
    run_tests()
    generate_tutorials()
