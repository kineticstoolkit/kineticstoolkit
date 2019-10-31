#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KTK development functions.

Author: Félix Chénier
Date: July 2019
"""

import ktk
import subprocess as _subprocess
import webbrowser as _webbrowser
import pytest
import os
import ktk.dev.tutorialcompiler as tutorialcompiler

def run_tests(module=None):
    """Run all unit tests."""
    cwd = os.getcwd()
    os.chdir(ktk.config['RootFolder'] + '/tutorials')
    tutorialcompiler.compile('tutorial_pushrimkinetics.py',
                             'pushrimkinetics.html')
    os.chdir(cwd)

    pytest.main([ktk.config['RootFolder'] + '/ktk/dev'])


def generate_tutorials():
    """Update the Jupyter tutorials into their final html form."""
    _subprocess.call(['jupyter-nbconvert', '--to=html', '--execute',
                      ktk.config['RootFolder'] + '/tutorials/*.ipynb'])
    _webbrowser.open('file:///' + ktk.config['RootFolder'] + '/tutorials/index.html',
                     new=2)


def release():
    run_tests()
    generate_tutorials()
