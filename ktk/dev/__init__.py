#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KTK development functions.

Author: Félix Chénier
Date: July 2019
"""

import ktk
import pytest
import os
import ktk.dev.tutorialcompiler as tutorialcompiler
import matplotlib.pyplot as plt
from functools import partial
from threading import Thread
from time import sleep

def run_tests(module=None):
    """Run all unit tests."""

#    # Run all tutorials
#    cwd = os.getcwd()
#    os.chdir(ktk.config['RootFolder'] + '/tutorials')
#    files = os.listdir()
#    for file in files:
#        if file[-3:].lower() == '.py':
#            print('==========================================')
#            print('Testing ' + file[:-3] + ' tutorial...')
#            print('------------------------------------------')
#            exec(open(file).read())
#            plt.close('all')
#            _subprocess.call([os.sys.executable, file])
#
#    os.chdir(cwd)

    # Run all old-fashioned tests
    pytest.main([ktk.config['RootFolder'] + '/ktk/dev'])


def generate_tutorials(name=None):
    """
    Generate the tutorials into their final html form.

    Parameters
    ----------
    name : str (optional)
        Name of the tutorial to build. For example: 'dbinterface'.
        Default is None, which means all tutorials are generated.

    Returns
    -------
    None.

    Open the function to modify the common header that is generated for each
    tutorial page, including the menu.
    """

    header = '''
Kinetics Toolkit (ktk)
======================
[Home](index.html) -
[Installing](install.html) -
[TimeSeries](timeseries.html) -
[filters](filters.html) -
[geometry](geometry.html) -
[kinematics](kinematics.html) -
[pushrimkinetics](pushrimkinetics.html) -
[inversedynamics](inversedynamics.html) -
[Player](player.html) -
[dbinterface](dbinterface.html)

-----------------------
'''

    cwd = os.getcwd()
    os.chdir(ktk.config['RootFolder'] + '/tutorials')

    if name is None:
        files = os.listdir()
    else:
        files = [name + '.py']

    print('==========================================')
    threads_running = [0]  # List of len 1 with the number of running threads
    for file in files:
        if file[-3:].lower() == '.py':
            print('Starting compiling ' + file[:-3] + ' tutorial...')
            threaded_function = partial(tutorialcompiler.compile,
                                        file, file[:-3] + '.html', header,
                                        threads_running)
            thread = Thread(target=threaded_function)
            thread.start()
            threads_running[0] += 1
#            tutorialcompiler.compile(file, file[:-3] + '.html', header)

    os.chdir(cwd)

    while threads_running[0] > 0:
        sleep(0.5)

    ktk.tutorials()


def release():
    run_tests()
    generate_tutorials()
