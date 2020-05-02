#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020 Félix Chénier

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"

"""
KTK development functions
-------------------------
This module contains fonctions related to development, tests and release.

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

| Getting started            | Low-level modules             | High-level modules                      |
|:--------------------------:|:-----------------------------:|:---------------------------------------:|
| [Home](index.html)         | [TimeSeries](timeseries.html) | [kinematics](kinematics.html)           |
| [Installing](install.html) | [filters](filters.html)       | [pushrimkinetics](pushrimkinetics.html) |
|                            | [geometry](geometry.html)     | [inversedynamics](inversedynamics.html) |
|                            |                               | [Player](player.html)                   |
|                            |                               | [DBinterface](dbinterface.html)         |

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
