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
import matplotlib.pyplot as plt

def run_tests(module=None):
    """Run all unit tests."""

    # Run all tutorials
    cwd = os.getcwd()
    os.chdir(ktk.config['RootFolder'] + '/tutorials')
    files = os.listdir()
    for file in files:
        if file[-3:].lower() == '.py':
            print('==========================================')
            print('Testing ' + file[:-3] + ' tutorial...')
            print('------------------------------------------')
            exec(open(file).read())
            plt.close('all')
#            _subprocess.call([os.sys.executable, file])

    os.chdir(cwd)

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
    """
    cwd = os.getcwd()
    os.chdir(ktk.config['RootFolder'] + '/tutorials')

    if name is None:
        files = os.listdir()
    else:
        files = [name + '.py']

    for file in files:
        if file[-3:].lower() == '.py':
            print('==========================================')
            print('Building ' + file[:-3] + ' tutorial...')
            print('------------------------------------------')
            tutorialcompiler.compile(file, file[:-3] + '.html')

    os.chdir(cwd)


#    _subprocess.call(['jupyter-nbconvert', '--to=html', '--execute',
#                      ktk.config['RootFolder'] + '/tutorials/*.ipynb'])
    _webbrowser.open('file:///' + ktk.config['RootFolder'] + '/tutorials/index.html',
                     new=2)




def release():
    run_tests()
    generate_tutorials()
