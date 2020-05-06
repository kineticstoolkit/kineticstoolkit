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
import os
import subprocess
import shutil
import webbrowser

from functools import partial
from threading import Thread
from time import sleep


def run_unit_tests():
    """Run all unit tests."""
    # Run pytest in another process to ensure that the workspace is and stays
    # clean, and all Matplotlib windows are closed correctly after the tests.
    cwd = os.getcwd()
    os.chdir(ktk.config['RootFolder'] + '/tests')
    subprocess.call(['pytest', '--ignore=interactive'])
    os.chdir(cwd)


def run_doc_tests():
    """Run all doc tests."""
    print('Running doc tests...')
    cwd = os.getcwd()
    os.chdir(ktk.config['RootFolder'] + '/ktk')
    for file in os.listdir():
        if file.endswith('.py'):
            print(file)
            subprocess.call(['python', '-m', 'doctest', file])
    os.chdir(cwd)


def generate_tutorials():
    """Generate the tutorials in html."""
    cwd = os.getcwd()
    os.chdir(ktk.config['RootFolder'] + '/tutorials')
    subprocess.call(['jupyter-nbconvert', '--to', 'html_toc',
                     'index.ipynb'])
    os.chdir(cwd)

    # Open tutorial
    webbrowser.open_new_tab(
        'file://' + ktk.config['RootFolder'] + '/tutorials/index.html')


def generate_doc():
    """Generate ktk's reference using pdoc."""
    cwd = os.getcwd()

    # Create a mirror of ktk
    shutil.rmtree(ktk.config['RootFolder'] + '/tmp', ignore_errors=True)
    shutil.rmtree(ktk.config['RootFolder'] + '/doc', ignore_errors=True)
    os.mkdir(ktk.config['RootFolder'] + '/tmp')
    shutil.copytree(ktk.config['RootFolder'] + '/ktk',
                    ktk.config['RootFolder'] + '/tmp/ktk')

    # Append the class definitions to ktk/__init__.py
    # so that pdoc includes the classes into ktk's toplevel
    with open(ktk.config['RootFolder'] + '/ktk/_timeseries.py',
              'r') as in_file:
        with open(ktk.config['RootFolder'] + '/tmp/ktk/__init__.py',
                  'a') as out_file:
            for line in in_file:
                out_file.write(line)

    # Run pdoc
    os.chdir(ktk.config['RootFolder'] + '/tmp')
    subprocess.call(['pdoc', '--html', '--config', 'show_source_code=False',
                     '--output-dir', ktk.config['RootFolder'] + '/doc', 'ktk'])

    # Cleanup
    shutil.rmtree(ktk.config['RootFolder'] + '/tmp', ignore_errors=True)
    os.chdir(cwd)

    # Open doc
    webbrowser.open_new_tab(
        'file://' + ktk.config['RootFolder'] + '/doc/ktk/index.html')


def update_readme():
    """Copy ktk's docstring into readme.md."""
    with open(ktk.config['RootFolder'] + '/README.md', 'w') as fid:
        fid.write(ktk.__doc__)


def compile_for_pypi():
    """Compile for PyPi."""
    update_readme()
    shutil.rmtree(ktk.config['RootFolder'] + '/dist', ignore_errors=True)
    shutil.rmtree(ktk.config['RootFolder'] + '/build', ignore_errors=True)
    os.chdir(ktk.config['RootFolder'])
    subprocess.call(['python', 'setup.py', 'sdist', 'bdist_wheel'])


def upload_to_pypi():
    """Upload to PyPi."""
    root_folder = ktk.config['RootFolder']
    subprocess.call([
        'osascript',
        '-e',
        'tell application "Terminal" to do script '
        f'"conda activate ktk; cd {root_folder}; twine upload dist/*"'])


def release():
    """Run all functions for release, without packaging and uploading."""
    run_doc_tests()
    run_unit_tests()
    update_readme()
    generate_tutorials()
    generate_doc()
