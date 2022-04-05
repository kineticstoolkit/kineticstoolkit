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

"""
Provide fonctions related to development, tests and release of Kinetics
Toolkit.

Note
----
This module is addressed to Kinetics Toolkit's developers only.

"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import kineticstoolkit.config

import os
import subprocess
import shutil
import webbrowser
import doctest


def run_unit_tests() -> None:  # pragma: no cover
    """Run all unit tests."""
    # Run pytest in another process to ensure that the workspace is and stays
    # clean, and all Matplotlib windows are closed correctly after the tests.
    print('Running unit tests...')

    cwd = os.getcwd()
    os.chdir(kineticstoolkit.config.root_folder + '/tests')
    subprocess.call(['coverage', 'run',
                     '--source', '../kineticstoolkit',
                     '--omit', '../kineticstoolkit/external/*',
                     '-m', 'pytest', '--ignore=interactive'],
                    env=kineticstoolkit.config.env)
    subprocess.call(['coverage', 'html'], env=kineticstoolkit.config.env)
    webbrowser.open_new_tab(
        'file://' + kineticstoolkit.config.root_folder +
        '/tests/htmlcov/index.html')
    os.chdir(cwd)


def run_style_formatter() -> None:  # pragma: no cover
    """Run style formatter (autopep8)."""
    print("Running autopep8...")
    subprocess.call(['autopep8', '-r', '-i',
                     kineticstoolkit.config.root_folder],
                    env=kineticstoolkit.config.env)


def run_static_type_checker() -> None:  # pragma: no cover
    """Run static typing checker (mypy)."""
    # Run pytest in another process to ensure that the workspace is and stays
    # clean, and all Matplotlib windows are closed correctly after the tests.
    print("Running mypy...")
    cwd = os.getcwd()
    os.chdir(kineticstoolkit.config.root_folder)
    subprocess.call(['mypy', '--config-file', 'kineticstoolkit/mypy.ini',
                     '-p', 'kineticstoolkit'],
                    env=kineticstoolkit.config.env)
    os.chdir(cwd)


def run_doc_tests() -> None:  # pragma: no cover
    """Run all doc tests."""
    print("Running doc tests...")
    cwd = os.getcwd()
    os.chdir(kineticstoolkit.config.root_folder + '/kineticstoolkit')
    for file in os.listdir():
        if file.endswith('.py'):
            try:
                module = eval('kineticstoolkit.' + file.split('.py')[0])
                doctest.testmod(module,
                                optionflags=doctest.NORMALIZE_WHITESPACE)
                print(f"Doctests passed in file {file}.")
            except Exception:
                print(f"Could not run the doctest in file {file}.")
    os.chdir(cwd)


def compile_for_pypi() -> None:  # pragma: no cover
    """Compile for PyPi."""
    shutil.rmtree(kineticstoolkit.config.root_folder + '/dist',
                  ignore_errors=True)
    shutil.rmtree(kineticstoolkit.config.root_folder + '/build',
                  ignore_errors=True)
    os.chdir(kineticstoolkit.config.root_folder)
    subprocess.call(['python', 'setup.py', 'sdist', 'bdist_wheel'],
                    env=kineticstoolkit.config.env)


def upload_to_pypi() -> None:  # pragma: no cover
    """Upload to PyPi. Only works on macOS for now."""
    root_folder = kineticstoolkit.config.root_folder
    subprocess.call([
        'osascript',
        '-e',
        'tell application "Terminal" to do script '
        f'"conda activate ktk; cd {root_folder}; twine upload dist/*"'],
        env=kineticstoolkit.config.env)


def run_tests() -> None:  # pragma: no cover
    """Run all testing and building functions."""
    run_style_formatter()
    run_doc_tests()
    run_static_type_checker()
    run_unit_tests()
