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
import multiprocessing
import time
import json


def run_unit_tests() -> None:
    """Run all unit tests."""
    # Run pytest in another process to ensure that the workspace is and stays
    # clean, and all Matplotlib windows are closed correctly after the tests.
    print('Running unit tests...')
    cwd = os.getcwd()
    os.chdir(kineticstoolkit.config.root_folder + '/tests')
    subprocess.call(['pytest', '--ignore=interactive'])
    os.chdir(cwd)


def run_static_type_checker() -> None:
    """Run static typing checker (mypy)."""
    # Run pytest in another process to ensure that the workspace is and stays
    # clean, and all Matplotlib windows are closed correctly after the tests.
    print('Running mypy...')
    cwd = os.getcwd()
    os.chdir(kineticstoolkit.config.root_folder)
    subprocess.call(['mypy', '--ignore-missing-imports', '-p',
                     'kineticstoolkit'])
    os.chdir(cwd)


def run_doc_tests() -> None:
    """Run all doc tests."""
    print('Running doc tests...')
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
                print(f'Could not run the doctest in file {file}.')
    os.chdir(cwd)

def _generate_tutorial(file: str) -> None:
    """Generate one tutorial."""
    print(file)
    subprocess.call(['jupyter-nbconvert', '--execute',
                     '--log-level', 'WARN', '--inplace',
                     '--to', 'notebook', file])

    # Remove execution metadata (which changes after each run and pollutes
    # the git repository)
    with open(file, 'r') as fid:
        contents = json.load(fid)

    for cell in contents['cells']:
        cell['metadata'].pop('execution', None)

    with open(file, 'w') as fid:
        json.dump(contents, fid, indent=1)


def build_tutorials() -> None:
    """Build the markdown from notebooks tutorials."""
    print('Generating tutorials...')
    now = time.time()
    cwd = os.getcwd()

    # Run notebooks to generate the tutorials
    os.chdir(kineticstoolkit.config.root_folder + '/doc')
    files = [file for file in os.listdir() if file.endswith('.ipynb')]
    with multiprocessing.Pool() as pool:
        pool.map(_generate_tutorial, files)

    os.chdir(cwd)
    print(f'Done in {time.time() - now} seconds.')

def build_website(clean: bool = False) -> None:
    """
    Build the website using sphinx.

    Set clean to True to `make clean` beforehand.

    """
    cwd = os.getcwd()
    os.chdir(kineticstoolkit.config.root_folder + '/doc')

    if clean:
        subprocess.call(['make', 'clean'])

    # Generate API
    print('Generating API...')
    os.environ['SPHINX_APIDOC_OPTIONS'] = \
        'members,undoc-members,autosummary'
    subprocess.call(['sphinx-apidoc', '-q', '-e', '-f', '-d', '3',
                      '-o', 'api', '../kineticstoolkit', 'external',
                      'kineticstoolkit.cmdgui'])

    # Generate site
    print('Generating site...')
    subprocess.call(['make', 'html'])

    # Open site
    webbrowser.open_new_tab(
        'file://' + kineticstoolkit.config.root_folder + '/doc/_build/html/index.html')
    os.chdir(cwd)


def clean() -> None:
    """Delete temporary files that were used by the release process."""
    cwd = os.getcwd()

    # Clean /doc folder
    os.chdir(kineticstoolkit.config.root_folder + '/doc')

    files = os.listdir()
    for file in files:

        # Remove all .md files that have a corresponding .ipynb file (since
        # they were generated from the .ipynb).
        if (file.endswith('.md') and
            (file[0:-len('.md')] + '.ipynb') in files):
            print(f'Removing doc/{file}.')
            os.remove(file)

        # Remove _files folder, since they are required by the site's html
        # files but not by the source md and ipynb
        elif file.endswith('_files') or file == 'api':
            print(f'Removing doc/{file}.')
            shutil.rmtree(file)

    os.chdir(cwd)


def compile_for_pypi() -> None:
    """Compile for PyPi."""
    shutil.rmtree(kineticstoolkit.config.root_folder + '/dist',
                  ignore_errors=True)
    shutil.rmtree(kineticstoolkit.config.root_folder + '/build',
                  ignore_errors=True)
    os.chdir(kineticstoolkit.config.root_folder)
    subprocess.call(['python', 'setup.py', 'sdist', 'bdist_wheel'])


def upload_to_pypi() -> None:
    """Upload to PyPi. Only works on macOS for now."""
    root_folder = kineticstoolkit.config.root_folder
    subprocess.call([
        'osascript',
        '-e',
        'tell application "Terminal" to do script '
        f'"conda activate ktk; cd {root_folder}; twine upload dist/*"'])


def release() -> None:
    """Run all functions for release, without packaging and uploading."""
    run_doc_tests()
    run_static_type_checker()
    run_unit_tests()
    build_tutorials()
    build_website(True)
