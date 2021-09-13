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


def update_version() -> None:  # pragma: no cover
    """Update VERSION based on the current branch name."""
    # Get the current branch name
    cwd = os.getcwd()
    os.chdir(kineticstoolkit.config.root_folder)
    branch_name = subprocess.check_output(
        ['git', 'branch', '--show-current']).decode()
    if 'stable' in branch_name:
        shutil.copy(
            kineticstoolkit.config.root_folder
            + '/kineticstoolkit/STABLE_VERSION',
            kineticstoolkit.config.root_folder
            + '/kineticstoolkit/VERSION')
    else:
        shutil.copy(
            kineticstoolkit.config.root_folder
            + '/kineticstoolkit/DEVELOPMENT_VERSION',
            kineticstoolkit.config.root_folder
            + '/kineticstoolkit/VERSION')

    # Update version in config module
    with open(
            kineticstoolkit.config.root_folder
            + '/kineticstoolkit/VERSION', 'r') as fid:
        kineticstoolkit.config.version = fid.read()

    os.chdir(cwd)


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


def _generate_tutorial(file: str) -> None:  # pragma: no cover
    """Generate one tutorial."""
    print(file)
    subprocess.call(['jupyter-nbconvert', '--execute',
                     '--log-level', 'WARN', '--inplace',
                     '--to', 'notebook', file],
                    env=kineticstoolkit.config.env)

    # Remove execution metadata (which changes after each run and pollutes
    # the git repository)
    with open(file, 'r') as fid:
        contents = json.load(fid)

    for cell in contents['cells']:
        cell['metadata'].pop('execution', None)

    with open(file, 'w') as fid:
        json.dump(contents, fid, indent=1)


def build_tutorials() -> None:  # pragma: no cover
    """Build the markdown from notebooks tutorials."""
    print('Generating tutorials...')
    now = time.time()
    cwd = os.getcwd()

    # Run notebooks to generate the tutorials
#    os.chdir(kineticstoolkit.config.root_folder + '/doc')

    files = []

    for (dirpath, dirnames, filenames) in os.walk(
            kineticstoolkit.config.root_folder + '/doc'):

        parent_dir = os.path.basename(dirpath)

        if '_build' not in dirpath and not parent_dir.startswith('.'):

            for file in filenames:

                if file.endswith('.ipynb'):
                    files.append(dirpath + '/' + file)

    with multiprocessing.Pool() as pool:
        pool.map(_generate_tutorial, files)

#    os.chdir(cwd)
    print(f'Done in {time.time() - now} seconds.')


def build_website(clean: bool = False) -> None:  # pragma: no cover
    """
    Build the website using sphinx.

    Set clean to True to `make clean` beforehand.

    """
    cwd = os.getcwd()
    os.chdir(kineticstoolkit.config.root_folder + '/doc')

    if clean:
        shutil.rmtree(kineticstoolkit.config.root_folder + '/doc/api',
                      ignore_errors=True)
        subprocess.call(['make', 'clean'],
                        env=kineticstoolkit.config.env)

    # Generate site
    print('Generating site...')
    subprocess.call(['make', 'html'], env=kineticstoolkit.config.env)

    # Move site to documentation repository
    if kineticstoolkit.config.version == 'master':
        doc_folder = (
            kineticstoolkit.config.root_folder
            + '/../kineticstoolkit_doc/master'
        )
    else:
        doc_folder = (
            kineticstoolkit.config.root_folder
            + '/../kineticstoolkit_doc/stable'
        )

    try:
        shutil.rmtree(doc_folder)
    except Exception:
        pass

    os.makedirs(doc_folder, exist_ok=True)
    os.rename(
        kineticstoolkit.config.root_folder + '/doc/_build/html',
        doc_folder,
    )

    # Open site
    webbrowser.open_new_tab(
        'file://' + doc_folder + '/index.html'
    )
    os.chdir(cwd)


def clean() -> None:  # pragma: no cover
    """Remove outputs in jupyter notebooks (before committing to git)."""
    for (folder, _, files) in os.walk(
            kineticstoolkit.config.root_folder + '/doc'):

        if '/_build' in folder or '/.ipynb_checkpoints' in folder:
            for file in files:
                if '.ipynb' in file:
                    os.remove(f"{folder}/{file}")

        elif any(['.ipynb' in file for file in files]):
            print(f"Cleaning folder {folder}")
            subprocess.call(['jupyter-nbconvert',
                             '--ClearOutputPreprocessor.enabled=True',
                             '--log-level', 'WARN',
                             '--inplace',
                             f'{folder}/*.ipynb'],
                            env=kineticstoolkit.config.env)


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


def build() -> None:  # pragma: no cover
    """Run all testing and building functions."""
    update_version()
    run_style_formatter()
    run_doc_tests()
    run_static_type_checker()
    run_unit_tests()
    build_tutorials()
    build_website(True)
