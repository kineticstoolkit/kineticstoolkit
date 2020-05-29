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
Contains fonctions related to development, tests and release of ktk.

"""

import ktk.config

import os
import subprocess
import shutil
import webbrowser
import doctest


def run_unit_tests():
    """Run all unit tests."""
    # Run pytest in another process to ensure that the workspace is and stays
    # clean, and all Matplotlib windows are closed correctly after the tests.
    print('Running unit tests...')
    cwd = os.getcwd()
    os.chdir(ktk.config.root_folder + '/tests')
    subprocess.call(['pytest', '--ignore=interactive'])
    os.chdir(cwd)


def run_doc_tests():
    """Run all doc tests."""
    print('Running doc tests...')
    cwd = os.getcwd()
    os.chdir(ktk.config.root_folder + '/tests')
    for file in os.listdir():
        if file.endswith('.py'):
            print(file)
            try:
                module = eval('ktk.' + file.split('.py')[0])
                doctest.testmod(module)
            except Exception:
                pass
    os.chdir(cwd)


def generate_tutorials():
    """Generate the html tutorials."""
    cwd = os.getcwd()
    os.chdir(ktk.config.root_folder + '/tutorials')
    subprocess.call(['pwd'])
    subprocess.call(['jupyter-nbconvert', '--execute', '--to', 'html_toc',
                     'index.ipynb'])
    os.chdir(cwd)

    # Open tutorial
    webbrowser.open_new_tab(
        'file://' + ktk.config.root_folder + '/tutorials/index.html')


def generate_doc():
    """Generate ktk's reference using pdoc."""
    cwd = os.getcwd()

    # Run pdoc
    try:
        os.mkdir(ktk.config.root_folder + '/doc/')
    except Exception:
        pass

    try:
        shutil.rmtree(ktk.config.root_folder + '/doc/ktk')
    except Exception:
        pass

    os.chdir(ktk.config.root_folder + '/ktk')
    subprocess.call(['pdoc', '--html', '--config', 'show_source_code=False',
                     '--output-dir', ktk.config.root_folder + '/doc', 'ktk'])

    # Cleanup
    os.chdir(cwd)

    # Open doc
    webbrowser.open_new_tab(
        'file://' + ktk.config.root_folder + '/doc/ktk/index.html')


def update_readme():
    """Copy ktk's docstring into readme.md (up to the -------- separator)."""
    from ktk import __doc__ as ktkdoc
    # I only load it here because I don't want the whole ktk suite to be loaded
    # automatically, which would be the case if dev loaded it.
    with open(ktk.config.root_folder + '/README.md', 'w') as fid:
        fid.write(ktkdoc.split(
            '------------------------------------------------------------')[0])


def compile_for_pypi():
    """Compile for PyPi."""
    update_readme()
    shutil.rmtree(ktk.config.root_folder + '/dist', ignore_errors=True)
    shutil.rmtree(ktk.config.root_folder + '/build', ignore_errors=True)
    os.chdir(ktk.config.root_folder)
    subprocess.call(['python', 'setup.py', 'sdist', 'bdist_wheel'])


def upload_to_pypi():
    """Upload to PyPi."""
    root_folder = ktk.config.root_folder
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
