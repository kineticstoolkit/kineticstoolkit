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
    """Generate the markdown from notebooks tutorials."""
    cwd = os.getcwd()

    # Run notebooks to generate the tutorials
    os.chdir(ktk.config.root_folder + '/doc')
    for file in os.listdir():
        if file.endswith('.ipynb'):
            subprocess.call(['jupyter-nbconvert', '--execute',
                             '--to', 'markdown', file])
    os.chdir(cwd)


def generate_api():
    """Generate ktk's reference api using pdoc."""
    cwd = os.getcwd()

    # Run pdoc to generate the API documentation
    try:
        os.mkdir(ktk.config.root_folder + '/doc/api')
    except Exception:
        pass

    try:
        shutil.rmtree(ktk.config.root_folder + '/doc/api/ktk')
    except Exception:
        pass

    os.chdir(ktk.config.root_folder + '/ktk')
    subprocess.call(['pdoc', '--html', '--config', 'show_source_code=False',
                     '--output-dir',
                     ktk.config.root_folder + '/doc/api', 'ktk'])

    # Cleanup
    os.chdir(cwd)


def generate_site():
    """Build the website using mkdocs."""
    cwd = os.getcwd()
    os.chdir(ktk.config.root_folder)
    subprocess.call(['mkdocs', 'build'])
    # Open doc
    webbrowser.open_new_tab(
        'file://' + ktk.config.root_folder + '/site/index.html')
    os.chdir(cwd)


def clean():
    """Delete temporary files that were used by the release process."""
    cwd = os.getcwd()

    # Clean /doc folder
    os.chdir(ktk.config.root_folder + '/doc')

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

    # Clean /site folder
    os.chdir(ktk.config.root_folder + '/site')

    files = os.listdir()
    for file in files:

        # Remove all source files
        if file.endswith('.md') or file.endswith('.ipynb'):
            print(f'Removing site/{file}.')
            os.remove(file)

        if file == '__pycache__':
            print(f'Removing doc/{file}.')
            shutil.rmtree(file)

    os.chdir(cwd)


def compile_for_pypi():
    """Compile for PyPi."""
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
    generate_api()
    generate_tutorials()
    generate_site()
    clean()
