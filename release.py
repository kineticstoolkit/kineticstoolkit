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
release.py
----------

These little functions facilitates releasing ktk on PyPi.

"""

import ktk
import subprocess
import os

def update_readme():
    """Copy ktk's docstring into readme.md."""
    with open('README.md', 'w') as fid:
        fid.write(ktk.__doc__)

def update_tutorial():
    """Recompile the tutorial."""
    os.chdir(ktk.config['RootFolder'] + '/tutorials')
    subprocess.call(['jupyter-nbconvert',
                     '--to', 'html_toc', 'tutorial.ipynb'])

def compile_for_pypi():
    """Compile for PyPi."""
    os.chdir(ktk.config['RootFolder'])
    subprocess.call(['rm', '-rR', root_folder + '/dist'])
    subprocess.call(['rm', '-rR', root_folder + '/build'])
    subprocess.call(['python', 'setup.py', 'sdist', 'bdist_wheel'])

def upload_to_pypi():
    """Upload to PyPi's test server."""
    os.chdir(ktk.config['RootFolder'])
    subprocess.call([
            'osascript',
            '-e',
            'tell application "Terminal" to do script '
            f'"conda activate ktk; twine upload dist/*"'])

#%%

update_readme()
update_tutorial()
compile_for_pypi()

#%%
upload_to_pypi()
