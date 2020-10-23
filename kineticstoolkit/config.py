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
Provide configuration values for Kinetics Toolkit's inner working.

"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import os
import warnings
import platform
from typing import List


def __dir__() -> List[str]:
    return [
        'root_folder',
        'home_folder',
        'is_pc',
        'is_mac',
        'is_linux',
        'temp_folder',
        'version',
    ]

# Root folder (kineticstoolkit installation)
root_folder = os.path.dirname(os.path.dirname(__file__))

# Home folder
home_folder = os.path.expanduser("~")

# Kinetics Toolkit version
with open(root_folder + '/VERSION', 'r') as fid:
    version = fid.read()

# Operating system
is_pc = True if platform.system() == 'Windows' else False
is_mac = True if platform.system() == 'Darwin' else False
is_linux = True if platform.system() == 'Linux' else False

# Temporary folder
if is_pc:
    _base_temp_folder = os.environ['TEMP']
elif is_mac or is_linux:
    _base_temp_folder = os.environ['TMPDIR']

try:
    temp_folder = _base_temp_folder + '/kineticstoolkit'
    try:
        os.mkdir(temp_folder)
    except FileExistsError:
        pass
except Exception:
    warnings.warn('Could not set temporary folder.')
    temp_folder = '.'
