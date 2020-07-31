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
Provide user and auto-generated configuration for ktk.

Please edit this file to configure ktk.

In IPython or Spyder:

    import ktk.config
    edit ktk.config

Then restart the IPython kernel to apply these changes.

"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import os
import warnings
import platform


# --- User-configurable options
change_ipython_dict_repr = True  # Default is True
change_matplotlib_defaults = True  # Default is True
change_numpy_print_options = True  # Default is True


# --- Automatic configuration

# Root folder (ktk installation)
root_folder = os.path.dirname(os.path.dirname(__file__))

# Home folder
home_folder = os.path.expanduser("~")

# KTK version
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
    temp_folder = _base_temp_folder + '/ktk'
    try:
        os.mkdir(temp_folder)
    except FileExistsError:
        pass
except Exception:
    warnings.warn('Could not set temporary folder.')
    temp_folder = '.'
