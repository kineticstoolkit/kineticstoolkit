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
Module that provides user and auto-generated configuration for ktk.

Please edit this file to configure ktk.

In IPython or Spyder:

        >>> import ktk.config
        >>> edit ktk.config

Then restart the IPython kernel to apply these changes.

"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import os as _os
import platform as _platform


# --- User-configurable options
change_ipython_dict_repr = True  # Default is True
change_matplotlib_defaults = True  # Default is True
change_numpy_print_options = True  # Default is True


# --- Automatic configuration

# Root folder (ktk installation)
root_folder = _os.path.dirname(_os.path.dirname(__file__))

# Operating system
is_pc = True if _platform.system() == 'Windows' else False
is_mac = True if _platform.system() == 'Darwin' else False
is_linux = True if _platform.system() == 'Linux' else False
