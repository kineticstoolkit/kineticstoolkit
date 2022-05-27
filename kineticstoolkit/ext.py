#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2022 Félix Chénier

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
Provide extension support for Kinetics Toolkit.

Any module that begins by 'kineticstoolkit_' and that is on PYTHONPATH will
be found and imported as an extension. For example, when the future
kineticstoolkit_opensim module will be installed, it will automatically
become available using:

    # Import Kinetics Toolkit
    import kineticstoolkit as ktk

    # Access a function of kineticstoolkit_opensim
    ktk.ext.opensim."FUNCTION_NAME"

"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2022 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import warnings
import importlib
import pkgutil
import sys

my_dir = []

# Dynamically import extensions
for finder, name, ispkg in pkgutil.iter_modules(sys.path):

    if name.startswith('kineticstoolkit_'):
        try:
            locals()[name] = importlib.import_module(name)
            my_dir.append(name[len('kineticstoolkit_'):])
        except Exception:
            warnings.warn(
                f"There have been an error loading the {name} extension."
            )


def __dir__():
    return my_dir
