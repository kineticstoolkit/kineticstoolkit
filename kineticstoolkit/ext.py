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
"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2022 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import warnings
import importlib
import pkgutil
import sys
import os


def _load_module(name, verbose):
    """Load one module by name."""
    if "kineticstoolkit_" in name:
        short_name = name[len('kineticstoolkit_'):]
        try:
            if verbose:
                print(f"Loaded {name} extension.")
            globals()[short_name] = importlib.import_module(name)
            my_dir.append(short_name)
        except Exception:
            warnings.warn(
                f"There have been an error loading the {name} extension."
            )


def load_extensions(folder: str = '', verbose: bool = False):
    """
    Load all extensions found on PYTHONPATH.

    Any module that begins by 'kineticstoolkit_' and that is on PYTHONPATH will
    be found and imported as an extension. The simplest case would be a python
    file named `kineticstoolkit_myextension.py` that sits in the current
    folder, and that defines a function named `process_data`.

        ktk.ext.load_extensions()

    would import this file and its content would be accessible via

        ktk.ext.myextension

    for example:

        ktk.ext.myextension.process_data()

    Parameters
    ----------
    folder
        Optional. Name of the folder to scan for extensions. Default is
        the whole PYTHONPATH.

    Returns
    -------
    None.

    """
    # Dynamically import extensions

    # Clear my_dir and start over
    del my_dir[:]
    my_dir.append('load_extensions')

    if folder == "":
        # Scan PYTHONPATH
        for finder, name, ispkg in pkgutil.iter_modules(sys.path):
            _load_module(name, verbose)
    else:
        # Scan folder
        for name in os.listdir(folder):
            _load_module(name, verbose)


my_dir = ['load_extensions']


def __dir__():
    return my_dir
