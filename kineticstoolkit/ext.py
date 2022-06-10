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


def __getattr__(module_name):
    """Return an helpful message in case an extension was not loaded."""
    raise ModuleNotFoundError(
        f"The extension kineticstoolkit_{module_name} is not loaded. "
        f"You can load extensions using "
        f"`ktk.ext.load_extensions()`, or by "
        f"importing Kinetics Toolkit in lab mode "
        f"using `import kineticstoolkit.lab as ktk`. If this error still "
        f"happens, make sure that kineticstoolkit_{module_name} is correctly "
        f"installed."
    )


def _load_module(name, verbose):
    """Load one module by name."""
    if "kineticstoolkit_" in name:
        short_name = name[len('kineticstoolkit_'):]
        try:
            if verbose:
                print(f"Loaded {name}.")
            globals()[short_name] = importlib.import_module(name)
            loaded_extensions.append(short_name)
        except Exception:
            warnings.warn(
                f"There have been an error loading the {name} extension. "
                f"Please try to import {name} manually to get more insights "
                f"on the cause of the error."
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

    # Clear loaded_extensions and start over
    del loaded_extensions[:]

    if folder == "":
        # Scan PYTHONPATH
        for finder, name, ispkg in pkgutil.iter_modules(sys.path):
            _load_module(name, verbose)
    else:
        # Scan folder
        for name in os.listdir(folder):
            _load_module(name, verbose)


loaded_extensions = ['load_extensions']


def __dir__():
    return ['load_extensions'] + loaded_extensions
