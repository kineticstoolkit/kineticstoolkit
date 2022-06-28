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
import os
from typing import List, Optional, Set, Sequence


# def __getattr__(module_name):
#     """Return an helpful message in case an extension was not loaded."""
#     raise ModuleNotFoundError(
#         f"The extension kineticstoolkit_{module_name} is not loaded. "
#         f"You can load extensions using "
#         f"`ktk.ext.load_extensions()`, or by "
#         f"importing Kinetics Toolkit in lab mode "
#         f"using `import kineticstoolkit.lab as ktk`. If this error still "
#         f"happens, make sure that kineticstoolkit_{module_name} is correctly "
#         f"installed."
#     )


def _import_extensions() -> List[str]:
    """
    Import all installed kineticstoolkit extensions.

    Any module that begins by 'kineticstoolkit_' and that is on PYTHONPATH will
    be imported in the kineticstoolkit.ext namespace.

    Returns
    -------
    A list of the imported extension names.

    To test an extension in development that is not yet installed, start by
    adding the extension's folder to the PYTHONPATH. This folder is the outer
    one: it contains the folder than contains the __init__.py file. For
    example, for an extension named kineticstoolkit_awesomestuff located in
    the folder extension_folder:

        import kineticstoolkit as ktk
        import sys
        sys.path.append('extension_folder')
        ktk.import_extensions()

    """
    for finder, name, ispkg in pkgutil.iter_modules():
        if name.startswith("kineticstoolkit_"):
            short_name = name[len("kineticstoolkit_") :]
            try:
                globals()[short_name] = importlib.import_module(name)
                imported_extensions.add(short_name)
            except Exception:
                warnings.warn(
                    f"There have been an error importing the {name} "
                    f"extension. Please try to import {name} manually to get "
                    f"more insights on the cause of the error."
                )

    return list(imported_extensions)


imported_extensions = set()  # type: Set[str]


def __dir__():
    return imported_extensions


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
