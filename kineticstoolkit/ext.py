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
from __future__ import annotations


__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2022 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import warnings
import importlib
import pkgutil


def _import_extensions() -> list[str]:
    """
    Import all installed kineticstoolkit extensions.

    Any module that begins by ``kineticstoolkit_`` and that is on PYTHONPATH
    will be imported in the ``kineticstoolkit.ext`` namespace.

    Parameters
    ----------
    None

    Returns
    -------
    A list of the imported extension names.

    Warning
    -------
    This function, which has been introduced in 0.8, is still experimental and
    may change signature or behaviour in the future.

    Notes
    ----
    This function is called automatically if Kinetics Toolkit is imported in
    lab mode::

        import kineticstoolkit.lab as ktk

    If your extension is not imported, try importing it manually before
    reporting a bug::

       import kineticstoolkit_EXTENSION_NAME

    If your extension doesn't import, you need to edit your PYTHONPATH or to
    check that the extension is correctly installed.

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


imported_extensions = set()  # type: set[str]


def __dir__():
    return list(imported_extensions)


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
