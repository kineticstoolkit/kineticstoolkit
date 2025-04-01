#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020-2025 Félix Chénier

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Provide configuration values for Kinetics Toolkit's internal workings."""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2025 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import os
import warnings
import platform


def __dir__() -> list[str]:
    return [
        "root_folder",
        "home_folder",
        "is_pc",
        "is_mac",
        "is_linux",
        "temp_folder",
        "version",
        "pythonpath",
        "interactive_backend_warning",
    ]


# Root folder (kineticstoolkit installation)
root_folder = os.path.dirname(os.path.dirname(__file__))

# Home folder
home_folder = os.path.expanduser("~")

# Kinetics Toolkit version.
with open(root_folder + "/kineticstoolkit/VERSION", "r") as fid:
    version = fid.read()

# Operating system
is_pc = True if platform.system() == "Windows" else False
is_mac = True if platform.system() == "Darwin" else False
is_linux = True if platform.system() == "Linux" else False


# Temporary folder
def _try_folder(folder: str) -> bool:
    """Try this folder for write access as a temporary folder."""
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass
    except Exception:
        return False

    return True


try_list = []

if is_pc and "TEMP" in os.environ:
    try_list.append(os.environ["TEMP"] + "/kineticstoolkit")
if is_mac and "TMPDIR" in os.environ:
    try_list.append(os.environ["TMPDIR"] + "/kineticstoolkit")
if "HOME" in os.environ:
    try_list.append(os.environ["HOME"] + "/.kineticstoolkit")
# Last try
try_list.append(".")

for temp_folder in try_list:
    if _try_folder(temp_folder):
        break

if temp_folder == ".":
    warnings.warn("Could not set temporary folder.")


# Environment, including Python path. If PYTHONPATH is defined in Spyder and
# Spyder is opened as a standalone app, define PYTHONPATH as SPY_PYTHONPATH.
env = os.environ.copy()
if "SPY_PYTHONPATH" in env and "PYTHONPATH" not in env:
    env["PYTHONPATH"] = env["SPY_PYTHONPATH"]

# Others
interactive_backend_warning = True
