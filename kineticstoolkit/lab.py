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
Kinetics Toolkit - Lab mode
===========================

This module loads Kinetics Toolkit in lab mode. The standard way to use this
module is:

    import kineticstoolkit.lab as ktk

To get started, please consult Kinetics Toolkit's
[website](https://felixchenier.uqam.ca/kineticstoolkit)

"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"

from kineticstoolkit import *

# Import also some hidden functions
from kineticstoolkit import __dir__
import warnings

start_lab_mode()
