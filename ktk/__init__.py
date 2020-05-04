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
Kinetics Toolkit
================

Kinetics Toolkit (ktk) is an in-house biomechanical library developed by
Professor Félix Chénier at Université du Québec à Montréal.

There are two versions of ktk: a private version that is exclusively used in
my lab, and a public version that is distributed on PyPI. I usually wait
several months before releasing to the public, to ensure the modules are
stable and the API is mature enough to be shared.

The public version API is mostly stable. Although some methods and functions will be
added in the future, I do not expect to remove or rename much stuff. However please keep
in mind that this is experimental software. If you are using ktk, you are warmly invited
to contact me, first to say Hello, and so that I can warn you before doing major,
possibly breaking changes.

If you are interesting in collaborating either in research or software
development, please contact me at chenier.felix@uqam.ca

Laboratory website: https://felixchenier.uqam.ca

Kinetics Toolkit (ktk) website: https://felixchenier.uqam.ca/kineticstoolkit

Public version
--------------

[Tutorials](https://felixchenier.uqam.ca/ktk_dist/tutorials)

[API documentation](https://felixchenier.uqam.ca/ktk_dist/api)

Private unstable version
------------------------

[Tutorials](https://felixchenier.uqam.ca/ktk_lab/tutorials)

[API documentation](https://felixchenier.uqam.ca/ktk_lab/api)

"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"

import os as _os
import platform as _platform
import matplotlib as _mpl
import numpy as _np

__pdoc__ = {'dev': False}

# ---------------------------
# Set ktk configuration
# ---------------------------
# Root folder (ktk installation)
config = dict()
config['RootFolder'] = _os.path.dirname(_os.path.dirname(__file__))

# Operating system
config['IsPC'] = True if _platform.system() == 'Windows' else False
config['IsMac'] = True if _platform.system() == 'Darwin' else False
config['IsLinux'] = True if _platform.system() == 'Linux' else False

# ---------------------------
# Imports
# ---------------------------

from ktk._timeseries import TimeSeries, TimeSeriesEvent
from ktk._tools import explore, terminal

try:
    from ktk import dev
except Exception:
    pass
