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

Most ot ktk is closed source for now. I usually wait several months before
releasing to ensure the modules are stable and mature enough to be shared.

If you are interesting in collaborating either in research or software
development, please contact me at chenier.felix@uqam.ca

Project website: https://felixchenier.com/kineticstoolkit

"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"

import os as _os
import platform as _platform
import matplotlib as _mpl
import numpy as _np


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
from ktk._dbinterface import DBInterface
from ktk._player import Player
from ktk._loadsave import load, loadmat, save
from ktk import filters
from ktk import gui
from ktk import mplhelper
from ktk._tools import explore, terminal, update, tutorials
from ktk import geometry
from ktk import kinematics
from ktk import pushrimkinetics
from ktk import inversedynamics
from ktk import dev
from ktk import cycles
from ktk import _repr

# ---------------------------
# Customizations
# ---------------------------

# Modify the repr function for dicts in IPython
try:
    import IPython as _IPython
    _ip = _IPython.get_ipython()
    formatter = _ip.display_formatter.formatters['text/plain']
    formatter.for_type(dict, lambda n, p, cycle:
                       _repr._ktk_format_dict(n, p, cycle))
except Exception:
    pass

# Set alternative defaults to matplotlib
_mpl.rcParams['figure.figsize'] = [10, 5]
_mpl.rcParams['figure.dpi'] = 75
_mpl.rcParams['lines.linewidth'] = 1

# Set a custom color order that is compatible with 'char' colors, and that
# begins with RGB so that it is compatible with most XYZ color orders in other
# visualization softwares.
gui.set_color_order('xyz')

# Select default mode for numpy
_np.set_printoptions(suppress=True)
