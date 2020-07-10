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
Kinetics Toolkit - Development version
======================================

To get started, please consult ktk's
[website](https://felixchenier.uqam.ca/ktk_develop)

"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


__pdoc__ = {'dev': False, 'cmdgui': False, 'gui': False, 'external': False,
            'mplhelper': False}

# --- Imports
directory = []

from ktk.timeseries import TimeSeries, TimeSeriesEvent
directory.append('TimeSeries')
directory.append('TimeSeriesEvent')

from ktk.tools import explore, terminal, update, tutorials
directory.append('explore')
directory.append('terminal')
directory.append('update')
directory.append('tutorials')

from ktk.dbinterface import DBInterface
directory.append('DBInterface')

from ktk.player import Player
directory.append('Player')

from ktk.loadsave import load, loadmat, save
directory.append('load')
directory.append('loadmat')
directory.append('save')

from ktk import config
directory.append('config')

from ktk import filters
directory.append('filters')

from ktk import geometry
directory.append('geometry')

from ktk import kinematics
directory.append('kinematics')

from ktk import pushrimkinetics
directory.append('pushrimkinetics')

from ktk import inversedynamics
directory.append('inversedynamics')

from ktk import cycles
directory.append('cycles')

from ktk import emg
directory.append('emg')

from ktk import _repr
from ktk import gui

try:
    from ktk import dev
    directory.append('dev')
except Exception:
    pass


def __dir__():
    return directory

# --- Customizations

if config.change_ipython_dict_repr is True:
    # Modify the repr function for dicts in IPython
    try:
        import IPython as _IPython
        _ip = _IPython.get_ipython()
        formatter = _ip.display_formatter.formatters['text/plain']
        formatter.for_type(dict, lambda n, p, cycle:
                           _repr._ktk_format_dict(n, p, cycle))
    except Exception:
        pass

if config.change_matplotlib_defaults is True:
    # Set alternative defaults to matplotlib
    import matplotlib as _mpl
    _mpl.rcParams['figure.figsize'] = [10, 5]
    _mpl.rcParams['figure.dpi'] = 75
    _mpl.rcParams['lines.linewidth'] = 1
    gui.set_color_order('xyz')

if config.change_numpy_print_options is True:
    import numpy as _np
    # Select default mode for numpy
    _np.set_printoptions(suppress=True)
