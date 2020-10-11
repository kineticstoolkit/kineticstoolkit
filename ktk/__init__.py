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

from ktk.loadsave import load, save
directory.append('load')
directory.append('save')

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
