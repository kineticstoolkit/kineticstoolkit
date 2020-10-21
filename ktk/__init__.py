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


listing = []
import ktk.config

# --- Import released modules and functions
from ktk.timeseries import TimeSeries, TimeSeriesEvent
listing.append('TimeSeries')
listing.append('TimeSeriesEvent')

from ktk.tools import explore, terminal, update, tutorials
listing.append('explore')
listing.append('terminal')
listing.append('update')
listing.append('tutorials')

from ktk.player import Player
listing.append('Player')

from ktk.loadsave import load, save
listing.append('load')
listing.append('save')

from ktk import filters
listing.append('filters')

from ktk import kinematics
listing.append('kinematics')

from ktk import pushrimkinetics
listing.append('pushrimkinetics')

from ktk import cycles
listing.append('cycles')

from ktk import _repr
from ktk import gui

# --- Import unstable modules if we are on master
if ktk.config.version == 'master':

    from ktk.dbinterface import DBInterface
    listing.append('DBInterface')

    from ktk import geometry
    listing.append('geometry')

    from ktk import inversedynamics
    listing.append('inversedynamics')

    from ktk import emg
    listing.append('emg')

try:
    from ktk import dev
    listing.append('dev')
except Exception:
    pass


def __dir__():
    return listing
