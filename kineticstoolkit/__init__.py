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

To get started, please consult Kinetics Toolkit's
[website](https://felixchenier.uqam.ca/kineticstoolkit)

"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


listing = []
import kineticstoolkit.config

# --- Import released modules and functions
from kineticstoolkit.timeseries import TimeSeries, TimeSeriesEvent
listing.append('TimeSeries')
listing.append('TimeSeriesEvent')

from kineticstoolkit.tools import explore, terminal, update, tutorials
listing.append('explore')
listing.append('terminal')
listing.append('update')
listing.append('tutorials')

from kineticstoolkit.tools import start_lab_mode  # Keep this one hidden.

from kineticstoolkit.player import Player
listing.append('Player')

from kineticstoolkit.loadsave import load, save
listing.append('load')
listing.append('save')

from kineticstoolkit import filters
listing.append('filters')

from kineticstoolkit import kinematics
listing.append('kinematics')

from kineticstoolkit import pushrimkinetics
listing.append('pushrimkinetics')

from kineticstoolkit import cycles
listing.append('cycles')

from kineticstoolkit import _repr
from kineticstoolkit import gui

from kineticstoolkit import config

# --- Import unstable modules but append to listing only if we are on master
from kineticstoolkit.dbinterface import DBInterface
from kineticstoolkit import geometry
from kineticstoolkit import inversedynamics
from kineticstoolkit import emg
from kineticstoolkit import dev

if config.version == 'master':
    listing.append('DBInterface')
    listing.append('geometry')
    listing.append('inversedynamics')
    listing.append('emg')
    listing.append('dev')

def __dir__():
    return listing
