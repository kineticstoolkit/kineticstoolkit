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

>>> import kineticstoolkit as ktk

"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


listing = []
import kineticstoolkit.config  # noqa

# --- Import released modules and functions
from kineticstoolkit.timeseries import TimeSeries, TimeSeriesEvent  # noqa
listing.append('TimeSeries')
listing.append('TimeSeriesEvent')

from kineticstoolkit.tools import tutorials  # noqa
listing.append('tutorials')

from kineticstoolkit.tools import start_lab_mode  # noqa

from kineticstoolkit.player import Player  # noqa
listing.append('Player')

from kineticstoolkit.loadsave import load, save  # noqa
listing.append('load')
listing.append('save')

from kineticstoolkit import filters  # noqa
listing.append('filters')

from kineticstoolkit import kinematics  # noqa
listing.append('kinematics')

from kineticstoolkit import pushrimkinetics  # noqa
listing.append('pushrimkinetics')

from kineticstoolkit import cycles  # noqa
listing.append('cycles')

from kineticstoolkit import _repr  # noqa
from kineticstoolkit import gui  # noqa

from kineticstoolkit import geometry  # noqa
listing.append('geometry')

from kineticstoolkit import config  # noqa

# --- Import unstable modules but append to listing only if we are on master

from kineticstoolkit import inversedynamics  # noqa
from kineticstoolkit import emg  # noqa
from kineticstoolkit import anthropometrics  # noqa

try:
    from kineticstoolkit import dev  # noqa
except:
    pass

if config.version == 'master':
    listing.append('dev')
    listing.append('inversedynamics')
    listing.append('emg')
    listing.append('anthropometrics')


def __dir__():
    return listing


if __name__ == "__main__":  # pragma: no cover
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
