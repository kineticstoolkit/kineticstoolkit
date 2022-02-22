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
[website](https://kineticstoolkit.uqam.ca)

>>> import kineticstoolkit as ktk

"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import os


listing = []
unstable_listing = []


# --- Import released modules and functions
from kineticstoolkit.timeseries import TimeSeries, TimeSeriesEvent  # noqa
listing.append('TimeSeries')
listing.append('TimeSeriesEvent')

from kineticstoolkit.tools import tutorials  # noqa
listing.append('tutorials')

from kineticstoolkit.tools import start_lab_mode  # noqa
listing.append('start_lab_mode')

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

from kineticstoolkit import doc  # noqa
listing.append('doc')

from kineticstoolkit import _repr  # noqa
from kineticstoolkit import gui  # noqa

from kineticstoolkit import geometry  # noqa
listing.append('geometry')


# Load unstable and dev modules (but do not add those to the __dir__ listing)
from kineticstoolkit import dev  # noqa
unstable_listing.append('dev')

from kineticstoolkit import inversedynamics  # noqa
unstable_listing.append('inversedynamics')

from kineticstoolkit import emg  # noqa
unstable_listing.append('emg')

from kineticstoolkit import anthropometrics  # noqa
unstable_listing.append('anthropometrics')

from kineticstoolkit import config  # noqa


def enable_dev():
    """
    Enable development functions and unstable functions and modules.

    This function is exclusively reserved for Kinetics Toolkit development.
    It is autatically called if there is a file called "KTK_AUTO_ENABLE_DEV"
    in the current folder when importing Kinetics Toolkit.

    There is no disable_dev function that reverts to stable version. To this
    effect, one must relaunch the Python interpreter and reload Kinetics
    Toolkit.

    """
    config.version = 'master'


if os.path.exists('KTK_AUTO_ENABLE_DEV'):
    enable_dev()


def __dir__():
    if config.version == 'master':
        return listing + unstable_listing
    else:
        return listing


if __name__ == "__main__":  # pragma: no cover
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
