"""
Kinesiology Toolkit.

KTK Kinesiology Toolkit
=======================

This is a skeleton for an all new Kinesiology toolkit based on Python instead
of Matlab, so that it's free of Matlab limitations and more distributable and
usable on onboard instruments.

Author: Félix Chénier

Date: Started on July 2019
"""

name = 'ktk'

import os as _os

# Practical private constants
_ROOT_FOLDER = _os.path.dirname(__file__)


from ktk.timeseries import TimeSeries, TimeSeriesEvent
import ktk.loadsave as loadsave
import ktk.gui as gui
import ktk.pushrimkinetics as pushrimkinetics
import ktk.dev.dev as dev
import ktk._repr as _repr
import ktk.dbinterface as dbinterface
