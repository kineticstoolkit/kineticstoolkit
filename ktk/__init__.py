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
import platform as _platform

# ---------------------------
# Practical private constants
# ---------------------------

# Root folder (KTK installation)
_ROOT_FOLDER = _os.path.dirname(_os.path.dirname(__file__))

# Operating system
if _platform.system() == 'Windows':
    _ISPC = True
else:
    _ISPC = False

if _platform.system() == 'Darwin':
    _ISMAC = True
else:
    _ISMAC = False
    
if _platform.system() == 'Linux':
    _ISLINUX = True
else:
    _ISLINUX = False

# ---------------------------
# KTK Imports
# ---------------------------

from ktk.timeseries import TimeSeries, TimeSeriesEvent
import ktk.loadsave as loadsave
import ktk.gui as gui
import ktk.pushrimkinetics as pushrimkinetics
import ktk.dev.dev as dev
import ktk._repr as _repr
import ktk.dbinterface as dbinterface
