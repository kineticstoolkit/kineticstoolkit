"""
KTK Kinesiology Toolkit
=======================

This is a skeleton for an all new Kinesiology toolkit based on Python instead
of Matlab, so that it's free of Matlab limitations and more distributable and
usable on onboard instruments.

Methods:

    [ktk.TimeSeries][]

Author: Félix Chénier

Date: Started on July 2019
"""
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

from .timeseries import TimeSeries, TimeSeriesEvent
from . import filters
from . import loadsave
from . import gui
from . import pushrimkinetics
from .dev import dev
from . import dbinterface
from . import _repr


try:
    import IPython
    ip = IPython.get_ipython()
    formatter = ip.display_formatter.formatters['text/plain']
    formatter.for_type(dict, lambda n, p, cycle: _repr._ktk_format_dict(n, p, cycle))
except:
    pass

