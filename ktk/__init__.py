"""
KTK Kinesiology Toolkit
=======================

This is a skeleton for an all new Kinesiology toolkit based on Python instead
of Matlab, so that it's free of Matlab limitations and more distributable and
usable on onboard instruments.

Author: Félix Chénier

Date: Started on July 2019
"""
import os as _os
import platform as _platform
import matplotlib as _mpl


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

from ._timeseries import TimeSeries, TimeSeriesEvent
from . import filters
from . import loadsave
from . import gui
from . import pushrimkinetics
from . import dev
from . import dbinterface
from . import _repr


# Modify the repr function for dicts in iPython
try:
    import IPython as _IPython
    ip = _IPython.get_ipython()
    formatter = ip.display_formatter.formatters['text/plain']
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
_mpl.rcParams['axes.prop_cycle'] = _mpl.cycler(
      'color', ['r', 'g', 'b', 'c', 'm', 'y', 'k'])
