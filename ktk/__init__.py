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
# Set KTK configuration
# ---------------------------
# Root folder (KTK installation)
config = dict()
config['RootFolder'] = _os.path.dirname(_os.path.dirname(__file__))

# Operating system
config['IsPC'] = True if _platform.system() == 'Windows' else False
config['IsMac'] = True if _platform.system() == 'Darwin' else False
config['IsLinux'] = True if _platform.system() == 'Linux' else False

# ---------------------------
# KTK Imports
# ---------------------------

from ._timeseries import TimeSeries, TimeSeriesEvent
from ._loadsave import load, loadmat, save
from . import filters
from . import gui
from ._tools import explore, terminal
from . import kinematics
from . import pushrimkinetics
from . import dev
from . import dbinterface
from . import _repr


# Modify the repr function for dicts in iPython
try:
    import IPython as _IPython
    _ip = _IPython.get_ipython()
    formatter = _ip.display_formatter.formatters['text/plain']
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
gui.set_color_order('xyz')
