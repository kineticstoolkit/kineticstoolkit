"""
Kinetics Toolkit
================

Kinetics Toolkit (ktk) is an in-house biomechanical library developed by
Professor Félix Chénier at Université du Québec à Montréal. Originally
programmed in and for Matlab, ktk is quickly becoming a collection of Python
modules that aim to manage experimental files using a database and process
3d kinetics, 3d kinematics and EMG data.

Please see this webpage for support: https://felixchenier.com/kineticstoolkit

Author: Félix Chénier

Date: Started on July 2019
"""
import os as _os
import platform as _platform
import matplotlib as _mpl
import numpy as _np


# ---------------------------
# Set ktk configuration
# ---------------------------
# Root folder (ktk installation)
config = dict()
config['RootFolder'] = _os.path.dirname(_os.path.dirname(__file__))

# Operating system
config['IsPC'] = True if _platform.system() == 'Windows' else False
config['IsMac'] = True if _platform.system() == 'Darwin' else False
config['IsLinux'] = True if _platform.system() == 'Linux' else False

# ---------------------------
# Imports
# ---------------------------

from ktk._timeseries import TimeSeries, TimeSeriesEvent
from ktk._dbinterface import DBInterface
from ktk._player import Player
from ktk._loadsave import load, loadmat, save
from ktk import filters
from ktk import gui
from ktk._tools import explore, terminal, update, tutorials
from ktk import geometry
from ktk import kinematics
from ktk import pushrimkinetics
from ktk import inversedynamics
from ktk import dev
from ktk import cycles
from ktk import _repr

# ---------------------------
# Customizations
# ---------------------------

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

# Select default mode for numpy
_np.set_printoptions(suppress=True)
