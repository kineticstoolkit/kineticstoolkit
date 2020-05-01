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
from ktk._tools import explore, terminal
