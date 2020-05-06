"""
Kinetics Toolkit
================

Kinetics Toolkit (ktk) is an in-house biomechanical library developed
exclusively in Python, by Professor Félix Chénier at Université du Québec
à Montréal.

[Laboratory website](https://felixchenier.uqam.ca)

[Kinetics Toolkit (ktk) website](https://felixchenier.uqam.ca/kineticstoolkit)

Public version
--------------

The public open-source version API is mostly stable (although currently almost
empty). I do not expect to remove or rename much stuff. However please keep
in mind that this is experimental software. If you are using ktk or are
planning to be, you are warmly invited to contact me, first to say Hello :-),
and so that I can warn you before doing major, possibly breaking changes.

[Tutorials](https://felixchenier.uqam.ca/ktk_dist/tutorials)

[API documentation](https://felixchenier.uqam.ca/ktk_dist/api)


Private unstable version
------------------------

This version is exclusively used in my lab and is developed in parallel with
my research projects, following the needs of the moment. I usually wait several
months before releasing code to the public, mostly to ensure the modules are
stable and the API is mature and global enough to be shared. If you are
interested in collaborating either in research or software development, please
contact me at chenier.felix@uqam.ca

[Tutorials](https://felixchenier.uqam.ca/ktk_lab/tutorials)

[API documentation](https://felixchenier.uqam.ca/ktk_lab/api)


"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"

import os as _os
import platform as _platform
import matplotlib as _mpl
import numpy as _np

__pdoc__ = {'dev': False}

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

try:
    from ktk import dev
except Exception:
    pass


def _workspace():
    """Set ktk in custom workspace mode."""
    # Modify the repr function for dicts in IPython
    try:
        import IPython as _IPython
        _ip = _IPython.get_ipython()
        formatter = _ip.display_formatter.formatters['text/plain']
        formatter.for_type(dict, lambda n, p, cycle:
                           _repr._ktk_format_dict(n, p, cycle))
    except Exception:
        pass
