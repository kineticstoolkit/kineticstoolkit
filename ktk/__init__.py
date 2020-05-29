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

Kinetics Toolkit (ktk) is a pure-python biomechanical library developed by
Professor Félix Chénier at Université du Québec à Montréal, Canada. It does not
attempt to provide user-friendly graphical user interfaces (apart from the
Player class to visualize 3d kinematics) or magical blackboxes that process
everything automatically. It is rather a framework that aims to integrate
flexible classes and functions to facilitate research in biomechanics.

Although most ktk modules express data using the custom ktk.TimeSeries class,
it is still easy to integrate ktk with other environments using pandas
Dataframes as intermediate containers, using the TimeSeries' from_dataframe
and to_dataframe methods.

Kinetics Toolkit is addressed mainly to researchers and students in
biomechanics with a little background in programming, who want to understand
and control their data. This is why special attention is made to API
documentation and tutorials.

[Laboratory website](https://felixchenier.uqam.ca)

[Kinetics Toolkit (ktk) website](https://felixchenier.uqam.ca/kineticstoolkit)

Public version
--------------

The public open-source version API is mostly stable and currently includes:

- `timeseries.TimeSeries` : a generic class to represent time-varying
  n-dimensional data and events, with many methods to extract, merge and subset
  TimeSeries data.

- `kinematics` : a module that loads c3d and n3d files as TimeSeries of
  3d marker positions.

- `player.Player` : a class that allows visualizing 3d markers using a simple
  graphical user interface.

- and some helper functions.

Please be warned that this is mostly experimental software. If you are using
ktk or are planning to be, you are warmly invited to contact me, first to say
Hello :-), and so that I can warn you before doing major, possibly breaking
changes. Also remind that I develop ktk mainly for my lab and I have limited
resources for troubleshooting. You can however
[ask your questions](mailto:chenier.felix@uqam.ca)
and if I can answer, I'll do.

[Tutorials](https://felixchenier.uqam.ca/ktk_dist/tutorials)

[API documentation](https://felixchenier.uqam.ca/ktk_dist/api)


Private development version
---------------------------

The development version is exclusively used in my lab and is developed in
parallel with my research projects, following the needs of the moment. I
usually wait several months before releasing new code to the public, mostly to
ensure the modules are stable, well tested, documented, and the API is mature and global enough to be shared.

[Tutorials](https://felixchenier.uqam.ca/ktk_lab/tutorials)

[API documentation](https://felixchenier.uqam.ca/ktk_lab/api)


Credits
-------

Some external code has been directly included into ktk's source code. Here are
the credits for these nice people.

- Clay Flannigan : [icp](https://github.com/ClayFlannigan/icp) -
  Python implementation of m-dimensional Iterative Closest Point method

I also want to credit the people involved in ktk's dependencies:

- Pariterre and contributors : [ezc3d](https://github.com/pyomeca/ezc3d) -
  Easy to use C3D reader/writer for C++, Python and Matlab

- The dedicated people behind major software and packages used by ktk such as
  python, numpy, matplotlib, pandas, jupyter, pytest, pdoc3, etc.

-------------------------------------------------------------------------------


Customization
-------------

By default, importing ktk changes some defaults in IPython and matplotlib to
get a more 'research' and less 'programming' experience. Please note that this
does not affect anything besides visual representations. This behaviour can be
changed by modifying ktk's configuration (ktk/config.py). In Spyder/IPython:

    >>> import ktk.config
    >>> edit ktk.config

### Modification to repr of dictionaries ###

In ktk, data are often stored as dictionaries, which can lead to very large
printouts when we simply want to see the dictionary's contents. Importing ktk
changes the repr of dictionaries in IPython so that a summary of the dict's
content is shown, more like the representation of a Matlab struct.

    >>> import numpy as np
    >>> data = dict()
    >>> data['data1'] = np.arange(30)
    >>> data['data2'] = np.arange(30) ** 2
    >>> data['data3'] = np.arange(30) ** 3

Before importing ktk:

    >>> data
    {'data1': array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]),
     'data2': array([  0,   1,   4,   9,  16,  25,  36,  49,  64,  81, 100, 121, 144,
            169, 196, 225, 256, 289, 324, 361, 400, 441, 484, 529, 576, 625,
            676, 729, 784, 841]),
     'data3': array([    0,     1,     8,    27,    64,   125,   216,   343,   512,
              729,  1000,  1331,  1728,  2197,  2744,  3375,  4096,  4913,
             5832,  6859,  8000,  9261, 10648, 12167, 13824, 15625, 17576,
            19683, 21952, 24389])}

After importing ktk:

    >>> import ktk
    >>> data
    {
        'data1': <array of shape (30,)>,
        'data2': <array of shape (30,)>,
        'data3': <array of shape (30,)>
    }

### Modification to repr of numpy's floats ###

Numpy is set to display floats with floating point precision.

### Alternative defaults for matplotlib ###

We assume that most work with figure is interactive, on screen. In that view,
the following modifications are made to default matplotlib figures:

- The standard dpi is changed to 75, which allows for more space to work by
  reducing the font size on screen.

- The standard figure size is changed to [10, 5], which is a little bigger
  than the default and is thus more practical for interactive navigation.

- The default color order is changed to (rgbcmyko) with o being orange. The
  first colors, red, green and blue, are consistent the colours assigned to
  x, y and z in most 3D visualization softwares, and the next colours are
  consistent with Matlab's legacy color order.

"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"

# ######################################################################

__pdoc__ = {'dev': False, 'cmdgui': False}

# --- Imports
from ktk import config
from ktk.timeseries import TimeSeries, TimeSeriesEvent
from ktk.tools import explore, terminal, tutorials
from ktk.player import Player
from ktk import gui
from ktk import kinematics
from ktk import _repr

try:
    from ktk import dev
except Exception:
    pass


# --- Customizations

if config.change_ipython_dict_repr is True:
    # Modify the repr function for dicts in IPython
    try:
        import IPython as _IPython
        _ip = _IPython.get_ipython()
        formatter = _ip.display_formatter.formatters['text/plain']
        formatter.for_type(dict, lambda n, p, cycle:
                           _repr._ktk_format_dict(n, p, cycle))
    except Exception:
        pass

if config.change_matplotlib_defaults is True:
    # Set alternative defaults to matplotlib
    import matplotlib as _mpl
    _mpl.rcParams['figure.figsize'] = [10, 5]
    _mpl.rcParams['figure.dpi'] = 75
    _mpl.rcParams['lines.linewidth'] = 1
    gui.set_color_order('xyz')

if config.change_numpy_print_options is True:
    import numpy as _np
    # Select default mode for numpy
    _np.set_printoptions(suppress=True)
