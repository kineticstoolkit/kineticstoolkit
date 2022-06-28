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

"""Provide miscelleanous helper functions."""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import kineticstoolkit.config
import kineticstoolkit._repr as _repr
import kineticstoolkit.gui
import kineticstoolkit.ext
from kineticstoolkit.decorators import directory

import warnings


def start_lab_mode(
    change_ipython_dict_repr: bool = True,
    change_matplotlib_defaults: bool = True,
    change_numpy_print_options: bool = True,
    change_warnings_format: bool = True,
    import_extensions: bool = True,
) -> None:
    """
    Enable Kinetics Toolkit's lab goodies.

    This function does not affect Kinetics Toolkit's inner working. It exists
    mostly for cosmetic reasons, so that working with ktk in an IPython console
    (e.g., Spyder, Jupyter) is more enjoyable, at least to the developer's
    taste. It changes defaults and is not reversible in a given session. The
    usual way to call it is right after importing Kinetics Toolkit.

    Parameters
    ----------
    change_ipython_dict_repr
        Optional. True to summarize defaults dict printouts in IPython.
    change_matplotlib_defaults
        Optional. True to change default figure size, autolayout, dpi, line
        width and color order in Matplotlib.
    change_numpy_print_options
        Optional. True to change default print options in numpy to use fixed
        point notation in printouts.
    change_warnings_format
        Optional. True to change warnings module's default to a more extended
        format with file and line number.
    import_extensions
        Optional. True to import extensions into the kinetics.ext namespace.

    Returns
    -------
    None

    """
    if change_ipython_dict_repr:
        # Modify the repr function for dicts in IPython
        try:
            import IPython as _IPython

            _ip = _IPython.get_ipython()
            formatter = _ip.display_formatter.formatters["text/plain"]
            formatter.for_type(
                dict, lambda n, p, cycle: _repr._ktk_format_dict(n, p, cycle)
            )
        except Exception:
            pass

    if change_matplotlib_defaults:
        # Set alternative defaults to matplotlib
        import matplotlib as _mpl

        _mpl.rcParams["figure.figsize"] = [10, 5]
        _mpl.rcParams["figure.dpi"] = 75
        _mpl.rcParams["lines.linewidth"] = 1
        kineticstoolkit.gui.set_color_order("xyz")

    if change_numpy_print_options:
        import numpy as _np

        # Select default mode for numpy
        _np.set_printoptions(suppress=True)

    if change_warnings_format:
        # Monkey-patch warning.formatwarning
        def formatwarning(message, category, filename, lineno, line=None):
            return f"{category.__name__} [{filename}:{lineno}] {message}\n"

        warnings.formatwarning = formatwarning

    if import_extensions:
        kineticstoolkit.ext._import_extensions()


module_locals = locals()


def __dir__():
    return directory(module_locals)
