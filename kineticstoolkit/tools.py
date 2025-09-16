#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020-2025 Félix Chénier

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Provide miscellaneous helper functions."""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2025 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"

import matplotlib as mpl
import warnings
import kineticstoolkit.config
import kineticstoolkit._repr as _repr
import kineticstoolkit.gui
from kineticstoolkit.typing_ import check_param


def check_interactive_backend() -> None:
    """
    Warn if Matplotlib is not using an interactive backend.

    To disable these warnings, for instance if we are generating
    documentation and we need the Player to show a figure, set
    ktk.config.interactive_backend_warning to False.

    """
    if kineticstoolkit.config.interactive_backend_warning is False:
        return

    warn = lambda: warnings.warn(
        "This function requires that Matplotlib uses an interactive "
        "backend. Try typing `%matplotlib qt5` before running this "
        "function."
    )

    try:
        mpl.backends
    except AttributeError:  # No backend has been initialized
        warn()
        return

    try:
        mpl.backends.backend  # type:ignore
    except AttributeError:  # No backend has been initialized
        warn()
        return

    if "inline" in mpl.backends.backend:  # type:ignore
        warn()
        return


def change_defaults(
    change_ipython_dict_repr: bool = True,
    change_matplotlib_defaults: bool = True,
    change_numpy_print_options: bool = True,
    change_warnings_format: bool = True,
) -> None:
    """
    Enable Kinetics Toolkit's lab goodies.

    This function does not affect Kinetics Toolkit's inner workings. It exists
    mostly for cosmetic reasons, so that working with ktk in an IPython console
    (e.g., Spyder, Jupyter) is more enjoyable. It changes IPython, Matplotlib,
    and numpy's defaults for the current session only. The usual way to call
    it is right after importing Kinetics Toolkit.

    Parameters
    ----------
    change_ipython_dict_repr
        Optional. True to summarize default dictionary printouts in IPython. When
        False, dictionary printouts look like::

            {'data1': array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
                             17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]),
             'data2': array([  0,   1,   4,   9,  16,  25,  36,  49,  64,  81, 100, 121, 144,
                             169, 196, 225, 256, 289, 324, 361, 400, 441, 484, 529, 576, 625,
                              676, 729, 784, 841])}

        When True, dictionary printouts look like::

            {
                'data1': <array of shape (30,)>
                'data2': <array of shape (30,)>
            }

    change_matplotlib_defaults
        Optional. True to change default figure size, autolayout, dpi, line
        width, and colour order in Matplotlib. The dpi and figure size are
        optimized for interactive work in default Matplotlib figures.
        Additionally, the default colour order is changed to (rgbcmyko).
        The first colours, (rgb), are consistent with the colours assigned to
        x, y, and z in most 3D visualization software.

    change_numpy_print_options
        Optional. True to change default print options in numpy to use fixed
        point notation in printouts.

    change_warnings_format
        Optional. True to change the warnings module's default to a more extended
        format with file and line number.

    Returns
    -------
    None

    Note
    ----
    This function is called automatically when importing Kinetics Toolkit in
    lab mode::

        import kineticstoolkit.lab as ktk

    """
    check_param("change_ipython_dict_repr", change_ipython_dict_repr, bool)
    check_param("change_matplotlib_defaults", change_matplotlib_defaults, bool)
    check_param("change_numpy_print_options", change_numpy_print_options, bool)
    check_param("change_warnings_format", change_warnings_format, bool)

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
