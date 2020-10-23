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
Provide miscelleanous helper functions to the user.

These functions are accessible from Kinetics Toolkit's toplevel namespace
(i.e., ktk.explore).
"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import kineticstoolkit.config
import kineticstoolkit._repr as _repr
import kineticstoolkit.gui
from kineticstoolkit.decorators import stable, unstable, experimental

import os
import sys
import subprocess
import webbrowser as _webbrowser
from typing import Dict, List


listing = []  # type: List[str]


@stable(listing)
def explore(folder_name: str = '') -> None:
    """
    Open an Explorer window (on Windows) or a Finder window (on macOS)

    Parameters
    ----------
    folder_name
        Optional. The name of the folder to open the window in. Default is the
        current folder.

    """
    if not folder_name:
        folder_name = os.getcwd()

    if kineticstoolkit.config.is_pc is True:
        os.system(f'start explorer {folder_name}')

    elif kineticstoolkit.config.is_mac is True:
        subprocess.call(['open', folder_name])

    else:
        raise NotImplementedError('This function is only implemented on'
                                  'Windows and macOS.')


@stable(listing)
def terminal(folder_name: str = '') -> None:
    """
    Open a terminal window.

    Parameters
    ----------
    folder_name
        The name of the folder to open the terminal window in. Default is the
        current folder.

    Returns
    -------
    None.
    """
    if not folder_name:
        folder_name = os.getcwd()

    if kineticstoolkit.config.is_pc is True:
        os.system(f'cmd /c start /D {folder_name} cmd')

    elif kineticstoolkit.config.is_mac is True:
        subprocess.call([
                'osascript',
                '-e',
                """tell application "Terminal" to do script "cd '""" +
                    str(folder_name) + """'" """])
        subprocess.call([
                'osascript',
                '-e',
                'tell application "Terminal" to activate'])
    else:
        raise NotImplementedError('This function is only implemented on'
                                  'Windows and macOS.')


@stable(listing)
def start_lab_mode(*, config: Dict[str, bool] = {
        'change_ipython_dict_repr': True,
        'change_matplotlib_defaults': True,
        'change_numpy_print_options': True}) -> None:
    """
    Set Kinetics Toolkit to lab mode.

    This function does not affect Kinetics Toolkit's inner working. It exists
    mostly for cosmetic reasons, so that working with ktk in an IPython console
    (e.g., Spyder, Jupyter) is more enjoyable, at least to the developer's
    taste. It changes defaults and is not reversible in a given session. The
    usual way to call it is right after importing ktk.

    Parameters
    ----------
    config:
        'change_ipython_dict_repr' :
            True to summarize defaults dict printouts in IPython.
        'change_matplotlib_defaults' :
            True to change default figure size, dpi, line width and color
            order in Matplotlib.
        'change_numpy_print_options' :
            True to change default print options in numpy to use fixed point
            notation in printouts.

    Returns
    -------
    None

    """
    if config['change_ipython_dict_repr']:
        # Modify the repr function for dicts in IPython
        try:
            import IPython as _IPython
            _ip = _IPython.get_ipython()
            formatter = _ip.display_formatter.formatters['text/plain']
            formatter.for_type(dict, lambda n, p, cycle:
                               _repr._ktk_format_dict(n, p, cycle))
        except Exception:
            pass

    if config['change_matplotlib_defaults']:
        # Set alternative defaults to matplotlib
        import matplotlib as _mpl
        _mpl.rcParams['figure.figsize'] = [10, 5]
        _mpl.rcParams['figure.dpi'] = 75
        _mpl.rcParams['lines.linewidth'] = 1
        kineticstoolkit.gui.set_color_order('xyz')

    if config['change_numpy_print_options']:
        import numpy as _np
        # Select default mode for numpy
        _np.set_printoptions(suppress=True)


@unstable(listing)
def update() -> None:
    """
    Update Kinetics Toolkit to the latest master branch.

    If Kinetics Toolkit was installed using pip (default when using the stable
    version), then update using pip instead:

        pip upgrade kineticstoolkit

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    if kineticstoolkit.config.is_pc:
        current_dir = os.getcwd()
        os.chdir(kineticstoolkit.config.root_folder)
        print(subprocess.check_output(
            ['git', 'pull', 'origin', 'master']).decode(sys.getdefaultencoding()))
        os.chdir(current_dir)

    if kineticstoolkit.config.is_mac:
        subprocess.call([
            'osascript',
            '-e',
            'tell application "Terminal" to activate'])
        subprocess.call([
            'osascript',
            '-e',
            ('tell application "Terminal" to do script "cd \'' +
             kineticstoolkit.config.root_folder +
             '\'; git pull origin master"')])


@experimental(listing)
def tutorials() -> None:
    """
    Open the Kinetics Toolkits tutorials in a web browser.

    Usage: ktk.tutorials()
    """
    _webbrowser.open('file:///' + kineticstoolkit.config.root_folder +
                     '/tutorials/index.html',
                     new=2)


def __dir__():
    return listing
