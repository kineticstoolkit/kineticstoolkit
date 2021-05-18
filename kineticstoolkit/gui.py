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
Provide simple GUI functions.

Warning
-------
This module is private and should be considered only as helper functions
for Kinetics Toolkit's own use.
"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"

import kineticstoolkit.config as config
import limitedinteraction as li
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import os
from typing import Sequence, Union, Tuple, Any, List


def message(message: str = '', **kwargs) -> None:
    """
    Show a message window.

    Parameters
    ----------
    message
        The message to show. Use '' to close every message window.
    """
    li.message(
        message,
        icon=[config.root_folder + '/kineticstoolkit/logo.png',
              config.root_folder + '/kineticstoolkit/logo_hires.png'],
        **kwargs)


def button_dialog(message: str = 'Please select an option.',
                  choices: Sequence[str] = ['Cancel', 'OK'],
                  **kwargs) -> int:
    """
    Create a blocking dialog message window with a selection of buttons.

    Parameters
    ----------
    message
        Message that is presented to the user.
    choices
        List of button text.

    Returns
    -------
    int
        The button number (0 = First button, 1 = Second button, etc.) If the
        user closes the window instead of clicking a button, a value of -1 is
        returned.
    """
    return li.button_dialog(
        message, choices,
        icon=[config.root_folder + '/kineticstoolkit/logo.png',
              config.root_folder + '/kineticstoolkit/logo_hires.png'],
        **kwargs)


def set_color_order(setting: Union[str, Sequence[Any]]) -> None:
    """
    Define the standard color order for matplotlib.

    Parameters
    ----------
    setting
        Either a string or a list of colors.

        - If a string, it can be either:
            - 'default': Default v2.0 matplotlib colors.
            - 'classic': Default classic Matlab colors (bgrcmyk).
            - 'xyz': Same as classic but begins with rgb instead of bgr to
               be consistent with most 3d visualization softwares.

        - If a list, it can be either a list of chars from [bgrcmyk], a list of
          hexadecimal color values, or any list supported by matplotlib's
          axes.prop_cycle rcParam.

    """
    if isinstance(setting, str):
        if setting == 'default':
            thelist = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        elif setting == 'classic':
            thelist = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        elif setting == 'xyz':
            thelist = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'tab:orange']
        else:
            raise(ValueError('This setting is not recognized.'))
    elif isinstance(setting, list):
        thelist = setting
    else:
        raise(ValueError('This setting is not recognized.'))

    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=thelist)


def get_credentials() -> Tuple[str, str]:
    """
    Ask the user's username and password.

    Returns
    -------
    Tuple[str]
        A tuple of two strings containing the username and password,
        respectively, or an empty tuple if the user closed the window.

    """
    return li.input_dialog('Please enter your credentials',
                           descriptions=['Username:', 'Password:'],
                           masked=[False, True],
                           icon='lock')


def get_folder(initial_folder: str = '.') -> str:
    """
    Get folder interactively using a file dialog window.

    Parameters
    ----------
    initial_folder
        Optional. The initial folder of the file dialog.

    Returns
    -------
    str
        The full path of the selected folder. An empty string is returned if
        the user cancelled.

    """
    return li.get_folder(
        initial_folder,
        icon=[config.root_folder + '/kineticstoolkit/logo.png',
              config.root_folder + '/kineticstoolkit/logo_hires.png'])


def get_filename(initial_folder: str = '.') -> str:
    """
    Get file name interactively using a file dialog window.

    Parameters
    ----------
    initial_folder
        Optional. The initial folder of the file dialog.

    Returns
    -------
    str
        The full path of the selected file. An empty string is returned if the
        user cancelled.
    """
    return li.get_filename(
        initial_folder,
        icon=[config.root_folder + '/kineticstoolkit/logo.png',
              config.root_folder + '/kineticstoolkit/logo_hires.png'])
