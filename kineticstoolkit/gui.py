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
This module is currently experimental and its API could be modified in
the future without warnings. It should be considered only as helper functions
for Kinetics Toolkit's own use.
"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"

import kineticstoolkit.config

import subprocess
from threading import Thread
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import os
from typing import Sequence, Union, Tuple, Any, List
from kineticstoolkit.decorators import experimental, unstable

CMDGUI = kineticstoolkit.config.root_folder + "/kineticstoolkit/cmdgui.py"
_message_window_int = [0]
listing = []  # type: List[str]


@experimental(listing)
def message(message: str = '') -> None:
    """
    Show a message window.

    Parameters
    ----------
    message
        The message to show. Use '' to close every message window.
    """
    # Begins by deleting the current message
    for file in os.listdir(kineticstoolkit.config.temp_folder):
        if 'gui_message_flag' in file:
            os.remove(kineticstoolkit.config.temp_folder + '/' + file)

    if message is None or message == '':
        return

    print(message)

    _message_window_int[0] += 1
    flagfile = (kineticstoolkit.config.temp_folder + '/gui_message_flag' +
                str(_message_window_int))

    fid = open(flagfile, 'w')
    fid.write("DELETE THIS FILE TO CLOSE THE KINETICS TOOLKIT GUI MESSAGE "
              "WINDOW.")
    fid.close()

    command_call = [sys.executable, CMDGUI, 'message', 'Kinetics Toolkit',
                    message, flagfile]

    def threaded_function():
        subprocess.call(command_call,
                        stderr=subprocess.DEVNULL)

    thread = Thread(target=threaded_function)
    thread.start()
    plt.pause(0.5)
    plt.pause(0.1)


@experimental(listing)
def button_dialog(message: str = 'Please select an option.',
                  choices: Sequence[str] = ['Cancel', 'OK']) -> int:
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
    # Run the button dialog in a separate thread to allow updating matplotlib
    button = [None]
    command_call = [sys.executable, CMDGUI, 'button_dialog',
                    'Kinetics Toolkit', message] + list(choices)

    def threaded_function():
        button[0] = int(subprocess.check_output(command_call,
                        stderr=subprocess.DEVNULL))

    thread = Thread(target=threaded_function)
    thread.start()

    while button[0] is None:
        plt.pause(0.2)  # Update matplotlib so that is responds to user input

    return button[0]


@unstable(listing)
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


@experimental(listing)
def get_credentials() -> Tuple[str, str]:
    """
    Ask the user's username and password.

    Returns
    -------
    Tuple[str]
        A tuple of two strings containing the username and password,
        respectively, or an empty tuple if the user closed the window.

    """
    str_call = ['get_credentials', 'Kinetics Toolkit',
                'Please enter your login information.']
    temp = subprocess.check_output([sys.executable, CMDGUI] + str_call,
                                   stderr=subprocess.DEVNULL)

    result = temp.decode(sys.getdefaultencoding())
    return tuple(str.split(result))  # type: ignore


@experimental(listing)
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
    str_call = ['get_folder', 'kineticstoolkit.gui.get_folder', initial_folder]
    temp = subprocess.check_output([sys.executable, CMDGUI] + str_call,
                                    stderr=subprocess.DEVNULL)
    result = temp.decode(sys.getdefaultencoding())
    result = result.replace('\n', '').replace('\r', '')
    return result


@experimental(listing)
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
    str_call = ['get_filename', 'kineticstoolkit.gui.get_filename',
                initial_folder]
    temp = subprocess.check_output([sys.executable, CMDGUI] + str_call,
                                    stderr=subprocess.DEVNULL)
    result = temp.decode(sys.getdefaultencoding())
    result = result.replace('\n', '').replace('\r', '')
    return result


def __dir__():
    return listing
