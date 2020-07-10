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
Provides simple GUI functions.

Warning: This module is currently experimental and its API could be modified in
the future without warnings. It should be considered only as helper functions
for ktk's own use.
"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"

import ktk.config

import subprocess
from threading import Thread
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import os


CMDGUI = ktk.config.root_folder + "/ktk/cmdgui.py"
_message_window_int = [0]


def __dir__():
    return ('button_dialog',
            'get_credentials',
            'get_folder',
            'get_filename',
            'set_color_order',
            'message')


def message(message=None):
    """
    Show a message window.

    Parameters
    ----------
    message : str
        The message to show. Use '' or None to close every message window.

    Returns
    -------
    None.
    """

    # Begins by deleting the current message
    for file in os.listdir(ktk.config.temp_folder):
        if 'gui_message_flag' in file:
            os.remove(ktk.config.temp_folder + '/' + file)

    if message is None or message == '':
        return

    print(message)

    _message_window_int[0] += 1
    flagfile = (ktk.config.temp_folder + '/gui_message_flag' +
                str(_message_window_int))

    fid = open(flagfile, 'w')
    fid.write('DELETE THIS FILE TO CLOSE THE KTK GUI MESSAGE WINDOW.')
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


def button_dialog(message='Please select an option.',
                  choices=['Cancel', 'OK']):
    """
    Create a blocking dialog message window with a selection of buttons.

    Parameters
    ----------
    message : str
        Message that is presented to the user.
        Default is 'Please select an option'.
    choices : list of str
        List of button text. Default is ['Cancel', 'OK'].

    Returns
    -------
    int : the button number (0 = First button, 1 = Second button, etc. If the
    user closes the window instead of clicking a button, a value of -1 is
    returned.
    """
    # Run the button dialog in a separate thread to allow updating matplotlib
    button = [None]
    command_call = [sys.executable, CMDGUI, 'button_dialog',
                    'Kinetics Toolkit', message] + choices

    def threaded_function():
        button[0] = int(subprocess.check_output(command_call,
                        stderr=subprocess.DEVNULL))

    thread = Thread(target=threaded_function)
    thread.start()

    while button[0] is None:
        plt.pause(0.2)  # Update matplotlib so that is responds to user input

    return button[0]


def set_color_order(setting):
    """
    Define the standard color order for matplotlib.

    Parameters
    ----------
    setting : str or list
        Either a string or a list of colors.
        - If a string, it can be either:
            - 'default' : Default v2.0 matplotlib colors.
            - 'classic' : Default classic Matlab colors (bgrcmyk).
            - 'xyz' :     Same as classic but begins with rgb instead of bgr to
                          be consistent with most 3d visualization softwares.
        - If a list, it can be either a list of chars from [bgrcmyk], a list of
          hexadecimal color values, or any list supported by matplotlib's
          axes.prop_cycle rcParam.

    Returns
    -------
    None.

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


def get_credentials():
    """
    Ask the user's username and password.

    Returns
    -------
    credentials : tuple
        A tuple of two strings containing the username and password,
        respectively, or an empty tuple if the user closed the window.

    """
    str_call = ['get_credentials', 'KTK',
                'Please enter your login information.']
    result = subprocess.check_output([sys.executable, CMDGUI] + str_call,
                                     stderr=subprocess.DEVNULL)

    result = result.decode(sys.getdefaultencoding())
    result = str.split(result)
    return tuple(result)


def get_folder(initial_folder='.'):
    """
    Get folder interactively using a file dialog window.

    Parameters
    ----------
    initial_folder : str (optional)
        The initial folder of the file dialog. Default is the current folder.

    Returns
    -------
    folder : str
        A string with the full path of the selected folder. An empty string
        is returned if the user cancelled.

    """
    str_call = ['get_folder', 'ktk.gui.get_folder', initial_folder]
    result = subprocess.check_output([sys.executable, CMDGUI] + str_call,
                                     stderr=subprocess.DEVNULL)
    result = result.decode(sys.getdefaultencoding())
    result = result.replace('\n', '').replace('\r', '')
    return result


def get_filename(initial_folder='.'):
    """
    Get file name interactively using a file dialog window.

    Parameters
    ----------
    initial_folder : str (optional)
        The initial folder of the file dialog. Default is the current folder.

    Returns
    -------
    file : str
        A string with the full path of the selected file. An empty string
        is returned if the user cancelled.
    """
    str_call = ['get_filename', 'ktk.gui.get_filename', initial_folder]
    result = subprocess.check_output([sys.executable, CMDGUI] + str_call,
                                     stderr=subprocess.DEVNULL)
    result = result.decode(sys.getdefaultencoding())
    result = result.replace('\n', '').replace('\r', '')
    return result
