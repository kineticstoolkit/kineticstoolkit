#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module that provides simple GUI functions.

Author: Félix Chénier

Started on June 2019
"""

import subprocess
from threading import Thread
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import ktk
from time import sleep

CMDGUI = ktk.config['RootFolder'] + "/ktk/cmdgui.py"

def __dir__():
    return ('button_dialog',
            'get_credentials',
            'get_folder',
            'set_color_order',
            'message')


def message(message):
    """
    Print a message and two beeps to the user to request his/her action.

    Parameters
    ----------
    message : str
        The message to write in the console.

    Returns
    -------
    None.
    """
    dash_length = 80
    print('=' * dash_length)
    print(message)
    print('-' * dash_length + '\a', flush=True)  # Bell sound = \a
    sleep(0.4)
    print('\a', flush=True)  # Bell sound


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
    None
    """
    if isinstance(setting, str):
        if setting == 'default':
            thelist = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        elif setting == 'classic':
            thelist = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        elif setting == 'xyz':
            thelist = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        else:
            raise(ValueError('This setting is not recognized.'))
    elif isinstance(setting, list):
        thelist = setting
    else:
        raise(ValueError('This setting is not recognized.'))

    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(
          'color', thelist)


def get_credentials():
    """
    Ask the user's username and password.

    Returns
    -------
    A tuple of two strings containing the username and password, respectively,
    or an empty tuple if the user closed the window.
    """
    str_call = ['get_credentials', 'KTK', 'Please enter your login information.']
    result = subprocess.check_output([CMDGUI] + str_call,
                                     stderr=subprocess.DEVNULL)

    result = result.decode()
    result = str.split(result)
    return tuple(result)


def get_folder(title='KTK', initial_folder='.'):
    """
    Get folder interactively using a file dialog window.

    Parameters
    ----------
    title : str (optional)
        The title of the file dialog. On many OSes, no title is shown so it is
        a good idea to not rely on this title to give instructions to the user.
        Default : 'KTK'
    initial_folder : str (optional)
        The initial folder of the file dialog. Default is the current folder.

    Returns
    -------
    A string with the full path of the selected folder.
    """
    str_call = ['get_folder', title, initial_folder]
    result = subprocess.check_output([CMDGUI] + str_call,
                                     stderr=subprocess.DEVNULL)
    result = str.split(result.decode())
    return result[0]


def button_dialog(title='KTK', message='Please select an option.',
                  choices=['Cancel', 'OK']):
    """
    Ask the user to select among multiple buttons.

    Create a topmost dialog window with a selection of buttons.
    uibuttonsdialog(title, message, choices) is a blocking
    function that asks the user to click on a button, using a topmost dialog
    window.

    Parameters
    ----------
    title : str
        Title of the dialog window. Default is ''.
    message : str
        Message that is presented to the user.
        Default is 'Please select an option'.
    choices : list of str
        List of button text. Default is ['Cancel', 'OK'].

    Returns
    -------
    The button number (0 = First button, 1 = Second button, etc. If the
    user closes the window instead of clicking a button, a value of -1 is
    returned.
    """

    # Run the button dialog in a separate thread to allow updating matplotlib
    button = [None]
    command_call = [sys.executable, CMDGUI, 'button_dialog', title,
                    message] + choices

    def threaded_function():
        button[0] = int(subprocess.check_output(command_call,
                        stderr=subprocess.DEVNULL))

    thread = Thread(target=threaded_function)
    thread.start()

    while button[0] is None:
        plt.pause(0.2)  # Update matplotlib so that is responds to user input

    return button[0]
