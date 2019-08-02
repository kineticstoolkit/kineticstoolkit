#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module that provides simple GUI functions.

Author: Félix Chénier

Started on June 2019
"""

import subprocess
from ktk import _ROOT_FOLDER

CMDGUI = _ROOT_FOLDER + "/ktk/cmdgui.py"

def __dir__():
    """Generate a dir for tab-completion in IPython."""
    return ['button_dialog', 'get_credentials', 'get_folder']


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
    str_call = ['button_dialog', title, message] + choices
    result = subprocess.check_output([CMDGUI] + str_call,
                                     stderr=subprocess.DEVNULL)

    return int(result)
