#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module that provides simple GUI functions.

Author: Félix Chénier

Started on June 2019
"""

from PyQt5.QtWidgets import QMessageBox, QWidget
from PyQt5.QtCore import Qt


def __dir__():
    """Generate a dir for tab-completion in IPython."""
    return ['button_dialog', 'message']


def _get_main_dialog_window():
    """
    Create a main dialog window that is always on top.

    Returns
    -------
    widget : QWidget
        A widget that can serve as a parent for dialogs, so that they are
        always on top.

    """
    widget = QWidget(None, Qt.WindowStaysOnTopHint)
    return widget



def message(the_message):
    widget = _get_main_dialog_window()
    message_box = QMessageBox(_get_main_dialog_window())
    message_box.show()





def button_dialog(message, buttons):
    """
    Ask the user to choice among standard buttons.

    Parameters
    ----------
    message : str
        Message to write in to dialog window.
    buttons : list
        List of strings. The strings may be either:
            - 'cancel'
            - 'ok'
            - 'help'
            - 'open'
            - 'save'
            - 'saveall'
            - 'discard'
            - 'close'
            - 'apply'
            - 'reset'
            - 'yes'
            - 'yestoall'
            - 'no'
            - 'notoall'
            - 'restoredefault'
            - 'abort'
            - 'retry'
            - 'ignore'

    Returns
    -------
    The string corresponding to the clicked button.

    """
    strings = ['cancel', 'ok', 'help', 'open', 'save', 'saveall',
               'discard', 'close', 'apply', 'reset', 'yes', 'yestoall', 'no',
               'notoall', 'restoredefault', 'abort', 'retry',
               'ignore']

    codes = [QMessageBox.Cancel, QMessageBox.Ok, QMessageBox.Help,
             QMessageBox.Open, QMessageBox.Save, QMessageBox.SaveAll,
             QMessageBox.Discard, QMessageBox.Close, QMessageBox.Apply,
             QMessageBox.Reset, QMessageBox.Yes, QMessageBox.YesToAll,
             QMessageBox.No, QMessageBox.NoToAll, QMessageBox.RestoreDefaults,
             QMessageBox.Abort, QMessageBox.Retry, QMessageBox.Ignore]

    button_codes_to_show = QMessageBox.NoButton
    for one_button_string in buttons:
        button_codes_to_show |= codes[strings.index(one_button_string)]

    result = QMessageBox.information(_get_main_dialog_window(), 'KTK',
                                     message, button_codes_to_show)

    return strings[codes.index(result)]
