#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020 Félix Chénier
#
# This file is not for redistribution.
"""
Module that provides simple GUI functions.
"""

import subprocess
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import ktk
import matplotlib.widgets as widgets
from functools import partial
import time


def __dir__():
    return ['message', 'button_dialog', 'set_color_order',
            'get_credentials', 'get_folder', 'get_filename']


CMDGUI = ktk.config['RootFolder'] + "/ktk/cmdgui.py"
_message_window_int = [0]

_axes = {
    'GUIPane': None,
    'Message': None
}

# Defaults for side panes
_AX_LEFT = 0.65
_AX_WIDTH = 0.34

# Current pane
_ax_left = _AX_LEFT
_ax_width = _AX_WIDTH

# axes identifiers
_AXES_ID = {
    'GUIPane': 'ktk.gui.gui_pane',
    'Message': 'ktk.gui.message',
}


def _figure_safe_close(figure):
    """
    Bind the figure's close event to window destroy.

    This is a workaround for plt.close and close button that keep a zombie
    window lying behind.

    This bug is submitted to matplotlib.
    https://github.com/matplotlib/matplotlib/issues/17109
    """
    figure.canvas.mpl_connect('close_event',
                              lambda _: figure.canvas.manager.window.destroy())


def _figure():
    """
    Create a figure and bind its close event to window destroy.

    This is a workaround for plt.close and close button that keep a zombie
    window lying behind.

    This bug is submitted to matplotlib.
    https://github.com/matplotlib/matplotlib/issues/17109
    """
    figure = plt.figure()
    _figure_safe_close(figure)
    return figure


def _pause(time):
    """Pause a figure to refresh and work around matplotlib issue 17109."""
    plt.pause(time)
    _figure_safe_close(plt.gcf())


def _gcf():
    """
    Get current figure and bind its close event to window destroy.

    This is a workaround for plt.close and close button that keep a zombie
    window lying behind.

    This bug is submitted to matplotlib.
    https://github.com/matplotlib/matplotlib/issues/17109
    """
    figure = plt.gcf()
    _figure_safe_close(figure)
    return figure



def _get_axes(identifier):
    """
    Return the axes containing the side pane in the current figure.

    Parameters
    ----------
    identifier : str
        'GUIPane' or 'Message'

    Returns
    -------
    axes : matplotlib axes
        Axes containing the side pane in the current figure, or none if the
        current figure does not contain a side pane.

    """
    fig = _gcf()

    for axes in fig._get_axes():
        if axes.get_label() == _AXES_ID[identifier]:
            return axes
    return None


def _show_toolbar():
    """Reveal matplotlib's toolbar in current figure (Qt only)."""
    try:  # Try, setVisible method not always there
        _gcf().canvas.toolbar.setVisible(True)
    except AttributeError:
        pass


def _hide_toolbar():
    """Hide matplotlib's toolbar in current figure (Qt only)."""
    try:  # Try, setVisible method not always there
        _gcf().canvas.toolbar.setVisible(False)
    except AttributeError:
        pass


def _set_title(title):
    """Set the title of the current figure window."""
    _gcf().canvas.set_window_title(title)


def _add_gui_pane():
    """Add a side pane or fullframe pane to the current figure."""
    global _ax_left
    global _ax_width

    fig = _gcf()  # Create a figure if there is no current figure.

    # If this figure already has some axes, then create a side pane.
    # Otherwise, the figure is empty and ths this will be a full pane.
    if len(fig.get_axes()) > 0:
        plt.subplots_adjust(right=0.6)
        _ax_left = _AX_LEFT
        _ax_width = _AX_WIDTH
    else:
        _ax_left = 0
        _ax_width = 1
        _hide_toolbar()
        _set_title('Kinetics Toolkit')

    # Add the GUIPane axes if it doesn't exist.
    gui_pane = _get_axes('GUIPane')
    if gui_pane is None:
        gui_pane = plt.axes([_ax_left, 0.01, _ax_width, 0.98],
                            xticklabels=[],
                            yticklabels=[],
                            xticks=[],
                            yticks=[],
                            facecolor='w')
        gui_pane.spines['top'].set_visible(False)
        gui_pane.spines['right'].set_visible(False)
        gui_pane.spines['bottom'].set_visible(False)
        gui_pane.spines['left'].set_visible(False)
        gui_pane.set_label(_AXES_ID['GUIPane'])

    return gui_pane


def _remove_gui_pane():
    """Remove the gui pane from the current figure."""
    try:
        _get_axes('GUIPane').remove()
    except Exception:
        pass
    plt.subplots_adjust(right=0.9)

    # If there is nothing left in the figure, then close it.
    if len(_gcf().get_axes()) == 0:
        plt.close()


def message(text):
    """
    Write a message on the current figure.

    Parameters
    ----------
    text : str
        Message to write on the right of the current figure. If '' or None,
        erases the last message.

    Returns
    -------
    None

    """
    # Remove last text message if required.
    try:
        _get_axes('Message').remove()
    except Exception:
        pass

    # If we just wanted to erase
    if (text == '' or text is None):
        _remove_gui_pane()
        return None

    _add_gui_pane()

    # Write message
    ax_message = _get_axes('Message')
    if ax_message is None:
        ax_message = plt.axes([_ax_left, 0.01, _ax_width, 0.98],
                              xticklabels=[],
                              yticklabels=[],
                              xticks=[],
                              yticks=[])
        ax_message.set_label(_AXES_ID['Message'])
        ax_message.set_frame_on(False)

    ax_message.text(0.5, 0.8, text, wrap=True,
                    horizontalalignment='center')


def button_dialog(text='Please select an option.',
                  choices=['Cancel', 'OK']):
    """
    Ask the user to select a button from a list of buttons.

    Parameters
    ----------
    text : str
        Message that is presented to the user.
        The default is 'Please select an option'.
    choices : list of str
        List of button text. The default is ['Cancel', 'OK'].

    Returns
    -------
    button : int
        The button number (0 = First button, 1 = Second button, etc. If the
        user closes the window instead of clicking a button, a value of -1 is
        returned.

    """
    # Write message (this will take care of creating the gui pane)
    message('')
    message(text)

    fig = _gcf()
    _figure_safe_close(fig)

    # Write buttons
    button_pressed = [None]

    def button_callback(i_button, *args):
        button_pressed[0] = i_button

    def close_callback(*args):
        button_pressed[0] = -1

    ax_buttons = []
    buttons = []
    height = 0.075
    for i_choice, choice in enumerate(choices):
        ax_buttons.append(plt.axes([_ax_left + 0.01, 0.70 - height * i_choice,
                                    _ax_width - 0.02, height - 0.01]))
        buttons.append(widgets.Button(ax_buttons[-1], choice))
        buttons[-1].on_clicked(partial(button_callback, i_choice))

    fig.canvas.mpl_connect('close_event', partial(close_callback))

    # Wait for button press of figure close
    plt.show(block=False)
    fig.canvas.start_event_loop(0.01)

    # _figure_safe_close(fig)
    while button_pressed[0] is None:
        fig.canvas.flush_events()
        time.sleep(0.01)

    to_return = int(button_pressed[0])

    # Clear text and buttons
    message('')
    for ax in ax_buttons:
        ax.remove()

    _remove_gui_pane()

    return to_return


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
    folder : str
        A string with the full path of the selected folder.

    """
    str_call = ['get_folder', title, initial_folder]
    result = subprocess.check_output([sys.executable, CMDGUI] + str_call,
                                     stderr=subprocess.DEVNULL)
    result = result.decode('ascii')
    result = result.replace('\n', '').replace('\r', '')
    return result


def get_filename(title='KTK', initial_folder='.'):
    """
    Get file name interactively using a file dialog window.

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
    file : str
        A string with the full path of the selected file.
    """
    str_call = ['get_filename', title, initial_folder]
    result = subprocess.check_output([sys.executable, CMDGUI] + str_call,
                                     stderr=subprocess.DEVNULL)
    result = result.decode('ascii')
    result = result.replace('\n', '').replace('\r', '')
    return result
