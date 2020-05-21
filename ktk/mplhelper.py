#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020 Félix Chénier
#
# This file is not for redistribution.
"""Helper functions for Matplotlib."""

import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import numpy as np
from functools import partial
import time


_axes = {
    'SidePane': None,
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
    'SidePane': 'ktk.mplhelper.side_pane',
    'Message': 'ktk.mplhelper.message',
}

def _get_axes(identifier):
    """
    Return the axes containing the side pane in the current figure.

    Parameters
    ----------
    identifier : str
        'SidePane' or 'Message'

    Returns
    -------
    axes : matplotlib axes
        Axes containing the side pane in the current figure, or none if the
        current figure does not contain a side pane.

    """
    fig = plt.gcf()
    for axes in fig._get_axes():
        if axes.get_label() == _AXES_ID[identifier]:
            return axes
    return None


def _add_side_pane():
    """Add a side pane to the current figure."""
    global _ax_left
    global _ax_width

    # Make room to the right
    fig = plt.gcf()  # Create a figure if there is no current figure.

    # If this figure already has some axes, then create a side pane.
    # Otherwise, the figure is empty and ths this will be a full pane.
    if len(fig.get_axes()) > 0:
        plt.subplots_adjust(right=0.6)
        _ax_left = _AX_LEFT
        _ax_width = _AX_WIDTH
    else:
        _ax_left = 0
        _ax_width = 1

    # Add the SidePane axes if it doesn't exist.
    side_pane = _get_axes('SidePane')
    if side_pane is None:
        side_pane = plt.axes([_ax_left, 0.01, _ax_width, 0.98],
                             xticklabels=[],
                             yticklabels=[],
                             xticks=[],
                             yticks=[],
                             facecolor='w')
        side_pane.spines['top'].set_visible(False)
        side_pane.spines['right'].set_visible(False)
        side_pane.spines['bottom'].set_visible(False)
        side_pane.spines['left'].set_visible(False)
        side_pane.set_label(_AXES_ID['SidePane'])

    return side_pane


def _remove_side_pane():
    """Remove the side pane from the current figure."""
    try:
        _get_axes('SidePane').remove()
    except Exception:
        pass
    plt.subplots_adjust(right=0.9)


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
    axes : matplotlib axes
        Axes that contains the message.

    """
    # Remove last text message if required.
    try:
        _get_axes('Message').remove()
    except Exception:
        pass

    # If we just wanted to erase
    if (text == '' or text is None):
        _remove_side_pane()
        return None

    _add_side_pane()

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

    # Update
    plt.pause(0.01)
    return ax_message


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
    fig = plt.gcf()
    _add_side_pane()

    # Write message
    message(text)

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
    plt.pause(0.1)
    while button_pressed[0] is None:
        fig.canvas.flush_events()
        time.sleep(0.01)

    # Clear text and buttons
    message('')
    for ax in ax_buttons:
        ax.remove()

    _remove_side_pane()

    return button_pressed[0]


def plot_mean_std(x, y, axis, color, label):
    """
    Plot a mean ± std curve with shaded std area.

    Parameters
    ----------
    x : array
        1-dimension array corresponding to the x axis.

    y : array
        2-dimension array corresponding to the y axis.

    axis : int
        Axis on which the data to average is aligned.

    color : str or list-like
        Color of the curve and shaded area.

    label : str
        Label of the curve, for the legend.

    """
    plt.plot(x, np.nanmean(y, axis=axis), color=color, linewidth=2,
             label=label)

    plt.fill_between(
        x,
        np.nanmean(y, axis=axis) -
        np.nanstd(y, axis=axis),
        np.nanmean(y, axis=axis) +
        np.nanstd(y, axis=axis),
        color=color, alpha=0.1)
