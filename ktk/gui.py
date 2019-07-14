#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module that provides simple GUI functions.

Author: Félix Chénier

Started on June 2019
"""

# Imports
import tkinter as _tk
from functools import partial as _partial
import sys as _sys

# Create the gui root on first import
from ktk.guiroot import root as _root
from ktk.guiroot import create_root as _create_root


def _on_closing():
    print('This window cannot be closed.')


def init():
    """Initialize the root window's properties."""
    _root.geometry('%dx%d+%d+%d' % (
                   _root.winfo_screenwidth(),
                   100, 0, _root.winfo_screenheight()-110))
    _root.protocol("WM_DELETE_WINDOW", _on_closing)


init()



def _cleantk():
    """
    Clean the dead tkinter window.

    This is a workaround for a broken IPython in Spyder 3 on macOS, where the
    IPython tkinter update loop reveals the last closed tkinter window.
    """
    # Clean the tkinter window by creating an empty, transparent one and
    # then destroying it.
    root = _tk.Tk()
    root.wm_attributes("-alpha", 0)
    root.update()
    root.withdraw()
    root.destroy()
    root.quit()


def _set_window_position(root, x, y):
    """Set the window position to given coordinates."""
    root.wm_attributes("-alpha", 0)
    root.update()
    geometry = root.geometry()
    geometry = geometry[0:geometry.find('+')]
    root.geometry(geometry + '+' + str(x) + "+" + str(y))
    root.wm_attributes("-alpha", 1)


# ------------------------------------
# MODULE'S PUBLIC FUNCTIONS
# ------------------------------------

def clear_root():
    """Clear the root window."""
    slaves = _root.slaves()
    for the_slave in slaves:
        the_slave.destroy()


def focus():
    """Give the focus to the root window."""
    _root.attributes('-topmost', True)
    _root.update()
    _root.attributes('-topmost', False)


def message(message=''):
    """
    Write a message in the bottom gui window.

    Parameters
    ----------
    message : str
        Message to write. Write '' to hide the gui window.

    Returns
    -------
    None.
    """
    clear_root()

    if message != '':
        label = _tk.Label(_root, text=message)
        label.pack()
        _root.deiconify()
        focus()


def buttondialog(title='', message='Please select an option.',
                 choices=['Cancel', 'OK']):
    """
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
    root = _tk.Tk()

    # We use a list of length 1 to pass selected_choice by reference.
    selected_choice = [-2]

    def return_choice(ichoice):
        selected_choice[0] = ichoice

        # iconify works around a bug on macOS where iPython redraws the last
        # tk window even after it has been withdrew.
        # root.iconify()
        root.wm_attributes("-topmost", 0)
        root.withdraw()
        root.destroy()
        root.quit()

    # Close button returns -1
    root.protocol("WM_DELETE_WINDOW", _partial(return_choice, -1))

    root.title(title)
    lbl = _tk.Label(root, text=message)
    lbl.grid(row=0, columnspan=len(choices))

    ichoice = 0
    for choice in choices:
        btn = _tk.Button(root,
                         text=choice,
                         command=_partial(return_choice, ichoice))
        btn.grid(row=1, column=ichoice)
        ichoice = ichoice + 1

    root.resizable(width=False, height=False)
    root.wm_attributes("-topmost", 1)  # Force topmost window
    _set_window_position(root, 100, 50)
    root.mainloop()
    _cleantk()

    return(selected_choice[0])


# ------------------------------------
# MAIN FUNCTION FOR EXTERNAL CALLS
# ------------------------------------


if __name__ == '__main__':
    if len(_sys.argv) < 2:
        raise ValueError('Not enough arguments.')

    if _sys.argv[1] == 'buttondialog':
        the_call = ('buttondialog("""' + _sys.argv[2]
                    + '""","""' + _sys.argv[3] + '""",[')
        for i in range(4, len(_sys.argv)):
            the_call = the_call + '"""' + _sys.argv[i] + '"""'
            if i == len(_sys.argv) - 1:
                the_call += '])'
            else:
                the_call += ','
    print(eval(the_call))
