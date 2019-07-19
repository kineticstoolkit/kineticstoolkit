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
#from ktk.guiroot import root as _root


def _create_root(title='KTK', width=800, height=400):
    root = _tk.Tk()
    # Ensure the window is not created as a tab on macOS
    root.resizable(width=False, height=False)
    root.title(title)
    
    # Ensure the window is not closable by user
    def _on_closing():
        pass
    root.protocol("WM_DELETE_WINDOW", _on_closing)

    # Put the window in center of the screen
    root.geometry('%dx%d+%d+%d' % (
                  width, height,
                  root.winfo_screenwidth()/2-width/2,
                  root.winfo_screenheight()/2-height/2))

    # Set focus
    root.attributes('-topmost', True)
    root.update()
    root.attributes('-topmost', False)

    return root


def _destroy_root(root):
    """
    Clean the dead tkinter window.

    Contains a workaround for a broken IPython in Spyder 3 on macOS, where the
    IPython tkinter update loop reveals the last closed tkinter window.
    """
    root.destroy()

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


def get_credentials():
    """
    Get username and password using a password dialog.
    
    Parameters
    ----------
    None.
    
    Returns
    -------
    A tuple of strings: (username, password)
    """
    
    def ok_pressed(*args):
        credentials[0] = username_box.get()
        credentials[1] = password_box.get()
        _destroy_root(root)
    
    credentials = ['', '']
    
    root = _create_root(title='Enter credentials',width=300, height=100)
    username_box = _tk.Entry(root)
    username_box.insert(0, 'username')
    username_box.pack()
     
    # adds password entry widget and defines its properties
    password_box = _tk.Entry(root, show='*')
    password_box.insert(0, 'password')
    password_box.bind('<Return>', ok_pressed)
    password_box.pack()
     
    # adds login button and defines its properties
    login_btn = _tk.Button(root, text='OK', command=ok_pressed)
    login_btn.bind('<Return>', ok_pressed)
    login_btn.pack()
    
    root.mainloop()
    return tuple(credentials)



def show_message(message=''):
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
    pass


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
    # We use a list of length 1 to pass selected_choice by reference.
    selected_choice = [-2]

    root = _create_root()

    def return_choice(ichoice):
        selected_choice[0] = ichoice
        root.quit()

#    root.title(title)
    lbl = _tk.Label(root, text=message)
    lbl.pack()

    ichoice = 0
    for choice in choices:
        btn = _tk.Button(root,
                         text=choice,
                         command=_partial(return_choice, ichoice))
        btn.pack()
        ichoice = ichoice + 1

    root.mainloop()
    _destroy_root(root)

    return(selected_choice[0])
