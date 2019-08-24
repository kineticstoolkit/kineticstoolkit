#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module that provides simple GUI functions.

This module is not to be called directly, but rather accessed using separate
python instances using the ktk.gui module.

Author: Félix Chénier

Started on June 2019
"""

# Imports
import tkinter as tk
import tkinter.filedialog as filedialog

from functools import partial

import argparse
import subprocess
import time


def create_window(title='KTK', width=800, height=400):
    root = tk.Tk()
    # Ensure the window is not created as a tab on macOS
    root.resizable(width=False, height=False)
    root.title(title)

    # Make it transparent while we modify it.
    root.wm_attributes("-alpha", 0)
    #root.update()

    # # Ensure the window is not closable by user
    # def _on_closing():
    #     pass
    # root.protocol("WM_DELETE_WINDOW", _on_closing)

    # Set topmost
    root.attributes('-topmost', True)
    # root.update()
    # root.attributes('-topmost', False)

    return root


def get_window_geometry(root):
    """Return the root geometry as the tuple (width, height, left, top)."""
    root.update()
    geometry = root.geometry()

    size = geometry[:geometry.find('+')]
    position = geometry[geometry.find('+')+1:]

    width = int(size[:size.find('x')])
    height = int(size[geometry.find('x')+1:])

    left = int(position[:position.find('+')])
    top = int(position[position.find('+')+1:])

    return (width, height, left, top)


def center_window(root):
    """Center the root window on screen."""
    (width, height, left, top) = get_window_geometry(root)
    root.geometry('%dx%d+%d+%d' % (
                  width, height,
                  root.winfo_screenwidth()/2-width/2,
                  root.winfo_screenheight()/2-height/2))


def show_window(root):
    """Show the window (remove transparency)"""
    root.wm_attributes("-alpha", 1)


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


def get_credentials(title, message):
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
        root.quit()

    credentials = ['', '']

    root = create_window(title=title)
    instruction_label = tk.Label(root, text=message)
    instruction_label.pack()
    username_box = tk.Entry(root)
    username_box.insert(0, 'username')
    username_box.pack()

    # adds password entry widget and defines its properties
    password_box = tk.Entry(root, show='*')
    password_box.insert(0, 'password')
    password_box.bind('<Return>', ok_pressed)
    password_box.pack()

    # adds login button and defines its properties
    login_btn = tk.Button(root, text='OK', command=ok_pressed)
    login_btn.bind('<Return>', ok_pressed)
    login_btn.pack()

    center_window(root)
    show_window(root)
    root.mainloop()

    print(credentials[0])
    print(credentials[1])


def button_dialog(title='', message='Please select an option.',
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
    selected_choice = [-1]  # Default if we click close.

    root = create_window()

    def return_choice(ichoice):
        selected_choice[0] = ichoice
        root.quit()

    root.title(title)
    lbl = tk.Label(root, text=message)
    lbl.pack()

    ichoice = 0
    for choice in choices:
        btn = tk.Button(root,
                        text=choice,
                        command=partial(return_choice, ichoice))
        btn.pack()
        ichoice = ichoice + 1

    center_window(root)
    show_window(root)
    root.mainloop()

    print(selected_choice[0])


def get_folder(title, initial_folder):
    root = create_window()
    root.withdraw()
    print(filedialog.askdirectory(title=title, initialdir=initial_folder))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Run interactive gui dialogs.')
    parser.add_argument('dialog', help='buttondialog')
    parser.add_argument('title', help='Title of the dialog')
    parser.add_argument('args', help='Arguments', nargs="*")
    args = parser.parse_args()

    if args.dialog == 'button_dialog':
        button_dialog(args.title, args.args[0], args.args[1:])
    elif args.dialog == 'get_credentials':
        get_credentials(args.title, args.args[0])
    elif args.dialog == 'get_folder':
        get_folder(args.title, args.args[0])
