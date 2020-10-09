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

This module is not to be called directly, but rather accessed using separate
python instances using the ktk.gui module. The reason this uses another
instance of python is to separate the GUI event loops: the "normal" ktk
environment usually uses Qt5, but these simple GUI functions use tkinter.
"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


# Imports
import tkinter as tk
import tkinter.filedialog as filedialog
from functools import partial
import argparse
import time
import os

palette = {
    'fg': 'black',
    'bg': 'white',
}


def create_window(title='KTK'):
    root = tk.Tk()
    # Ensure the window is not created as a tab on macOS
    root.resizable(width=False, height=False)
    root.title(title)
    path_to_ico = (os.path.abspath(os.path.dirname(__file__)) +
                   '/logo_hires.png')
    root.iconphoto(False, tk.PhotoImage(file=path_to_ico))

    # Make it transparent while we modify it.
    root.wm_attributes("-alpha", 0)
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

def top_window(root):
    """Move the root window to the top of the screen."""
    (width, height, left, top) = get_window_geometry(root)
    root.geometry('%dx%d+%d+%d' % (
                  root.winfo_screenwidth(), height,
                  0, 0))


def right_window(root):
    """Move the root window to the right of the streen."""
    (width, height, left, top) = get_window_geometry(root)
    root.geometry(f'{width}x{root.winfo_screenheight()-200}+'
                  f'{root.winfo_screenwidth() - width}+100')

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

    # Logo
    logo = tk.PhotoImage(file=os.path.abspath(os.path.dirname(__file__)) +
                         '/logo.png')
    image = tk.Label(root, image=logo, **palette)
    image.pack(fill=tk.X)

    instruction_label = tk.Label(root, text=message, **palette)
    instruction_label.pack()

    username_box = tk.Entry(root, highlightbackground=palette['bg'])
    username_box.insert(0, 'username')
    username_box.pack(fill=tk.X)

    # adds password entry widget and defines its properties
    password_box = tk.Entry(root, show='*', highlightbackground=palette['bg'])
    password_box.insert(0, 'password')
    password_box.bind('<Return>', ok_pressed)
    password_box.pack(fill=tk.X)

    # adds login button and defines its properties
    login_btn = tk.Button(root, text='OK', command=ok_pressed,
                          highlightbackground=palette['bg'])
    login_btn.bind('<Return>', ok_pressed)
    login_btn.pack(fill=tk.X)

    # Fill the rest of the window
    lbl = tk.Label(root, **palette)
    lbl.pack(expand=True, fill=tk.BOTH)

    right_window(root)
    show_window(root)
    root.mainloop()

    print(credentials[0])
    print(credentials[1])


def message(title='Title', message='Message', flagfile='/dev/null'):
    """
    Shows a tkinter message window until a file doesn't exist anymore.
    """

    root = create_window()

    # Set topmost
    root.attributes('-topmost', True)

    root.title(title)
    root.minsize(200, 0)

    # Logo
    logo = tk.PhotoImage(file=os.path.abspath(os.path.dirname(__file__)) +
                         '/logo.png')
    image = tk.Label(root, image=logo, **palette)
    image.pack(fill=tk.X)

    # Text
    lbl = tk.Label(root, text=message, **palette)
    lbl.pack(fill=tk.X)

    # Fill the rest of the figure
    lbl = tk.Label(root, **palette)
    lbl.pack(expand=True, fill=tk.BOTH)

    right_window(root)
    show_window(root)

    def check_if_file_exists(file):
        try:
            fid = open(file, 'r')
        except FileNotFoundError:
            return False
        fid.close()
        return True

    while check_if_file_exists(flagfile):
        root.update()
        time.sleep(0.2)


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
    The button number (0 = First button, 1 = Second button, etc). If the
    user closes the window instead of clicking a button, a value of -1 is
    returned.

    """
    # We use a list of length 1 to pass selected_choice by reference.
    selected_choice = [-1]  # Default if we click close.

    root = create_window()
    root.minsize(200, 0)

    # Set topmost
    root.attributes('-topmost', True)

    def return_choice(ichoice):
        selected_choice[0] = ichoice
        root.quit()

    root.title(title)

    # Logo
    logo = tk.PhotoImage(file=os.path.abspath(os.path.dirname(__file__)) +
                         '/logo.png')
    image = tk.Label(root, image=logo, **palette)
    image.pack(fill=tk.X)


    # Label
    lbl = tk.Label(root, text=message, **palette)
    lbl.pack(fill=tk.X)

    ichoice = 0
    for choice in choices:
        btn = tk.Button(root,
                        text=choice,
                        height=2,
                        highlightbackground=palette['bg'],
                        command=partial(return_choice, ichoice))
        btn.pack(fill=tk.X)
        ichoice = ichoice + 1

    # Fill the rest of the figure
    lbl = tk.Label(root, **palette)
    lbl.pack(expand=True, fill=tk.BOTH)



    right_window(root)
    show_window(root)
    root.mainloop()

    print(selected_choice[0])


def get_folder(title, initial_folder):
    root = create_window()
    root.withdraw()
    print(filedialog.askdirectory(title=title, initialdir=initial_folder))


def get_filename(title, initial_folder):
    root = create_window()
    root.withdraw()
    time.sleep(0.1)
    root.update()
    print(filedialog.askopenfilename(title=title, initialdir=initial_folder))
    time.sleep(0.1)
    root.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Run interactive gui dialogs.')
    parser.add_argument('command', help='one of: buttondialog, '
                                        'get_credentials, '
                                        'get_folder, '
                                        'get_filename, '
                                        'message')
    parser.add_argument('title', help='Title of the dialog')
    parser.add_argument('args', help='Arguments', nargs="*")
    args = parser.parse_args()

    if args.command == 'button_dialog':
        button_dialog(args.title, args.args[0], args.args[1:])
    elif args.command == 'get_credentials':
        get_credentials(args.title, args.args[0])
    elif args.command == 'get_folder':
        get_folder(args.title, args.args[0])
    elif args.command == 'get_filename':
        get_filename(args.title, args.args[0])
    elif args.command == 'message':
        message(args.title, args.args[0], args.args[1])
