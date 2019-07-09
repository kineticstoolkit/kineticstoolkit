#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Felix Chenier
@name: ktkutils.py
@date: June 2019
@purpose: This module provides utilitary functions.
"""

# Imports
try:
    import tkinter as tk #Python 3
except:
    import Tkinter as tk #Python 2

from functools import partial


def uitakecontrol(bool_arg):
    """
    Take or release control of the GUI main loop.
    This function should be called before and after any function that sets a
    GUI main loop, for a correct integration with ipython and spyder.
    
    uitakecontrol(True): runs "%gui" without argument.
    uitakecontrol(False): runs "%gui qt".

    """
    if bool_arg == True:
        try:
            ipy = get_ipython()
            #ipy.magic("gui")
        except:
            print("Cannot take GUI control")
    else:
        try:
            ipy = get_ipython()
            #ipy.magic("gui qt")
        except:
            pass

def uibuttonsdialog(str_title, str_message, list_choices):
    """
    Creates a topmost dialog window with a selection of buttons.
        
    Inputs:
        str_title: Title of the window
        str_message: Message to show
        list_choices: Buttons captions. For example: ['OK', 'Cancel']
    
    Output:
        The button number (0 = First button, 1 = Second button, etc. If the
        user closes the window instead of clicking a button, a value of -1 is
        returned.
    """
    
    #uitakecontrol(True)
    
    root = tk.Tk()
    
    selectedchoice = [0, '']    # Hack to get a reference in returnchoice
    
    def returnchoice(ichoice):
        selectedchoice[0] = ichoice
        root.destroy()
        root.quit()

    root.protocol("WM_DELETE_WINDOW", partial(returnchoice, -1)) # Close button returns -1

    root.title(str_title)    
    lbl = tk.Label(root, text = str_message)
    lbl.grid(row = 0, columnspan = len(list_choices))
    
    ichoice = 0
    for choice in list_choices:
        btn = tk.Button(root, 
                        text = choice,
                        command = partial(returnchoice, ichoice))
        btn.grid(row = 1, column = ichoice)
        ichoice = ichoice + 1
            
    root.wm_attributes("-topmost", 1) # Force topmost    
    root.mainloop()
    
    #uitakecontrol(False)
    
    return(selectedchoice[0])
