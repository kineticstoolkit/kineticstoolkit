#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provide a tk root window for the gui module.

This very simple module exists so that modifications to the gui module won't
call a new tk root window on module reload.
"""

import tkinter


def create_root():
    """Create a tkinter root window."""
    root = tkinter.Tk()
    # Ensure the window is not created as a tab on macOS
    root.resizable(width=False, height=False)
    return root


root = create_root()
