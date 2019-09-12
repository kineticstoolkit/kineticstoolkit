#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 07:54:15 2019

@author: felix
"""
import os
import subprocess
import ktk

def explore():
    """Open a Windows Explorer or macOS Finder window in the current folder."""
    folder_name = os.getcwd()
    if ktk.config['IsPC'] is True:
        subprocess.call(['explorer', folder_name])
    elif ktk.config['IsMac'] is True:
        subprocess.call(['open', folder_name])
    else:
        raise NotImplementedError('This function is only implemented on'
                                  'Windows and macOS.')


def terminal():
    """Open a terminal in the current folder."""
    folder_name = os.getcwd()
    if ktk.config['IsPC'] is True:
        subprocess.call(['explorer', folder_name])
    elif ktk.config['IsMac'] is True:
        subprocess.call([
                'osascript',
                '-e',
                'tell application "Terminal" to do script "cd ' +
                        str(folder_name) + '"'])
        subprocess.call([
                'osascript',
                '-e',
                'tell application "Terminal" to activate'])
    else:
        raise NotImplementedError('This function is only implemented on'
                                  'Windows and macOS.')
