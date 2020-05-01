#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 07:54:15 2019

@author: felix
"""
import os
import subprocess
import ktk
import webbrowser as _webbrowser


def explore(folder_name=''):
    """
    Open an Explorer window (on Windows) or a Finder window (on macOS)

    Parameters
    ----------
    folder_name : str (optional)
        The name of the folder to open the window in. Default is the current
        folder.

    Returns
    -------
    None.
    """
    if not folder_name:
        folder_name = os.getcwd()

    if ktk.config['IsPC'] is True:
        os.system(f'start explorer {folder_name}')

    elif ktk.config['IsMac'] is True:
        subprocess.call(['open', folder_name])

    else:
        raise NotImplementedError('This function is only implemented on'
                                  'Windows and macOS.')


def terminal(folder_name=''):
    """
    Open a terminal window.

    Parameters
    ----------
    folder_name : str (optional)
        The name of the folder to open the terminal window in. Default is the
        current folder.

    Returns
    -------
    None.
    """
    if not folder_name:
        folder_name = os.getcwd()

    if ktk.config['IsPC'] is True:
        os.system(f'cmd /c start /D {folder_name} cmd')

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


def update():
    """
    Update KTK to the last available version.

    KTK needs to be installed as a git repository. This is the case with the
    default installation method using install.py.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    current_dir = os.getcwd()
    os.chdir(ktk.config['RootFolder'])
    print(subprocess.check_output(['git', 'pull', 'origin', 'master']).decode('ascii'))
    os.chdir(current_dir)


def tutorials():
    """
    Open the KTK tutorials in a web browser.

    Usage: ktk.tutorials()
    """
    _webbrowser.open('file:///' + ktk.config['RootFolder'] +
                     '/tutorials/index.html',
                     new=2)
