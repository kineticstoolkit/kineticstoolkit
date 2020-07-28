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
Provide miscelleanous helper functions to the user.

These functions are accessible from ktk's toplevel namespace
(i.e., ktk.explore).
"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import ktk.config

import os
import sys
import subprocess
import webbrowser as _webbrowser


def explore(folder_name: str = '') -> None:
    """
    Open an Explorer window (on Windows) or a Finder window (on macOS)

    Parameters
    ----------
    folder_name
        Optional. The name of the folder to open the window in. Default is the
        current folder.

    """
    if not folder_name:
        folder_name = os.getcwd()

    if ktk.config.is_pc is True:
        os.system(f'start explorer {folder_name}')

    elif ktk.config.is_mac is True:
        subprocess.call(['open', folder_name])

    else:
        raise NotImplementedError('This function is only implemented on'
                                  'Windows and macOS.')


def terminal(folder_name: str = '') -> None:
    """
    Open a terminal window.

    Parameters
    ----------
    folder_name
        The name of the folder to open the terminal window in. Default is the
        current folder.

    Returns
    -------
    None.
    """
    if not folder_name:
        folder_name = os.getcwd()

    if ktk.config.is_pc is True:
        os.system(f'cmd /c start /D {folder_name} cmd')

    elif ktk.config.is_mac is True:
        subprocess.call([
                'osascript',
                '-e',
                """tell application "Terminal" to do script "cd '""" +
                    str(folder_name) + """'" """])
        subprocess.call([
                'osascript',
                '-e',
                'tell application "Terminal" to activate'])
    else:
        raise NotImplementedError('This function is only implemented on'
                                  'Windows and macOS.')


def update() -> None:
    """
    Update ktk to the last available version - Not for the public version.

    If ktk was installed using pip (default when using the public open-source
    version), then update ktk using pip instead:

        pip upgrade ktk

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    if ktk.config.is_pc:
        current_dir = os.getcwd()
        os.chdir(ktk.config.root_folder)
        print(subprocess.check_output(
            ['git', 'pull', 'origin', 'master']).decode(sys.getdefaultencoding()))
        os.chdir(current_dir)

    if ktk.config.is_mac:
        subprocess.call([
            'osascript',
            '-e',
            'tell application "Terminal" to activate'])
        subprocess.call([
            'osascript',
            '-e',
            ('tell application "Terminal" to do script "cd \'' +
             ktk.config.root_folder +
             '\'; git pull origin master"')])


def tutorials() -> None:
    """
    Open the KTK tutorials in a web browser.

    Usage: ktk.tutorials()
    """
    _webbrowser.open('file:///' + ktk.config.root_folder +
                     '/tutorials/index.html',
                     new=2)
