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

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"

"""
ktk._tools
----------
This module provides various tools to the user.

"""

import os
import subprocess
import ktk


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

