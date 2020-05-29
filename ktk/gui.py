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
"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import matplotlib as mpl


def set_color_order(setting):
    """
    Define the standard color order for matplotlib.

    Parameters
    ----------
    setting : str or list
        Either a string or a list of colors.
        - If a string, it can be either:
            - 'default' : Default v2.0 matplotlib colors.
            - 'classic' : Default classic Matlab colors (bgrcmyk) with an added
                          'orange' at the end.
            - 'xyz' :     Same as classic but begins with rgb instead of bgr to
                          be consistent with most 3d visualization softwares.
        - If a list, it can be either a list of chars from [bgrcmyk], a list of
          hexadecimal color values, or any list supported by matplotlib's
          axes.prop_cycle rcParam.

    Returns
    -------
    None.

    """
    if isinstance(setting, str):
        if setting == 'default':
            thelist = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        elif setting == 'classic':
            thelist = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        elif setting == 'xyz':
            thelist = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'tab:orange']
        else:
            raise(ValueError('This setting is not recognized.'))
    elif isinstance(setting, list):
        thelist = setting
    else:
        raise(ValueError('This setting is not recognized.'))

    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=thelist)
