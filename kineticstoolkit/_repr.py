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
kineticstoolkit._repr.py
------------------------

Format the console output of dictionaries and classes with attributes.

This module formats the console output of dict in ipython, so that instead
of just using repr(), it displays a nicer list of keys with abbreviated
values if required, so that there is a maximum of one key per line. This is
very useful for nested dicts, since their repr() representation is recursive
and becomes unmanagable when the dict becomes larger.

It also provides helper functions to nicely format the repr() of data classes.

"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import numpy as np


def _format_dict_entries(value, quotes=True):
    """
    Format a dict nicely on screen.

    This function makes every element of a dict appear on a separate line,
    with each keys right aligned:
        {
           'key1' : value1
           'key2' : value2
        'longkey' : value3
        }

    Parameters
    ----------
    value : dict
        The dict that we want to show on screen.
    quotes : bool (optional)
        Indicated if the keys must be surrounded by quotes. The default is
        True.

    Returns
    -------
    A string that should be shown by the __repr__ method.

    """
    out = ''

    the_keys = value.keys()
    if len(the_keys) > 0:

        # Find the widest field name
        the_max_length = 0
        for the_key in the_keys:
            the_max_length = max(the_max_length, len(repr(the_key)))

        max_length_to_show = 77 - the_max_length

        for the_key in sorted(the_keys):

            # Print the key
            if quotes is False and isinstance(the_key, str):
                to_show = repr(the_key)[1:-1]  # Remove quotes
            else:
                to_show = repr(the_key)

            out += (to_show.rjust(the_max_length + 6) + ': ')  # +6 to tab

            # Print the value
            if isinstance(value[the_key], dict):
                out += '<dict with ' + str(len(value[the_key])) + ' entries>'
            elif isinstance(value[the_key], list):
                out += '<list of ' + str(len(value[the_key])) + ' items>'
            elif isinstance(value[the_key], np.ndarray):
                out += '<array of shape ' + str(np.shape(value[the_key])) + '>'
            else:
                to_show = repr(value[the_key])

                # Remove line breaks and multiple-spaces
                to_show = ' '.join(to_show.split())
                if len(to_show) <= max_length_to_show:
                    out += to_show
                else:
                    out += (to_show[0:max_length_to_show - 3] + '...')

            if the_key != sorted(the_keys)[-1]:
                out += ','

            # Print the ending } if needed
            out += '\n'

    return out


def _format_class_attributes(obj):
    """
    Format a class that has attributes nicely on screen.

    This class lists every attribute of a class on a separate line, using the
    _format_dict_entries function:

        ClassName with attributes:
           'attribute1' : value1
           'attribute2' : value2
        'longattribute' : value3

    Parameters
    ----------
    obj: Any
        The class instance.

    Returns
    -------
    A string that should be shown by the __repr__ method.

    """
    # Return the type of class (header)
    class_name = type(obj).__name__
    out = class_name + ' with attributes:\n'

    # Return the list of attributes
    out += _format_dict_entries(obj.__dict__, quotes=False)
    return out


def _ktk_format_dict(value, p, cycle):
    """Format a dict nicely on screen in ipython."""
    try:
        get_ipython()

        if cycle:
            p.pretty("...")
        else:
            p.text('{\n')
            p.text(_format_dict_entries(value))
            p.text('}')

    except:
        p.text(repr(value))
