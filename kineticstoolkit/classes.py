#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020-2025 Félix Chénier

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# mypy: ignore-errors
# MyPy complains with it, but tests tell that it works.
"""
Provides lists and dictionaries with callback functions.

Provides the MonitoredList and MonitoredDict classes, which derive from
list and dict respectively, and provide callback functions for
appending, extending, removing items, etc.

"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2025 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


from collections import UserList, UserDict


class MonitoredList(list):
    """A python list with a configurable callback on modification."""

    def __init__(self, *args, callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.callback = callback

    def _trigger_callback(self, action, *args):
        if self.callback:
            self.callback(action, *args)

    def __repr__(self):
        return super().__repr__()

    def __str__(self):
        return super().__str__()

    def append(self, item):
        super().append(item)
        self._trigger_callback("append", self, item)

    def extend(self, other):
        super().extend(other)
        self._trigger_callback("extend", self, other)

    def insert(self, index, item):
        super().insert(index, item)
        self._trigger_callback("insert", self, index, item)

    def remove(self, item):
        super().remove(item)
        self._trigger_callback("remove", self, item)

    def pop(self, index=-1):
        item = super().pop(index)
        self._trigger_callback("pop", self, index)
        return item

    def clear(self):
        super().clear()
        self._trigger_callback("clear", self)

    def __setitem__(self, index, item):
        super().__setitem__(index, item)
        self._trigger_callback("setitem", self, index, item)

    def __delitem__(self, index):
        super().__delitem__(index)
        self._trigger_callback("delitem", self, index)


class MonitoredDict(dict):
    """A python dict with a configurable callback on modification."""

    def __init__(self, *args, callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.callback = callback

    def _trigger_callback(self, action, *args):
        if self.callback:
            self.callback(action, *args)

    def __repr__(self):
        return super().__repr__()

    def __str__(self):
        return super().__str__()

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._trigger_callback("setitem", self, key, value)

    def __delitem__(self, key):
        super().__delitem__(key)
        self._trigger_callback("delitem", self, key)

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self._trigger_callback("update", self, args, kwargs)

    def pop(self, key, default=None):
        value = super().pop(key, default)
        self._trigger_callback("pop", self, key)
        return value

    def popitem(self):
        item = super().popitem()
        self._trigger_callback("popitem", self)
        return item

    def clear(self):
        super().clear()
        self._trigger_callback("clear", self)


def list_to_monitored_list(value: list, callback) -> MonitoredList:
    """
    Convert a list and its contents to a MonitoredList.

    Recursively navigates through the list and converts any inner list to
    a MonitoredList, and any inner dict to a MonitoredDict.

    Sets are ignored, but this could change if needed in the future.

    Parameters
    ----------
    value
        The list to be converted

    callback
        The callback function to be called when the content is modified.

    Returns
    -------
    MonitoredList
        The converted list.

    Caution
    -------
    The callback function will most likely be called while performing the
    conversion.

    """
    output = []
    for item in value:
        if isinstance(item, list):
            output.append(list_to_monitored_list(item, callback))
        elif isinstance(item, dict):
            output.append(dict_to_monitored_dict(item, callback))
        else:
            output.append(item)
    return MonitoredList(output, callback=callback)


def dict_to_monitored_dict(value: dict, callback) -> MonitoredDict:
    """
    Convert a dict and its contents to a MonitoredDict.

    Recursively navigates through the dict and converts any inner list to
    a MonitoredList, and any inner dict to a MonitoredDict.

    Sets are ignored, but this could change if needed in the future.

    Parameters
    ----------
    value
        The dict to be converted

    callback
        The callback function to be called when the content is modified.

    Returns
    -------
    MonitoredDict
        The converted dict.

    """
    output = {}
    for key in value:
        contents = value[key]
        if isinstance(contents, list):
            output[key] = list_to_monitored_list(contents, callback)
        elif isinstance(contents, dict):
            output[key] = dict_to_monitored_dict(contents, callback)
        else:
            output[key] = contents
    return MonitoredDict(output, callback=callback)
