#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2022 Félix Chénier

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provide functions related to exceptions. For internal use only."""
from __future__ import annotations


__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2022 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


from typing import Any
import numpy as np


class TimeSeriesRangeError(Exception):
    """The requested operation exceeds the TimeSeries' time range."""


class TimeSeriesEventNotFoundError(Exception):
    """The requested event occurrence was not found."""


def simplify_annotation_string(string: str) -> str:
    """
    Simplify an annotation to keep only the outer definitions.

    Example
    -------
    >>> string = "list[str] | dict[str, list[int] | int] | None"
    >>> simplify_annotation_string(string)
    'list|dict|None'
    """
    n_open_brackets = 0
    out_list = []
    for c in string:
        if c == "[":
            n_open_brackets += 1
        if n_open_brackets == 0 and c != " ":
            out_list.append(c)
        if c == "]":
            n_open_brackets -= 1
    return "".join(out_list)


def check_types(function, args: dict[str, Any]):
    """
    Check that a function's arguments are of correct type.

    Parameters
    ----------
    function
        The function to be checked.

    args
        Usually, locals(). Those are the arguments of the function.

    Raises
    ------
    TypeError
        If one of the arguments is of wrong type.

    Warning
    -------
    This function does not check into constructs such as lists or dicts.
    It does not check for ArrayLike. It may help users finding some bugs in
    their scripts, but it's really not
    foulproof.

    """
    try:
        annotations = function.__annotations__
    except AttributeError:
        print("No annotation in this function")
        return

    def raise_type_error():
        raise TypeError(
            f"In function '{function.__name__}', parameter '{arg}' expects a "
            f"variable of type '{expected_type}'. However, this variable of "
            f"type '{value_type}' was provided: {value}."
        )

    for arg in args:
        if arg in annotations:
            value = args[arg]
            value_type = str(type(value)).split("'")[1]
            expected_type = simplify_annotation_string(annotations[arg])

            ok = False
            for one_expected_type in expected_type.split("|"):
                if one_expected_type == "float":
                    # Just ensure that it's equal to its float version
                    try:
                        if np.isnan(value) or np.isinf(value):
                            ok = True
                        elif value == float(value):
                            ok = True
                    except Exception:
                        pass

                elif one_expected_type == "int":
                    if isinstance(value, int) or isinstance(value, np.integer):
                        ok = True

                elif one_expected_type == "bool":
                    if isinstance(value, bool):
                        ok = True

                elif one_expected_type == "str":
                    if isinstance(value, str):
                        ok = True

                elif one_expected_type == "list":
                    if isinstance(value, list):
                        ok = True

                elif one_expected_type == "dict":
                    if isinstance(value, dict):
                        ok = True

                elif one_expected_type == "tuple":
                    if isinstance(value, tuple):
                        ok = True

                elif one_expected_type == "set":
                    if isinstance(value, set):
                        ok = True

                elif one_expected_type == "TimeSeries":
                    if "TimeSeries" in value_type:
                        ok = True

                else:  # expected type not in list, ok by default.
                    ok = True

            if not ok:
                raise_type_error()


if __name__ == "__main__":  # pragma: no cover
    import doctest
    import numpy as np

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
