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


import numpy as np


def check_types(function, args):
    try:
        annotations = function.__annotations__
    except AttributeError:
        print("No annotation in this function")
        return

    def raise_type_error():
        raise TypeError(
            f"In function '{function.__name__}', parameter '{arg}' expects a "
            f"variable of type {expected_type}. However, this variable of "
            f"type {value_type} was provided: {value}."
        )

    for arg in args:
        if arg in annotations:
            value = args[arg]
            value_type = str(type(value)).split("'")[1]
            expected_type = annotations[arg]

            if expected_type == "float":
                # Just ensure that it's equal to its float version
                try:
                    if value != float(value):
                        raise_type_error()
                except ValueError:
                    raise_type_error()

            if expected_type == "int" and not isinstance(value, int):
                raise_type_error()

            if expected_type == "bool" and not isinstance(value, bool):
                raise_type_error()

            if expected_type == "str" and not isinstance(value, str):
                raise_type_error()

            if expected_type.lower().startswith("list") and not isinstance(
                value, list
            ):
                raise_type_error()

            if expected_type.lower().startswith("dict") and not isinstance(
                value, dict
            ):
                raise_type_error()

            if expected_type == "TimeSeries":
                if "TimeSeries" not in value_type:
                    raise_type_error()
                value._check_well_typed()
