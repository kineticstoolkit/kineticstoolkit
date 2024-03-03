#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2022-2024 Félix Chénier

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

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2022-2024 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


class TimeSeriesRangeError(Exception):
    """The requested operation exceeds the TimeSeries' time range."""


class TimeSeriesEventNotFoundError(Exception):
    """The requested event occurrence was not found."""


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
