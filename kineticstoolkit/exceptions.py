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

"""Provide error classes to generate meaningful error messages to users."""

from __future__ import annotations


__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"

# Exceptions


class TimeSeriesShapeError(ValueError):
    """Raised when the shape TimeSeries data and time mismatch."""


class TimeSeriesTypeError(TypeError):
    """Raised when the TimeSeries' attributes are of wrong type."""


class TimeSeriesEmptyTimeError(ValueError):
    """Raised when the TimeSeries' time attribute is empty."""


class TimeSeriesEmptyDataError(ValueError):
    """Raised when the TimeSeries has no data."""


class TimeSeriesEventNotFoundError(ValueError):
    """Raised when an occurrence of a TimeSeries event could be found."""


class TimeSeriesNonIncreasingTimeError(ValueError):
    """Raised when the TimeSeries' time is not continuously increasing."""
