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
class KTKBaseException(Exception):
    """Base class for KTK's exceptions."""

class TimeSeriesShapeError(KTKBaseException):
    """Raised when the TimeSeries data or time has a dimension problem."""

class TimeSeriesTypeError(KTKBaseException):
    """Raised when the TimeSeries' attributes are of wrong type."""

class TimeSeriesEventNotFoundError(KTKBaseException):
    """Raised when an occurrence of a TimeSeries event could be found."""

class KTKValueError(KTKBaseException):
    """Raised as an equivalent of standard ValueError, but aimed to users."""
    
class KTKIndexError(KTKBaseException):
    """Raised as an equivalent of standard IndexError, but aimed to users."""

class KTKKeyError(KTKBaseException):
    """Raised as an equivalent of standard KeyError, but aimed to users."""

class KTKError(KTKBaseException):
    """Raised when an unexpected error happened, mostly due to a bug in KTK."""
