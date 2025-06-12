#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2022-2025 Félix Chénier

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
Provide functions related to exceptions.

For internal use only.

"""
import warnings

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2022-2025 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


class TimeSeriesRangeError(Exception):
    """The requested operation exceeds the TimeSeries' time range."""


class TimeSeriesEventNotFoundError(Exception):
    """The requested event occurrence was not found."""


class TimeSeriesMergeConflictError(Exception):
    """Both TimeSeries have a same data key."""


def raise_ktk_error(e) -> None:
    """Re-raise an exception with a user message on how to report this bug."""
    print(
        "==================================================================\n"
        "                     KINETICS TOOLKIT ERROR                       \n"
        "==================================================================\n"
        "An error has been encountered. This is most probably due to a bug \n"
        "in Kinetics Toolkit. We all despise bugs, and it would be very    \n"
        "nice if you could help us by reporting this bug. To do so, please \n"
        "go to https://github.com/kineticstoolkit/kineticstoolkit/issues,  \n"
        "select 'New Issue' and fill in the required information.          \n"
        "                                                                  \n"
        "Please include the whole error message, starting at the line      \n"
        "Traceback (most recent call last):                                \n"
        "down to the end in the Traceback section of the bug report.       \n"
        "                                                                  \n"
        "Sorry that you encountered this problem and thank you for helping \n"
        "us fix it.                                                        \n"
        "==================================================================\n"
        "                PLEASE READ THE ERROR MESSAGE ABOVE               \n"
        "==================================================================\n"
    )
    raise (e)


# Create a set to keep track of issued warnings
issued_warnings = set()  # type: set[tuple[str, Exception]]


def warn_once(message: str, category=UserWarning, stacklevel: int = 1) -> None:
    """Raise a warning only once."""
    if (message, category) not in issued_warnings:
        warnings.warn(message, category, stacklevel=stacklevel + 1)
        issued_warnings.add((message, category))


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
