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
Provide decorators for Kinetics Toolkit's functions.

The following decorator can be used on each Kinetics Toolkit's's function:

    - @stable: Documented, stable function in a release.

    - @experimental: Documented, experimental function in a release.

    - @deprecated: Documented but deprecated function in a release.

    - @unstable: Documented only in the development version.

    - @dead: Undocumented, deprecated function in the development version.

    - @private: Undocumented, for private use by the module only.

    Each of these decorators take a list as argument, and adds the function
    to the list if this function should be added to the __dir__ magic function.

"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


from functools import wraps
import warnings
import textwrap
import kineticstoolkit.config


experimental_docstring = """
Warning
-------
Experimental function.

This function's signature and default behaviour are not completely settled
and may change slightly in the future.
"""

deprecated_docstring = """
Warning
-------
Deprecated function.

This function will be removed in a future release. Please consult the
function's detailed help to see replacement options.
"""

unstable_docstring = """
Warning
-------
Unreleased function.

This function is currently being developped and has not been released yet.
Although it should be working, please do not use this function in stable
production work since it may not be fully tested, it may change signature and
behaviour drastically or even be deleted.
"""

dead_docstring = """
Warning
-------
Deprecated function.

This function was removed from the documentation but is still temporarily
accessible. Its code will be completely removed in a future release.
"""

private_docstring = """
Warning
-------
Private function.

This function should be used by Kinetics Toolkit only. Do not base your work
on this function.
"""


def _inject_in_docstring(docstring: str, text: str) -> str:
    """Inject a string into the top of a docstring, after line 1."""
    if docstring == '' or docstring is None:
        return text
    result = []
    splitted = textwrap.dedent(docstring).split('\n')
    first_line_done = False
    for line in splitted:
        if not first_line_done:
            if len(line) != 0:
                first_line_done = True
                result.append(line)
                for text_line in text.split('\n'):
                    result.append(text_line)
            else:
                result.append(line)
        else:
            result.append(line)
    return '\n'.join(result)


def stable(listing):
    """
    Decorate stable Kinetics Toolkit's functions.

    Adds this function to the main documentation.

    Parameter listing is a list of attributes of the module that will be
    returned by the module's or class' __dir__ function.
    """
    # Define a new decorator, named "decorator", to return
    def decorator(func):
        # Ensure the decorated function keeps its metadata
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Call the function being decorated and return the result
            return func(*args, **kwargs)
        listing.append(func.__name__)
        return wrapper
    # Return the new decorator
    return decorator


def experimental(listing):
    """
    Decorate experimental Kinetics Toolkit's functions.

    Adds this function to the main documentation with no warning for the
    moment. Also adds a warning section to its docstring.

    Parameter listing is a list of attributes of the module that will be
    returned by the module's or class' __dir__ function.
    """
    # Define a new decorator, named "decorator", to return
    def decorator(func):
        # Ensure the decorated function keeps its metadata
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Call the function being decorated and return the result
            return func(*args, **kwargs)
        listing.append(func.__name__)
        wrapper.__doc__ = _inject_in_docstring(
            func.__doc__, experimental_docstring)
        return wrapper
    # Return the new decorator
    return decorator


def deprecated(listing):
    """
    Decorate deprecated Kinetics Toolkit's functions.

    Adds this function to the main documentation and generates a FutureWarning
    on use. Also adds a warning section to its docstring.

    Parameter listing is a list of attributes of the module that will be
    returned by the module's or class' __dir__ function.
    """
    # Define a new decorator, named "decorator", to return
    def decorator(func):
        # Ensure the decorated function keeps its metadata
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn("This function is deprecated and will be removed "
                          "in a future version of Kinetics Toolkit. Please "
                          "consult the API for replacement solutions.",
                          FutureWarning)
            # Call the function being decorated and return the result
            return func(*args, **kwargs)
        listing.append(func.__name__)
        wrapper.__doc__ = _inject_in_docstring(
            func.__doc__, deprecated_docstring)
        return wrapper
    # Return the new decorator
    return decorator


def unstable(listing):
    """
    Decorate unstable Kinetics Toolkit's functions.

    Adds this function to the main documentation only if
    kineticstoolkit.config.version == 'master'.  Also adds a warning section
    to its docstring.

    Parameter listing is a list of attributes of the module that will be
    returned by the module's or class' __dir__ function.
    """
    # Define a new decorator, named "decorator", to return
    def decorator(func):
        # Ensure the decorated function keeps its metadata
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Call the function being decorated and return the result
            return func(*args, **kwargs)
        if kineticstoolkit.config.version == 'master':
            listing.append(func.__name__)
        wrapper.__doc__ = _inject_in_docstring(
            func.__doc__, unstable_docstring)
        return wrapper
    # Return the new decorator
    return decorator


def dead(listing):
    """
    Decorate dead Kinetics Toolkit's functions.

    Does not add this function to the main documentation and generates a
    FutureWarning on use. Also adds a warning section to its docstring.

    Parameter listing is a list of attributes of the module that will be
    returned by the module's or class' __dir__ function.
    """
    # Define a new decorator, named "decorator", to return
    def decorator(func):
        # Ensure the decorated function keeps its metadata
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn("This function is deprecated and will be removed "
                          "in a future version of Kinetics Toolkit. Please "
                          "consult the API for replacement solutions.",
                          FutureWarning)
            # Call the function being decorated and return the result
            return func(*args, **kwargs)
        wrapper.__doc__ = _inject_in_docstring(
            func.__doc__, dead_docstring)
        return wrapper
    # Return the new decorator
    return decorator


def private(listing):
    """
    Decorate private Kinetics Toolkit's functions.

    Does not add this function to the main documentation.
    Also adds a warning section to its docstring.

    Parameter listing is a list of attributes of the module that will be
    returned by the module's or class' __dir__ function.
    """
    # Define a new decorator, named "decorator", to return
    def decorator(func):
        # Ensure the decorated function keeps its metadata
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Call the function being decorated and return the result
            return func(*args, **kwargs)
        wrapper.__doc__ = _inject_in_docstring(
            func.__doc__, private_docstring)
        return wrapper
    # Return the new decorator
    return decorator
