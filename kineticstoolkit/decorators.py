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

    - @unstable:
        Documented only in the development version.

    - @deprecated(since: str, until: str, details: str):
        Documented but deprecated function in a release.

    - @dead:
        Undocumented, deprecated function in the development version.

    Each of these decorators add the _include_in_dir property to the decorated
    function. The provided function ``directory`` looks at these properties
    to return a custom __dir__ to Kinematics Toolkit's classes. See such class
    for example.

"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


from functools import wraps
import warnings
import textwrap
import kineticstoolkit.config
from typing import Dict, List, Any


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


def deprecated(since: str, until: str, details: str):
    """
    Decorate deprecated Kinetics Toolkit's functions.

    Generates a FutureWarning and adds a warning section to its docstring.
    These functions are included in API documentation.

    """
    def real_decorator(func):
        func_name = func.__name__
        string = (f"The function {func_name} is deprecated since "
                  f"{since} and is scheduled to be removed "
                  f"in {until}. {details}")

        # Ensure the decorated function keeps its metadata
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(string, category=FutureWarning, stacklevel=2)
            # Call the function being decorated and return the result
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                return func(*args, **kwargs)
        wrapper.__doc__ = _inject_in_docstring(
            func.__doc__, f"\nWarning\n-------\n{string}")
        wrapper._is_deprecated = True
        return wrapper
    return real_decorator


def unstable(func):
    """
    Decorate unstable Kinetics Toolkit's functions.

    if kineticstoolkit.config.version == 'master':
        Adds this function to the main documentation.
    else:
        Generate a KTKUnstableWarning on use.

    Also adds a warning section to its docstring.

    """
    func_name = func.__name__
    string = (f"The function {func_name} is unstable, which means it may not "
              f"be tested or settled yet. Please avoid using this function "
              f"in production code.")

    # Ensure the decorated function keeps its metadata
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Call the function being decorated and return the result
        return func(*args, **kwargs)

    wrapper._is_unstable = True

    wrapper.__doc__ = _inject_in_docstring(
        func.__doc__, f"\nWarning\n-------\n{string}")
    return wrapper


def dead(since: str, until: str, details=''):
    """
    Decorate dead Kinetics Toolkit's functions.

    Does not add this function to the main documentation and generates a
    FutureWarning on use. Also adds a warning section to its docstring.

    Parameter listing is a list of attributes of the module that will be
    returned by the module's or class' __dir__ function.
    """
    def real_decorator(func):
        func_name = func.__name__
        string = (f"The function {func_name} is deprecated since "
                  f"{since} and is scheduled to be removed "
                  f"in {until}. {details}")

        # Ensure the decorated function keeps its metadata
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(string, category=FutureWarning, stacklevel=2)
            # Call the function being decorated and return the result
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                return func(*args, **kwargs)
        wrapper.__doc__ = _inject_in_docstring(
            func.__doc__, f"\nWarning\n-------\n{string}")
        wrapper._is_dead = True
        return wrapper
    return real_decorator


def directory(module_locals: Dict[str, Any]) -> List[str]:
    """
    Return the module's public directory for dir function.

    Parameters
    ----------
    module_locals
        The module's locals as generated by the locals() function.

    Returns
    -------
    List of public objects.

    """
    dir_ = []
    for key in module_locals:

        if key.startswith('_'):
            continue

        try:
            if (
                    '_is_unstable' in module_locals[key].__dict__
                    and module_locals[key].__dict__['_is_unstable'] is True
                    and kineticstoolkit.config.version != 'master'
            ):
                continue

            if (
                    '_is_dead' in module_locals[key].__dict__
                    and module_locals[key].__dict__['_is_dead'] is True
            ):
                continue

        except AttributeError:
            continue

        dir_.append(key)

    return dir_


def __dir__():
    return []
