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

This function is currently being developped, tested or validated and has
not been released yet. It may change signature and behaviour or even be
deleted.
"""

unstable_warning = """
The function {name} is currently being developped, tested or validated and has
not been released yet. It may change signature and behaviour or even be
deleted.
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

private_warning = """
The function {name} should be used by Kinetics Toolkit only. Do not base
your work on this function.
"""

class KTKUnstableWarning(UserWarning):
    """Warning raised when using an unstable Kinetics Toolkig function."""
    pass


class KTKPrivateWarning(UserWarning):
    """Warning raised when using a private Kinetics Toolkig function."""
    pass


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


def stable(func):
    """
    Decorate stable Kinetics Toolkit's functions.

    Adds this function to the main documentation.

    """
    # Ensure the decorated function keeps its metadata
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Call the function being decorated and return the result
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=KTKUnstableWarning)
            warnings.filterwarnings("ignore", category=KTKPrivateWarning)
            return func(*args, **kwargs)
    wrapper._include_in_dir = True
    return wrapper


def experimental(func):
    """
    Decorate experimental Kinetics Toolkit's functions.

    Adds this function to the main documentation with no warning for the
    moment. Also adds a warning section to its docstring.

    """
    # Ensure the decorated function keeps its metadata
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Call the function being decorated and return the result
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=KTKUnstableWarning)
            warnings.filterwarnings("ignore", category=KTKPrivateWarning)
            return func(*args, **kwargs)

    wrapper.__doc__ = _inject_in_docstring(
        func.__doc__, experimental_docstring)
    wrapper._include_in_dir = True
    return wrapper


def deprecated(func):
    """
    Decorate deprecated Kinetics Toolkit's functions.

    Adds this function to the main documentation and generates a FutureWarning
    on use. Also adds a warning section to its docstring.

    """
    # Ensure the decorated function keeps its metadata
    @wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn("This function is deprecated and will be removed "
                      "in a future version of Kinetics Toolkit. Please "
                      "consult the API for replacement solutions.",
                      FutureWarning)
        # Call the function being decorated and return the result
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=KTKUnstableWarning)
            warnings.filterwarnings("ignore", category=KTKPrivateWarning)
            return func(*args, **kwargs)
    wrapper.__doc__ = _inject_in_docstring(
        func.__doc__, deprecated_docstring)
    wrapper._include_in_dir = True
    return wrapper


def unstable(func):
    """
    Decorate unstable Kinetics Toolkit's functions.

    if kineticstoolkit.config.version == 'master':
        Adds this function to the main documentation.
    else:
        Generate a KTKUnstableWarning on use.

    Also adds a warning section to its docstring.

    """
    # Ensure the decorated function keeps its metadata
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Call the function being decorated and return the result
        if kineticstoolkit.config.version != 'master':
            warnings.warn(KTKUnstableWarning(
                unstable_warning.format(name=func.__name__)))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=KTKUnstableWarning)
            warnings.filterwarnings("ignore", category=KTKPrivateWarning)
            return func(*args, **kwargs)

    if kineticstoolkit.config.version == 'master':
        wrapper._include_in_dir = True
    else:
        wrapper._include_in_dir = False

    wrapper.__doc__ = _inject_in_docstring(
        func.__doc__, unstable_docstring)
    return wrapper


def dead(func):
    """
    Decorate dead Kinetics Toolkit's functions.

    Does not add this function to the main documentation and generates a
    FutureWarning on use. Also adds a warning section to its docstring.

    Parameter listing is a list of attributes of the module that will be
    returned by the module's or class' __dir__ function.
    """
    # Ensure the decorated function keeps its metadata
    @wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn("This function is deprecated and will be removed "
                      "in a future version of Kinetics Toolkit. Please "
                      "consult the API for replacement solutions.",
                      FutureWarning)
        # Call the function being decorated and return the result
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=KTKUnstableWarning)
            warnings.filterwarnings("ignore", category=KTKPrivateWarning)
            return func(*args, **kwargs)

    wrapper.__doc__ = _inject_in_docstring(
        func.__doc__, dead_docstring)
    wrapper._include_in_dir = False
    return wrapper


def private(func):
    """
    Decorate private Kinetics Toolkit's functions.

    Does not add this function to the main documentation.
    Also adds a warning section to its docstring.

    Parameter listing is a list of attributes of the module that will be
    returned by the module's or class' __dir__ function.

    """
    # Ensure the decorated function keeps its metadata
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Call the function being decorated and return the result
        warnings.warn(KTKPrivateWarning(
            private_warning.format(name=func.__name__)))
        return func(*args, **kwargs)
    wrapper.__doc__ = _inject_in_docstring(
        func.__doc__, private_docstring)
    wrapper._include_in_dir = False
    return wrapper


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
        try:
            if module_locals[key]._include_in_dir:
                dir_.append(key)
        except AttributeError:
            pass
    return dir_


def __dir__():
    return []
