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
Provide decorators for ktk functions.

The following decorator can be used on each ktk's function and method:
    @stable: Documented, stable function in a ktk release.
    @experimental: Documented, experimental function in a ktk release.
    @deprecated: Documented but deprecated function in a ktk release.
    @unstable: Documented only in the development version.
    @dead: Undocumented, deprecated function in the development version.
    @private: Undocumented, for private use by the module only.

    Each of these decorators take a list as argument, and adds the function
    to the list if this function should be added to the __dir__ magic function.

"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


from functools import wraps
import warnings
import ktk.config


def stable(listing):
    """
    Decorate stable ktk functions.

    Add this function to the main documentation.

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
    Decorate experimental ktk functions.

    Add this function to the main documentation with no warning for the
    moment.

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


def deprecated(listing):
    """
    Decorate deprecated ktk functions.

    Add this function to the main documentation and generate a FutureWarning
    on use.

    Parameter listing is a list of attributes of the module that will be
    returned by the module's or class' __dir__ function.
    """
    # Define a new decorator, named "decorator", to return
    def decorator(func):
        # Ensure the decorated function keeps its metadata
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn("This function is deprecated and will be removed "
                          "in a future version of ktk. Please consult ktk's "
                          "API for replacement solutions.",
                          FutureWarning)
            # Call the function being decorated and return the result
            return func(*args, **kwargs)
        listing.append(func.__name__)
        return wrapper
    # Return the new decorator
    return decorator


def unstable(listing):
    """
    Decorate unstable ktk functions.

    Add this function to the main documentation only if
    ktk.config.version == 'master'

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
        if ktk.config.version == 'master':
            listing.append(func.__name__)
        return wrapper
    # Return the new decorator
    return decorator


def dead(listing):
    """
    Decorate dead ktk functions.

    Do not add this function to the main documentation and generate a
    FutureWarning on use.

    Parameter listing is a list of attributes of the module that will be
    returned by the module's or class' __dir__ function.
    """
    # Define a new decorator, named "decorator", to return
    def decorator(func):
        # Ensure the decorated function keeps its metadata
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn("This function is deprecated and will be removed "
                          "in a future version of ktk. Please consult ktk's "
                          "API for replacement solutions.",
                          FutureWarning)
            # Call the function being decorated and return the result
            return func(*args, **kwargs)
        return wrapper
    # Return the new decorator
    return decorator


def private(listing):
    """
    Decorate private ktk functions.

    Do not add this function to the main documentation.

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
        return wrapper
    # Return the new decorator
    return decorator
