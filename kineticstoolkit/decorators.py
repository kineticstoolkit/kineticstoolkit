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

    - @deprecated(since: str, until: str, details: str):
        Documented but deprecated function in a release.

    - @dead:
        Undocumented, deprecated function in the development version.

    Each of these decorators add the _include_in_dir property to the decorated
    function. The provided function ``directory`` looks at these properties
    to return a custom __dir__ to Kinematics Toolkit's classes. See such class
    for example.

"""
from __future__ import annotations


__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


from functools import wraps
import warnings
import textwrap
import kineticstoolkit.exceptions as exceptions


def _inject_in_docstring(docstring: str, text: str) -> str:
    """Inject a string into the top of a docstring, after line 1."""
    if docstring == "" or docstring is None:
        return text
    result = []
    splitted = textwrap.dedent(docstring).split("\n")
    first_line_done = False
    for line in splitted:
        if not first_line_done:
            if len(line) != 0:
                first_line_done = True
                result.append(line)
                for text_line in text.split("\n"):
                    result.append(text_line)
            else:
                result.append(line)
        else:
            result.append(line)
    return "\n".join(result)


def deprecated(since: str, until: str, details: str):
    """
    Decorate deprecated Kinetics Toolkit's functions.

    Generates a FutureWarning and adds a warning section to its docstring.
    These functions are included in API documentation.

    """

    def real_decorator(func):
        func_name = func.__name__
        string = (
            f"The function {func_name} is deprecated since "
            f"{since} and is scheduled to be removed "
            f"in {until}. {details}"
        )

        # Ensure the decorated function keeps its metadata
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(string, category=FutureWarning, stacklevel=2)
            # Call the function being decorated and return the result
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                return func(*args, **kwargs)

        wrapper.__doc__ = _inject_in_docstring(
            func.__doc__, f"\nWarning\n-------\n{string}"
        )
        wrapper._is_deprecated = True
        return wrapper

    return real_decorator


def timeseries_method(func):
    """
    Decorate TimeSeries methods for error solving in case of exception.

    Generates a FutureWarning and adds a warning section to its docstring.
    These functions are included in API documentation.

    """

    # Ensure the decorated function keeps its metadata
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            retval = func(*args, **kwargs)
        except exceptions.KTKBaseException:
            raise
        except Exception:
            args[0]._check_well_shaped()
            raise exceptions.KTKError(
                """
                An error happened, and it seems to be due to a problem in the
                Kinetics Toolkit package. You are advised to:
                    
                (1) Ensure that your copy of Kinetics Toolkit is up to date. Consult
                    this web page to see how to update:
                        
                    https://kineticstoolkit.uqam.ca/doc/ktk_installing.html
                
                (2) Consult the discussion board and issue tracker to see if a similar
                    issue has been notified, and if a workaround is available.
                        
                (3) Ask for help in the Q&A discussion board.
                        
                (4) Issue a bug on the issue tracker. Please copy-paste the entire
                    traceback, starting on line "Traceback (most recent call last):"
                    in this issue.
                    
                Discussion board:
                    
                    https://github.com/felixchenier/kineticstoolkit/discussions
                    
                Issue tracker:
                    
                    https://github.com/felixchenier/kineticstoolkit/issues        
                
                Please don't leave this error unreported. Reporting an error is always
                the first step toward fixing it.
                
                """
            )
        else:
            return retval

    return wrapper


def __dir__():
    return []
