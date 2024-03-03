#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020-2024 Félix Chénier

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
Typing module to typecheck everything in realtime using beartype.
"""
from __future__ import annotations

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2024 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"

from typing import NewType, TYPE_CHECKING
from numpy.typing import ArrayLike as npt_ArrayLike
from numbers import Integral, Real, Complex
from beartype import (
    beartype,
    BeartypeConf,
    BeartypeViolationVerbosity,
    BeartypeHintOverrides,
)

typecheck = beartype(
    conf=BeartypeConf(
        hint_overrides=BeartypeHintOverrides(
            {int: Integral, float: Real, complex: Complex}
        ),
        violation_param_type=TypeError,
        violation_return_type=TypeError,
        violation_verbosity=BeartypeViolationVerbosity.MINIMAL,
    )
)

# Define custom types so that beartype, sphinx,
# mypy and the user are all happy
if TYPE_CHECKING:  # mypy is running
    ArrayLike = npt_ArrayLike
else:  # runtime
    ArrayLike = NewType("ArrayLike", npt_ArrayLike)  # mypy cries


# Custom check function
def check_type(
    name: str,
    value,
    expected_type,
    *,
    cast : bool = False,
    seq_value_type = None,
    seq_length : int | None = None,
    array_shape : tuple | None = None,
    dict_key_type = None,
    dict_value_type = None,
):
    """
    Check that a given parameter has the expected type and optional specs.
    
    Parameters
    ----------
    name
        Name of the parameter. Will be returned in the exception.
    value
        Value of the parameter.
    expected_type
        Expected type of the parameter.
    cast
        Optional. True to cast the value to the given type before performing
        the checks. In this case, the casted value is returned.
    seq_value_type
        Optional. For a tuple or list, ensures that every element is of the
        given type. Does not recurse into nested variables.
    seq_length
        Optional. Check that a tuple or list has a fixed length.
    array_shape
        Optional. Check that an array has a given shape. Use -1 for
        dimensions that do not matter. For instance, to check if an array
        as a shape of Nx4x4, we would use `array_shape = (-1, 4, 4)`.
    dict_key_type
        Optional. Check that every key of a dict is of a given type. Does not
        recurse into nested variables.
    dict_value_type
        Optional. Check that every value of a dict is of a given type. Does not
        recurse into nested variables.

    Returns
    -------
    Any
        The value (if cast=False), or the casted value (if cast=True)
        

    Raises
    ------
    ValueError
        If the value does not meet the given criteria.
        
    """
    # Accept ints as floats
    numerical_tower = lambda x: float(x) if isinstance(x, int) else x
    value = numerical_tower(value)
    # Cast if asked, otherwise check type.
    if cast:
        value = expected_type(value)
    elif not isinstance(value, expected_type):
        raise ValueError(
            f"{name} must be of type {expected_type}, however it is of type "
            f"{type(value)}, with a value of {value}."
        )
    # Other specs
    if seq_value_type is not None:
        for element in value:            
            if not isinstance(numerical_tower(element), seq_value_type):
                raise ValueError(
                    f"{name} must contain only elements of type "
                    f"{seq_value_type}, however it contains a value of type "
                    f"{type(element)}, with a value of {element}."
                )
                
    if seq_length is not None and len(value) != seq_length:
        raise ValueError(
            f"{name} must have a length of {seq_length}, however it has "
            f"a length of {len(value)}."
        )
        
    if array_shape is not None:
        value_shape = value.shape
        if len(value_shape) != len(array_shape):
            raise ValueError(
                f"{name} must have {len(array_shape)} dimensions, however it "
                f"has {len(value_shape)} dimensions with a shape of "
                f"{value_shape}."
            )
        for i_dim, dim in enumerate(array_shape):
            if dim != -1 and dim != value_shape[i_dim]:
                raise ValueError(
                    f"Dimension {i_dim} of {value} must be {dim}, however it "
                    f"is {value_shape[i_dim]} ({value} has a shape of "
                    f"{value_shape}."
                )

    if dict_key_type is not None:
        for key in value:
            if not isinstance(key, dict_key_type):
                raise ValueError(
                    f"{name} must contain only keys of type "
                    f"{dict_key_type}, however it contains a key of type "
                    f"{type(key)}, with a value of {key}."
                )

    if dict_value_type is not None:
        for key in value:
            if not isinstance(value[key], dict_value_type):
                raise ValueError(
                    f"{name} must contain only values of type "
                    f"{dict_value_type}, however it contains a value of type "
                    f"{type(value)}, with a value of {value}."
                )

    return value                
