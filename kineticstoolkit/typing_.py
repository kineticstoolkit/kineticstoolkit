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
Typing module to typecheck everything in realtime.
"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2024 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"

from numbers import Integral, Real, Complex
from typing import NewType, TYPE_CHECKING
from numpy.typing import ArrayLike as npt_ArrayLike

# Define custom types so that sphinx and mypy and the users are all happy
if TYPE_CHECKING:  # mypy is running
    ArrayLike = npt_ArrayLike
else:  # runtime
    # mypy cries but sphinx is fine and doesn't expand ArrayLike
    ArrayLike = NewType("ArrayLike", npt_ArrayLike)


PARAM_MAPPING = {
    int: Integral,
    float: Real,
    complex: Complex,
    None: type(None),
}


def check_param(
    name: str,
    value,
    expected_type,
    *,
    contents_type=None,
    key_type=None,
    length: int | None = None,
    ndims: int | None = None,
    shape: tuple | None = None,
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
    contents_type
        Optional. For a tuple or list, ensures that every element is of the
        given type. For a dictionary, ensures that every value is of the
        current type. Does not recurse into nested variables.
    key_type
        Optional. Check that every key of a dict is of a given type. Does not
        recurse into nested variables.
    length
        Optional. Check that a tuple or list has a fixed length.
    ndims
        Optional. Check that an array has a given number of dimensions.
    array_shape
        Optional. Check that an array has a given shape. Use -1 for
        dimensions that do not matter. For instance, to check if an array
        as a shape of Nx4x4, we would use `array_shape = (-1, 4, 4)`.
    dict_value_type
        Optional. Check that every value of a dict is of a given type. Does not
        recurse into nested variables.

    Returns
    -------
    Any
        The value


    Raises
    ------
    TypeError
        If the value or its contents is of the wrong type.

    ValueError
        If the value does not meet the given criteria.

    """
    if isinstance(expected_type, tuple):
        mapped_expected_type = tuple(
            [PARAM_MAPPING.get(_, _) for _ in expected_type]
        )
    else:
        mapped_expected_type = PARAM_MAPPING.get(expected_type, expected_type)  # type: ignore

    # Check type
    if not isinstance(value, mapped_expected_type):  # type: ignore
        raise TypeError(
            f"{name} must be of type {expected_type}, however it is of type "
            f"{type(value)}, with a value of {value}."
        )
    # Other specs
    if contents_type is not None:

        if isinstance(contents_type, tuple):
            mapped_contents_type = tuple(
                [PARAM_MAPPING.get(_, _) for _ in contents_type]
            )
        else:
            mapped_contents_type = PARAM_MAPPING.get(
                contents_type, contents_type
            )  # type: ignore

        if isinstance(value, dict):
            value_list = value.values()
        else:
            value_list = value
        for element in value_list:
            if not isinstance(element, mapped_contents_type):  # type: ignore
                raise TypeError(
                    f"{name} must contain only elements of type "
                    f"{contents_type}, however it contains a value of type "
                    f"{type(element)}, with a value of {element}."
                )

    if length is not None and len(value) != length:
        raise ValueError(
            f"{name} must have a length of {length}, however it has "
            f"a length of {len(value)}."
        )

    if ndims is not None:
        value_shape = value.shape
        if len(value_shape) != ndims:
            raise ValueError(
                f"{name} must have {ndims} dimensions, however it "
                f"has {len(value_shape)} dimensions with a shape of "
                f"{value_shape}."
            )

    if shape is not None:
        value_shape = value.shape
        if len(value_shape) != len(shape):
            raise ValueError(
                f"{name} must have {len(shape)} dimensions, however it "
                f"has {len(value_shape)} dimensions with a shape of "
                f"{value_shape}."
            )
        for i_dim, dim in enumerate(shape):
            if dim != -1 and dim != value_shape[i_dim]:
                raise ValueError(
                    f"Dimension {i_dim} of {name} must be {dim}, however it "
                    f"is {value_shape[i_dim]} since {value} has a shape of "
                    f"{value_shape}."
                )

    if key_type is not None:

        if isinstance(key_type, tuple):
            mapped_key_type = tuple(
                [PARAM_MAPPING.get(_, _) for _ in key_type]
            )
        else:
            mapped_key_type = PARAM_MAPPING.get(key_type, key_type)  # type: ignore

        for key in value:
            if not isinstance(key, mapped_key_type):  # type: ignore
                raise TypeError(
                    f"{name} must contain only keys of type "
                    f"{key_type}, however it contains a key of type "
                    f"{type(key)}, with a value of {key}."
                )

    return value
