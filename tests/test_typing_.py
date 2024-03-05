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
Test for typing_ module.
"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2024 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"

from kineticstoolkit.typing_ import check_param
from kineticstoolkit import TimeSeries, TimeSeriesEvent
import numpy as np
import pandas as pd


TYPES = {
    bool: True,
    int: 3,
    float: 3.4,
    complex: 5 + 4j,
    str: "super",
    list: [1, 2, "a"],
    tuple: (1, 2, "a"),
    dict: {1: "1", "2": 2, 3: "a"},
    TimeSeries: TimeSeries([1, 2, 3]),
    TimeSeriesEvent: TimeSeriesEvent(0.1, "event"),
    np.ndarray: np.eye(3),
    pd.DataFrame: pd.DataFrame(np.eye(3)),
    pd.Series: pd.Series([1, 2, 3]),
}


def test_check_param():
    for key in TYPES:
        check_param("test", TYPES[key], key)
    check_param("test", [1, 2, 3], list, length=3, contents_type=int)
    check_param("test", ("a", "b", "c"), tuple, length=3, contents_type=str)
    check_param(
        "test",
        {1: "a", 2: "b"},
        dict,
        length=2,
        key_type=int,
        contents_type=str,
    )
    check_param("test", np.eye(3), np.ndarray, shape=(3, 3), ndims=2)

    # Now do similar tests but with errors
    try:
        check_param("test", [1, 2, 3], list, length=3, contents_type=str)
        raise Exception("This should fail.")
    except TypeError:
        pass
    try:
        check_param("test", [1, 2, "a"], list, length=3, contents_type=str)
        raise Exception("This should fail.")
    except TypeError:
        pass

    try:
        check_param("test", ("a", "b", "c"), tuple, contents_type=int)
        raise Exception("This should fail.")
    except TypeError:
        pass
    try:
        check_param("test", ("a", "b", 3), tuple, contents_type=int)
        raise Exception("This should fail.")
    except TypeError:
        pass
    try:
        check_param("test", ("a", "b"), tuple, length=3)
        raise Exception("This should fail.")
    except ValueError:
        pass

    try:
        check_param(
            "test",
            {1: "a", 2: "b"},
            dict,
            length=2,
            key_type=str,
        )
        raise Exception("This should fail.")
    except TypeError:
        pass

    try:
        check_param("test", np.eye(4), np.ndarray, shape=(3, 3))
        raise Exception("This should fail.")
    except ValueError:
        pass
    try:
        check_param("test", np.eye(4), np.ndarray, shape=(3, 3, 3))
        raise Exception("This should fail.")
    except ValueError:
        pass
    try:
        check_param("test", np.eye(3), np.ndarray, ndims=1)
        raise Exception("This should fail.")
    except ValueError:
        pass

    # Check that ints work as floats
    check_param("test", 1, float)
    # but that floats don't work as int
    try:
        check_param("test", 1.0, int)
        raise Exception("This should fail.")
    except TypeError:
        pass

    # Check that tuples of types work
    check_param("test", 1, (int, str, None))
    check_param("test", "a", (int, str, None))
    try:
        check_param("test", [1, 2, 3], (int, str))
        raise Exception("This should fail.")
    except TypeError:
        pass


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
