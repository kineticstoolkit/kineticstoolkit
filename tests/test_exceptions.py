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
from __future__ import annotations

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"

"""Unit tests for the exceptions module."""

import kineticstoolkit as ktk
import numpy as np


def test_check_types():
    """Test ktk.exceptions.check_types."""

    def test_function(
        bool_var: bool,
        int_var: int,
        float_var: float,
        str_var: str,
        list_var: list[str | float],
        tuple_var: tuple[str | float],
        dict_var: dict[str, float],
        set_var: set[str | float],
        ts_var: ktk.TimeSeries,
    ) -> None:
        ktk.exceptions.check_types(test_function, locals())

    # Should pass
    test_function(
        bool_var=True,
        int_var=1,
        float_var=1.1,
        str_var="a",
        list_var=["a", 1.1],
        tuple_var=("a", 1.1),
        dict_var={"a": 1.1},
        set_var={"a", 1.1},
        ts_var=ktk.TimeSeries(time=np.arange(10)),
    )

    # Should pass (numeric tower)
    test_function(
        bool_var=True,
        int_var=1,
        float_var=1,
        str_var="a",
        list_var=["a", 1.1],
        tuple_var=("a", 1.1),
        dict_var={"a": 1.1},
        set_var={"a", 1.1},
        ts_var=ktk.TimeSeries(time=np.arange(10)),
    )

    # Should fail
    try:
        test_function(
            bool_var=1,
            int_var=1,
            float_var=1.1,
            str_var="a",
            list_var=["a", 1.1],
            tuple_var=("a", 1.1),
            dict_var={"a": 1.1},
            set_var={"a", 1.1},
            ts_var=ktk.TimeSeries(time=np.arange(10)),
        )
        raise ValueError("This should have failed")
    except TypeError:
        pass

    # Should fail
    try:
        test_function(
            bool_var=False,
            int_var=1.5,
            float_var=1.1,
            str_var="a",
            list_var=["a", 1.1],
            tuple_var=("a", 1.1),
            dict_var={"a": 1.1},
            set_var={"a", 1.1},
            ts_var=ktk.TimeSeries(time=np.arange(10)),
        )
        raise ValueError("This should have failed")
    except TypeError:
        pass

    # Should fail
    try:
        test_function(
            bool_var=False,
            int_var=1,
            float_var="a",
            str_var="a",
            list_var=["a", 1.1],
            tuple_var=("a", 1.1),
            dict_var={"a": 1.1},
            set_var={"a", 1.1},
            ts_var=ktk.TimeSeries(time=np.arange(10)),
        )
        raise ValueError("This should have failed")
    except TypeError:
        pass

    # Should fail
    try:
        test_function(
            bool_var=False,
            int_var=1,
            float_var=1.1,
            str_var=1.1,
            list_var=["a", 1.1],
            tuple_var=("a", 1.1),
            dict_var={"a": 1.1},
            set_var={"a", 1.1},
            ts_var=ktk.TimeSeries(time=np.arange(10)),
        )
        raise ValueError("This should have failed")
    except TypeError:
        pass

    # Should fail
    try:
        test_function(
            bool_var=False,
            int_var=1,
            float_var=1.1,
            str_var="a",
            list_var="a",
            tuple_var=("a", 1.1),
            dict_var={"a": 1.1},
            set_var={"a", 1.1},
            ts_var=ktk.TimeSeries(time=np.arange(10)),
        )
        raise ValueError("This should have failed")
    except TypeError:
        pass

    # Should fail
    try:
        test_function(
            bool_var=False,
            int_var=1,
            float_var=1.1,
            str_var="a",
            list_var=["a", 1.1],
            tuple_var={"a": 1.1},
            dict_var={"a": 1.1},
            set_var={"a", 1.1},
            ts_var=ktk.TimeSeries(time=np.arange(10)),
        )
        raise ValueError("This should have failed")
    except TypeError:
        pass

    # Should fail
    try:
        test_function(
            bool_var=False,
            int_var=1,
            float_var=1.1,
            str_var="a",
            list_var=["a", 1.1],
            tuple_var=("a", 1.1),
            dict_var=("a", 1.1),
            set_var={"a", 1.1},
            ts_var=ktk.TimeSeries(time=np.arange(10)),
        )
        raise ValueError("This should have failed")
    except TypeError:
        pass

    # Should fail
    try:
        test_function(
            bool_var=False,
            int_var=1,
            float_var=1.1,
            str_var="a",
            list_var=["a", 1.1],
            tuple_var=("a", 1.1),
            dict_var={"a": 1.1},
            set_var=["a", 1.1],
            ts_var=ktk.TimeSeries(time=np.arange(10)),
        )
        raise ValueError("This should have failed")
    except TypeError:
        pass

    # Should fail
    try:
        test_function(
            bool_var=False,
            int_var=1,
            float_var=1.1,
            str_var="a",
            list_var=["a", 1.1],
            tuple_var=("a", 1.1),
            dict_var=("a", 1.1),
            set_var={"a", 1.1},
            ts_var="timeseries",
        )
        raise ValueError("This should have failed")
    except TypeError:
        pass


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
