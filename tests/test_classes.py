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
"""Tests for ktk.Player."""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2024 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"

from kineticstoolkit.classes import (
    list_to_monitored_list,
    MonitoredList,
    MonitoredDict,
)


def test_list_to_monitored_list():

    count = [0]

    def callback(*args, **kwargs):
        count[0] += 1

    a = [
        1,  # 0
        1.1,  # 1
        "a",  # 2
        (1, 2),  # 3
        [1, 2],  # 4
        {1, 2},  # 5
        {1: "one", 2: "two"},  # 6
        [1, [2, 3], {3: "three", 4: "four"}, "a"],  # 7
        {  # 8
            1: "one",
            2: "two",
            3: [1, 2, 3],
            4: (1, 2),
            5: {6: "six", 7: "seven"},
        },
    ]

    b = list_to_monitored_list(a, callback)
    b.append(5)
    b[6][3] = "three"
    b[7].extend([1, 2, 3])
    b[7][1].append(4)
    b[7][2][5] = "five"
    b[8][88] = "eighty-eight"
    b[8][3].append(4)
    b[8][5][8] = "eight"
    assert count[0] == 8

    assert str(b) == (
        "[1, 1.1, 'a', (1, 2), [1, 2], {1, 2}, "
        "{1: 'one', 2: 'two', 3: 'three'}, "
        "[1, [2, 3, 4], {3: 'three', 4: 'four', 5: 'five'}, "
        "'a', 1, 2, 3], {1: 'one', 2: 'two', 3: [1, 2, 3, 4], 4: (1, 2), "
        "5: {6: 'six', 7: 'seven', 8: 'eight'}, 88: 'eighty-eight'}, 5]"
    )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
