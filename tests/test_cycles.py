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

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"

"""
Unit tests for the cycles module.
"""
import ktk
import numpy as np
import matplotlib.pyplot as plt


def test_normalize():
    # Create a TimeSeries with some events directly synced with the data and
    # some other that aren't.
    ts = ktk.TimeSeries()
    ts.time = np.linspace(0, 20, 201)  # 0 to 20 seconds by 0.1s
    ts.data['test'] = ts.time ** 2
    ts.add_event(1.0, 'push')
    ts.add_event(2.0, 'recovery')
    ts.add_event(3.0, 'push')
    ts.add_event(3.95, 'recovery')
    ts.add_event(5.0, 'push')
    ts.add_event(6.05, 'recovery')
    ts.add_event(7.05, 'push')
    ts.add_event(8.0, 'recovery')
    ts.add_event(8.95, 'push')
    ts.add_event(10.05, 'recovery')

    ts1 = ktk.cycles.time_normalize(ts, 'push', 'recovery')
    assert len(ts1.events) == 10  # No missing events

    # Test that if we re-time-normalize, we obtain the same TimeSeries
    ts2 = ktk.cycles.time_normalize(ts1, 'push', '_')
    assert ts1 == ts2

    # Samething but with push to next push
    ts1 = ktk.cycles.time_normalize(ts, 'push', '_')
    assert len(ts1.events) == 12  # No missing events

    # Test that if we re-time-normalize, we obtain the same TimeSeries
    ts2 = ktk.cycles.time_normalize(ts1, 'push', '_')
    assert ts1 == ts2

    # There should be no nan in ts2
    assert ~ts2.isnan('test').all()


def test_most_repeatable_cycles():
    # Create a TimeSeries with 5 cycles, one of those is different from
    # the others
    data = np.array([
        np.sin(np.arange(0, 10, 0.1)) + 0.00,  # 0 - most diff. 3rd removed
        np.sin(np.arange(0, 10, 0.1)) + 0.10,  # 1 - 1st of remain. 5th removed
        np.cos(np.arange(0, 10, 0.1)) + 0.15,  # 2 - cos. 2nd removed
        np.sin(np.arange(0, 10, 0.1)) + 0.15,  # 3 - with nans. 1st removed
        np.sin(np.arange(0, 10, 0.1)) + 0.12])  # 4 - 2nd of remn. 4th removed
    # Put some nans in the fourth cycle
    data[3, 30] = np.nan

    test = ktk.cycles.most_repeatable_cycles(data)

    assert test == [1, 4, 0, 2, 3]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
