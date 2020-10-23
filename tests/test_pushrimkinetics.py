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
Unit tests for the pushrimkinetics module.

Most of the tests are the tutorial itself.
"""

import kineticstoolkit as ktk
import numpy as np


def test_read_csv_txt_file():
    """Test that read_file works similarly for SW's csv and txt files."""
    filename_csv = ('../doc/data/pushrimkinetics/'
                    'sample_sw_csvtxt.csv')
    kinetics_csv = ktk.pushrimkinetics.read_file(
        filename_csv, file_format='smartwheel')

    filename_txt = ('../doc/data/pushrimkinetics/'
                    'sample_sw_csvtxt.TXT')
    kinetics_txt = ktk.pushrimkinetics.read_file(
        filename_txt, file_format='smartwheeltxt')

    assert np.all(np.abs(kinetics_csv.data['Channels'] -
                         kinetics_csv.data['Channels']) < 1E-10)


def test_remove_offsets():
    """Test that remove_offsets works with and without a baseline."""
    kinetics = ktk.pushrimkinetics.read_file(
        '../doc/data/pushrimkinetics/'
        'sample_swl_overground_propulsion_withrubber.csv',
        file_format='smartwheel')

    baseline = ktk.pushrimkinetics.read_file(
        '../doc/data/pushrimkinetics/'
        'sample_swl_overground_baseline_withrubber.csv',
        file_format='smartwheel')

    no_offsets1 = ktk.pushrimkinetics.remove_offsets(kinetics)
    no_offsets2 = ktk.pushrimkinetics.remove_offsets(kinetics, baseline)

    # Assert that all force differences are within 1 N
    assert np.all(np.abs(no_offsets1.data['Forces'] -
                         no_offsets2.data['Forces']) < 1)

    # Assert that all moment differences are within 0.1 Nm
    assert np.all(np.abs(no_offsets1.data['Moments'] -
                         no_offsets2.data['Moments']) < 0.1)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
