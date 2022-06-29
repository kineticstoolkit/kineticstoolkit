#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2022 Félix Chénier

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Unit tests for extension support."""


__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2022 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import kineticstoolkit as ktk
import sys


sys.path.append(
    ktk.config.root_folder + "/tests/kineticstoolkit_testextension"
)
ktk.import_extensions()


def test_testextension():
    """Test testextension."""

    assert ktk.ext.testextension.return_empty_timeseries() == ktk.TimeSeries()


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
