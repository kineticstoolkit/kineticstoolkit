#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020-2022 Félix Chénier

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for Kinetics Toolkit's doc module."""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2021 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import kineticstoolkit as ktk


def test_download_sample_data():
    """Test ktk.doc.download_sample_data."""
    # Download the usual way
    file1 = ktk.doc.download("filters_noisy_signals.ktk.zip")
    # Use the local version
    file2 = ktk.doc.download(
        "filters_noisy_signals.ktk.zip", force_download=True
    )
    assert file1 != file2
    data1 = ktk.load(file1)
    data2 = ktk.load(file2)
    assert data1 == data2


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
