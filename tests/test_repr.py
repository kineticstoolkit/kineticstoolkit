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
test_repr

Unit tests for the _repr module.

"""

import kineticstoolkit._repr as _repr


def test_format_dict_entries():
    """Test dict formatting."""
    # Test with all strings in keys
    d = {
        "key1": "value1",
        "key2": "value2",
    }
    assert (
        _repr._format_dict_entries(d)
        == "    'key1': 'value1'\n    'key2': 'value2'\n"
    )
    assert (
        _repr._format_dict_entries(d, quotes=False)
        == "    key1: 'value1'\n    key2: 'value2'\n"
    )

    # Test with mixed types in keys and long values
    d = {
        "1": "value1",
        2: "value2",
        3.0: "A" * 200,
    }
    assert (
        _repr._format_dict_entries(d)
        == "    '1': 'value1'\n      2: 'value2'\n    3.0: 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA...\n"
    )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
