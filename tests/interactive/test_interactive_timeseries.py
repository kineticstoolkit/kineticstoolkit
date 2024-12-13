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
"""Interactive tests for TimeSeries."""

import kineticstoolkit.lab as ktk
import numpy as np
import matplotlib.pyplot as plt


def test_uisync():
    """Test the uisync method of TimeSeries."""
    ts = ktk.TimeSeries(time=np.arange(100))
    ts.data["signal1"] = np.sin(ts.time)
    ts.data["signal2"] = np.cos(ts.time)

    ts = ts.ui_sync("signal1")
    fig = plt.figure()
    ts.plot()
    ktk.gui.button_dialog("Check that this setted the zero.", ["OK"])
    plt.close(fig)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
