#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020 Félix Chénier
#
# This file is not for redistribution.
"""
Interactive tests for TimeSeries.
"""

import ktk
import numpy as np
import matplotlib.pyplot as plt

def test_uisync():
    """Test the uisync method of TimeSeries."""
    ts = ktk.TimeSeries(time=np.arange(100))
    ts.data['signal1'] = np.sin(ts.time)
    ts.data['signal2'] = np.cos(ts.time)

    ts.ui_sync('signal1')
    fig = plt.figure()
    ts.plot()
    ktk.mplhelper.button_dialog('Check that this setted the zero.', ['OK'])
    plt.close(fig)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
