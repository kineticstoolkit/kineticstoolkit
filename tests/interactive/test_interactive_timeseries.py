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

def test_uisync():
    """Test the uisync method of TimeSeries."""
    ts = ktk.TimeSeries(time=np.arange(100))
    ts.data['signal1'] = np.sin(ts.time)
    ts.data['signal2'] = np.cos(ts.time)

    print('Click somewhere to check that it becomes the new zero time.')
    ts.ui_sync('signal1')
    ts.plot()

    ktk.mplhelper.button_dialog('Check that this setted the zero.', ['OK'])

