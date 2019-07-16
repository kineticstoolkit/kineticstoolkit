"""
Kinesiology Toolkit.

KTK Kinesiology Toolkit
=======================

This is a skeleton for an all new Kinesiology toolkit based on Python instead
of Matlab, so that it's free of Matlab limitations and more distributable and
usable on onboard instruments.

Author: Félix Chénier

Date: Started on July 2019
"""

from ktk.timeseries import TimeSeries, TimeSeriesEvent
import ktk.io as io
import ktk.gui as gui
import ktk.pushrimkinetics as pushrimkinetics
import ktk.dev.dev as dev
import ktk._ipython as _ipython
