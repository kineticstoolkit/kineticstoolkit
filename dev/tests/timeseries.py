#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the TimeSeries and TimeSeriesEvent class methods.

Author: Félix Chénier
Date: July 2019
"""

import unittest
import ktk
import numpy as np


class TimeSeriesTest(unittest.TestCase):
    """TimeSeries unit tests."""

    def test_empty_constructor(self):
        """Test the empty constructor."""
        ts = ktk.TimeSeries()
        self.assertIsInstance(ts.time, np.ndarray)


if __name__ == '__main__':
    unittest.main()
