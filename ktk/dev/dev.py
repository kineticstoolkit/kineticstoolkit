#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KTK development functions.

Author: Félix Chénier
Date: July 2019
"""

import subprocess


def runtests():
    """Run all unit tests."""
    subprocess.call('./ktk_timeseries_test.py')
