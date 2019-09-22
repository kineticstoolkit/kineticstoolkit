#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for ktk.kinematics
"""

import ktk
import numpy as np


def test_read_c3d_file():
    # Regression tests with a reference mat made on KTK for Matlab
    ts = ktk.kinematics.read_c3d_file(
            '../../tutorials/data/kinematics/sprintbasket.c3d')
    ref = ktk.loadmat(
            '../../tutorials/data/kinematics/sprintbasket.mat')

    # ref seems to miss a list sample. We will then samples compare 0:-1.
    for label in ts.data.keys():
        reflabel = label.replace(':', '_')
        assert(np.nanmean(
                np.abs(ts.data[label][0:-1] - ref.data[reflabel])) < 1E-6)


def test_open_in_mokka():
    #  Open previously generated kinematics and play it in Mokka
    data = ktk.loadmat(
            '../../tutorials/data/kinematics/kinematics_racingwheelchair.mat')
    ktk.kinematics.open_in_mokka(data['markers'])
