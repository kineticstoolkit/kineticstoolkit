#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the ktk.pushrimkinetics functions.

Author: Félix Chénier
Date: July 2019
"""

import ktk
import numpy as np


def _assert_almost_equal(float1, float2):
    assert abs(float1 - float2) < 1E-6


def test_read_file():
    """Test the read_file function."""
    # This is a non-regression test based on the Matlab's KTK tutorial.
    kinetics = ktk.pushrimkinetics.read_file(
            ktk._ROOT_FOLDER +
            '/tutorials/data/pushrimkinetics/' +
            'sample_swl_overground_propulsion_withrubber.csv')

    _assert_almost_equal(np.mean(kinetics.data['Forces']),
                         -0.0044330903410570)
    _assert_almost_equal(np.mean(kinetics.data['Moments']),
                         0.5374323092944534)
    _assert_almost_equal(np.mean(kinetics.data['Angle']),
                         46.4698216459348359)
    _assert_almost_equal(np.mean(kinetics.data['Channels']),
                         2059.6018397986695163)
    _assert_almost_equal(np.mean(kinetics.data['Index']),
                         3841.5000000000000000)


def test_no_regressions():
    """Test the methods against Matlab/KTK for absence of regression."""
    # Read file
    kinetics = ktk.pushrimkinetics.read_file(
            ktk._ROOT_FOLDER +
            '/tutorials/data/pushrimkinetics/' +
            'sample_swl_overground_propulsion_withrubber.csv')

    # calculate_forces_and_moments
    test = ktk.pushrimkinetics.calculate_forces_and_moments(
            kinetics, 'LIO-123')  # Random calibration here.
    forces = np.nanmean(test.data['Forces'], 0)
    moments = np.nanmean(test.data['Moments'], 0)
    _assert_almost_equal(forces[0], -8.849994801918)
    _assert_almost_equal(forces[1], -11.672364564453)
    _assert_almost_equal(forces[2], -2.646989586045)
    _assert_almost_equal(moments[0], -0.039625979603)
    _assert_almost_equal(moments[1], -0.088833025939)
    _assert_almost_equal(moments[2], 2.297597031073)

    # Find recovery indices
    indices = np.nonzero(ktk.pushrimkinetics.find_recovery_indices(
            kinetics.data['Moments'][:, 2]))
    _assert_almost_equal(np.mean(indices),
                         3906.6627036547774878)

    # Remove sinusoids
    new_kinetics = ktk.pushrimkinetics.remove_sinusoids(kinetics)
    _assert_almost_equal(np.mean(new_kinetics.data['Forces']),
                         1.2971684579009064)
    _assert_almost_equal(np.mean(new_kinetics.data['Moments']),
                         0.4972708141781993)

    # Remove sinusoids using a baseline
    baseline = ktk.pushrimkinetics.read_file(
            ktk._ROOT_FOLDER +
            '/tutorials/data/pushrimkinetics/' +
            'sample_swl_overground_baseline_withrubber.csv')
    newnew_kinetics = ktk.pushrimkinetics.remove_sinusoids(kinetics,
                                                           baseline)
    _assert_almost_equal(np.mean(newnew_kinetics.data['Forces']),
                         1.4048102831351081)

    # Detect pushes
    float_event_times = []
    pushes = ktk.pushrimkinetics.detect_pushes(new_kinetics)
    event_times = np.array(pushes.events)[:, 0]
    for i in range(0, len(event_times)):
        float_event_times.append(float(event_times[i]))

    assert len(pushes.events) == 77
    _assert_almost_equal(np.mean(float_event_times), 17.2115256494)
