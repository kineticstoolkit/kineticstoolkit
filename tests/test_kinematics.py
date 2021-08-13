#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020 Félix Chénier
#
# This file is not for redistribution.
"""
Unit tests for ktk.kinematics.

For now this is the original tutorial without any additional check.
"""

import kineticstoolkit as ktk
import numpy as np
import warnings


def test_read_n3d_file():
    """Regression test."""
    markers = ktk.kinematics.read_n3d_file(
        ktk.config.root_folder + '/data/kinematics/sample_optotrak.n3d'
    )

    tol = 1e-4
    assert np.abs(np.sum(markers.data['Marker0']) - 172.3365) < tol
    assert np.abs(np.sum(markers.data['Marker40']) + 45.3753) < tol
    assert markers.time_info['Unit'] == 's'
    assert markers.data_info['Marker40']['Unit'] == 'm'

    labels = [
        'Probe1',
        'Probe2',
        'Probe3',
        'Probe4',
        'Probe5',
        'Probe6',
        'FRArrD',
        'FRArrG',
        'FRav',
        'ScapulaG1',
        'ScapulaG2',
        'ScapulaG3',
        'ScapulaD1',
        'ScapulaD2',
        'ScapulaD3',
        'Tete1',
        'Tete2',
        'Tete3',
        'Sternum',
        'BrasG1',
        'BrasG2',
        'BrasG3',
        'EpicondyleLatG',
        'AvBrasG1',
        'AvBrasG2',
        'AvBrasG3',
        'NAG',
        'GantG1',
        'GantG2',
        'GantG3',
        'BrasD1',
        'BrasD2',
        'BrasD3',
        'EpicondyleLatD',
        'AvBrasD1',
        'AvBrasD2',
        'AvBrasD3',
        'NAD',
        'GantD1',
        'GantD2',
        'GantD3',
    ]

    markers = ktk.kinematics.read_n3d_file(
        ktk.config.root_folder + '/data/kinematics/sample_optotrak.n3d',
        labels=labels,
    )

    assert np.abs(np.sum(markers.data['Probe1']) - 172.3365) < tol
    assert np.abs(np.sum(markers.data['GantD3']) + 45.3753) < tol
    assert markers.time_info['Unit'] == 's'
    assert markers.data_info['GantD3']['Unit'] == 'm'


def test_read_c3d_file():
    """Regression test."""
    # Regression tests for readc3dfile from OptiTrack Motive
    markers = ktk.kinematics.read_c3d_file(
        ktk.config.root_folder + '/data/kinematics/walkingOptiTrack.c3d'
    )

    assert (
        np.abs(np.nanmean(markers.data['Foot_Marker1'][:, 0:3]) - 0.1098)
        < 0.0001
    )
    assert (
        np.abs(np.nanmean(markers.data['Foot_Marker2'][:, 0:3]) - 0.1526)
        < 0.0001
    )
    assert (
        np.abs(np.nanmean(markers.data['Foot_Marker3'][:, 0:3]) - 0.1625)
        < 0.0001
    )
    assert (
        np.abs(np.nanmean(markers.data['Foot_Marker4'][:, 0:3]) - 0.1622)
        < 0.0001
    )
    assert markers.data['Foot_Marker1'][0, 3] == 1

    assert markers.time_info['Unit'] == 's'
    assert markers.data_info['Foot_Marker1']['Unit'] == 'm'


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
