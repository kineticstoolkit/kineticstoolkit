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
        ktk.config.root_folder +
        '/data/kinematics/sample_optotrak.n3d')

    tol = 1E-4
    assert(np.abs(np.sum(markers.data['Marker0']) - 172.3365) < tol)
    assert(np.abs(np.sum(markers.data['Marker40']) + 45.3753) < tol)
    assert(markers.time_info['Unit'] == 's')
    assert(markers.data_info['Marker40']['Unit'] == 'm')

    labels = [
        'Probe1', 'Probe2', 'Probe3', 'Probe4', 'Probe5', 'Probe6',
        'FRArrD', 'FRArrG', 'FRav', 'ScapulaG1', 'ScapulaG2', 'ScapulaG3',
        'ScapulaD1', 'ScapulaD2', 'ScapulaD3', 'Tete1', 'Tete2', 'Tete3',
        'Sternum', 'BrasG1', 'BrasG2', 'BrasG3', 'EpicondyleLatG',
        'AvBrasG1', 'AvBrasG2', 'AvBrasG3', 'NAG', 'GantG1', 'GantG2',
        'GantG3', 'BrasD1', 'BrasD2', 'BrasD3', 'EpicondyleLatD', 'AvBrasD1',
        'AvBrasD2', 'AvBrasD3', 'NAD', 'GantD1', 'GantD2', 'GantD3']

    markers = ktk.kinematics.read_n3d_file(
        ktk.config.root_folder +
        '/data/kinematics/sample_optotrak.n3d', labels=labels)

    assert(np.abs(np.sum(markers.data['Probe1']) - 172.3365) < tol)
    assert(np.abs(np.sum(markers.data['GantD3']) + 45.3753) < tol)
    assert(markers.time_info['Unit'] == 's')
    assert(markers.data_info['GantD3']['Unit'] == 'm')


def test_read_c3d_file():
    """Regression test."""
    # Regression tests for readc3dfile from OptiTrack Motive
    markers = ktk.kinematics.read_c3d_file(
        ktk.config.root_folder +
        '/data/kinematics/walkingOptiTrack.c3d')

    assert(np.abs(np.nanmean(markers.data['Foot_Marker1'][:, 0:3]) -
                  0.1098) < 0.0001)
    assert(np.abs(np.nanmean(markers.data['Foot_Marker2'][:, 0:3]) -
                  0.1526) < 0.0001)
    assert(np.abs(np.nanmean(markers.data['Foot_Marker3'][:, 0:3]) -
                  0.1625) < 0.0001)
    assert(np.abs(np.nanmean(markers.data['Foot_Marker4'][:, 0:3]) -
                  0.1622) < 0.0001)
    assert(markers.data['Foot_Marker1'][0, 3] == 1)

    assert(markers.time_info['Unit'] == 's')
    assert(markers.data_info['Foot_Marker1']['Unit'] == 'm')


def test_reconstruction_old():
    """Test the deprecated reconstruction chain - that calls the new one."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)

        marker_names = ['Probe1', 'Probe2', 'Probe3', 'Probe4', 'Probe5', 'Probe6',
                        'WheelchairRearR', 'WheelchairRearL', 'WheelchairFront',
                        'ScapulaL1', 'ScapulaL2', 'ScapulaL3',
                        'ScapulaR1', 'ScapulaR2', 'ScapulaR3',
                        'Head1', 'Head2', 'Head3',
                        'Sternum',
                        'ArmL1', 'ArmL2', 'ArmL3',
                        'LateralEpicondyleL', 'ForearmL1', 'ForearmL2', 'ForearmL3',
                        'NAG',
                        'GloveL1', 'GloveL2', 'GloveL3',
                        'ArmR1', 'ArmR2', 'ArmR3',
                        'LateralEpicondyleR', 'ForearmR1', 'ForearmR2', 'ForearmR3',
                        'NAR',
                        'GloveR1', 'GloveR2', 'GloveR3']

        config = dict()
        config['RigidBodies'] = dict()

        # Read the static trial
        markers = ktk.kinematics.read_n3d_file(
            ktk.config.root_folder +
            '/data/kinematics/sample_static.n3d',
            labels=marker_names)

        # Create the rigid body configurations
        config['RigidBodies']['ArmR'] = ktk.kinematics.create_rigid_body_config(
            markers, ['ArmR1', 'ArmR2', 'ArmR3'])

        config['RigidBodies']['ForearmR'] = ktk.kinematics.create_rigid_body_config(
            markers, ['ForearmR1', 'ForearmR2', 'ForearmR3'])

        config['RigidBodies']['Probe'] = {
            'MarkerNames': ['Probe1', 'Probe2', 'Probe3',
                            'Probe4', 'Probe5', 'Probe6'],
            'LocalPoints': np.array([[
                [2.1213,   2.1213,  2.0575,   2.1213,   1.7070,   1.7762],
                [-15.8328, 15.8508, 16.0096,  16.1204,  -15.5780, -15.6057],
                [86.4285,  86.4285, 130.9445, 175.4395, 175.3805, 130.8888],
                [1000,     1000,    1000,     1000,     1000,     1000]]]
            ) / 1000
        }

        config['VirtualMarkers'] = dict()

        def process_probing_acquisition(file_name, rigid_body_name):

            # Load the markers
            markers = ktk.kinematics.read_n3d_file(
                file_name, labels=marker_names)

            # Follow the rigid bodies in those markers
            rigid_bodies = ktk.kinematics.register_markers(markers,
                                                           config['RigidBodies'])

            # Add the marker 'ProbeTip' in markers. This is the origin of the Probe
            # rigid body.
            markers.data['ProbeTip'] = rigid_bodies.data['Probe'][:, :, 3]
            markers = markers.add_data_info('ProbeTip', 'Color', 'r')

            # Create the marker configuration
            return ktk.kinematics.create_virtual_marker_config(
                markers, rigid_bodies, 'ProbeTip', rigid_body_name)

        config['VirtualMarkers']['AcromionR'] = process_probing_acquisition(
            ktk.config.root_folder +
            '/data/kinematics/sample_probing_acromion_R.n3d', 'ArmR')

        config['VirtualMarkers']['MedialEpicondyleR'] = process_probing_acquisition(
            ktk.config.root_folder +
            '/data/kinematics/sample_probing_medial_epicondyle_R.n3d', 'ArmR')

        config['VirtualMarkers']['OlecraneR'] = process_probing_acquisition(
            ktk.config.root_folder +
            '/data/kinematics/sample_probing_olecrane_R.n3d', 'ForearmR')

        config['VirtualMarkers']['RadialStyloidR'] = process_probing_acquisition(
            ktk.config.root_folder +
            '/data/kinematics/sample_probing_radial_styloid_R.n3d', 'ForearmR')

        config['VirtualMarkers']['UlnarStyloidR'] = process_probing_acquisition(
            ktk.config.root_folder +
            '/data/kinematics/sample_probing_ulnar_styloid_R.n3d', 'ForearmR')

        # Process an experimental trial
        markers = ktk.kinematics.read_n3d_file(
            ktk.config.root_folder +
            '/data/kinematics/sample_propulsion.n3d', labels=marker_names)

        rigid_bodies = ktk.kinematics.register_markers(
            markers, config['RigidBodies'])

        for virtual_marker in config['VirtualMarkers']:
            local_coordinates = config['VirtualMarkers'][virtual_marker]['LocalPoint']
            rigid_body_name = config['VirtualMarkers'][virtual_marker]['RigidBodyName']
            reference_frame = rigid_bodies.data[rigid_body_name]

            markers.data[virtual_marker] = ktk.geometry.get_global_coordinates(
                local_coordinates, reference_frame)
            # Assign a color for these virtual markers
            markers = markers.add_data_info(virtual_marker, 'Color', 'c')


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
