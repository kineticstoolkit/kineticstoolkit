#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright Félix Chénier 2020
"""
Tests for ktk.Player
"""
import kineticstoolkit as ktk
import matplotlib.pyplot as plt


def test_instanciate_and_to_html5():
    """Test that instanciating a Player does not crash."""

    # Load markers
    kinematics = ktk.load(
        ktk.config.root_folder +
        '/data/inversedynamics/basketball_kinematics.ktk.zip')

    kinematics = kinematics['Kinematics']

    # The player can be instanciated to show markers
    pl = ktk.Player(kinematics['Markers'], target=[-5, 0, 0])
    plt.pause(0.01)
    pl.close()

    # The player can be instanciated to show rigid bodies
    pl = ktk.Player(kinematics['ReferenceFrames'],
                    target=[-5, 0, 0])
    plt.pause(0.01)
    pl.close()

    # Or the player can be instanciated to show both markers and rigid bodies
    pl = ktk.Player(kinematics['Markers'],
                    kinematics['ReferenceFrames'],
                    target=[-5, 0, 0])
    plt.pause(0.01)

    # Test that to_html5 doesn't crash
    assert pl.to_html5() is not None

    pl.close()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
