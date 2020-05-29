#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright Félix Chénier 2020
"""
Tests for ktk.Player
"""
import ktk
import matplotlib.pyplot as plt

def test_instanciate():
    """Test that instanciating a Player does not crash."""

    # Load markers
    kinematics = ktk.kinematics.read_c3d_file(
        ktk.config.root_folder +
        '/tutorials/data/sample_OptiTrack_walking.c3d')

    # The player can be instanciated to show markers
    pl = ktk.Player(kinematics, target=[-5, 0, 0])
    plt.pause(0.01)
    pl.close()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])