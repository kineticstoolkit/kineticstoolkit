#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright Félix Chénier 2020
"""
Tests for ktk.Player
"""
import kineticstoolkit as ktk
import matplotlib.pyplot as plt
import warnings


def test_instanciate_and_to_html5():
    """Test that instanciating a Player does not crash."""

    # Load markers
    kinematics = ktk.load(
        ktk.doc.download("inversedynamics_kinematics.ktk.zip")
    )

    kinematics = kinematics["Kinematics"]

    # The player can be instanciated to show markers
    pl = ktk.Player(kinematics["Markers"], target=[-5, 0, 0])
    plt.pause(0.01)
    pl.close()

    # The player can be instanciated to show rigid bodies
    pl = ktk.Player(kinematics["ReferenceFrames"], target=[-5, 0, 0])
    plt.pause(0.01)
    pl.close()

    # Or the player can be instanciated to show both markers and rigid bodies
    pl = ktk.Player(
        kinematics["Markers"],
        kinematics["ReferenceFrames"],
        target=[-5, 0, 0],
        up="z",
    )
    plt.pause(0.01)

    # Test that to_html5 doesn't crash
    assert pl.to_html5() is not None

    pl.close()


def test_issue137():
    """
    Player should complain, but not fail, if some TimeSeries are not Nx4 or
    Nx4x4
    """

    # Load markers
    kinematics = ktk.load(
        ktk.doc.download("inversedynamics_kinematics.ktk.zip")
    )

    kinematics = kinematics["Kinematics"]["Markers"]
    kinematics.data["test"] = kinematics.time

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        pl = ktk.Player(kinematics)  # Shouldn't crash
        plt.pause(0.01)
        pl.close()


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
