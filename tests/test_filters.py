#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020-2022 Félix Chénier

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for Kinetics Toolkit's filters modules."""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2021 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import kineticstoolkit as ktk
import numpy as np
import warnings


def test_savgol():
    """Test savgol filter."""
    # Let define a TimeSeries were data1 is time^2 and data2 is time^4.
    time = np.linspace(0, 1, 100)

    tsin = ktk.TimeSeries(time=time)
    tsin.data["data1"] = time**2
    tsin.data["data2"] = np.hstack(
        [time[:, np.newaxis] ** 2, time[:, np.newaxis] ** 4]
    )

    # Using `ktk.filters.savgol`, we can smooth or derivate these data.
    # Smooth:
    y = ktk.filters.savgol(tsin, window_length=3, poly_order=2, deriv=0)
    # 1st derivative:
    doty = ktk.filters.savgol(tsin, window_length=3, poly_order=2, deriv=1)
    # 2nd derivative:
    ddoty = ktk.filters.savgol(tsin, window_length=3, poly_order=2, deriv=2)

    tol = 5e-3  # Numerical tolerance (large because I compare a filtered
    # signal with a non-filtered signal).

    assert np.max(np.abs(y.data["data1"][1:-2] - time[1:-2] ** 2)) < tol
    assert np.max(np.abs(doty.data["data1"][1:-2] - 2 * time[1:-2])) < tol
    assert np.max(np.abs(ddoty.data["data1"][1:-2] - 2)) < tol

    assert np.max(np.abs(y.data["data2"][1:-2, 0] - time[1:-2] ** 2)) < tol
    assert np.max(np.abs(doty.data["data2"][1:-2, 0] - 2 * time[1:-2])) < tol
    assert np.max(np.abs(ddoty.data["data2"][1:-2, 0] - 2)) < tol

    assert np.max(np.abs(y.data["data2"][1:-2, 1] - time[1:-2] ** 4)) < tol
    assert (
        np.max(np.abs(doty.data["data2"][1:-2, 1] - 4 * (time[1:-2] ** 3)))
        < tol
    )
    assert (
        np.max(np.abs(ddoty.data["data2"][1:-2, 1] - 12 * time[1:-2] ** 2))
        < tol
    )

    # Test if it still works with nans in data
    tsin.data["data2"][10, 1] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = ktk.filters.savgol(tsin, window_length=3, poly_order=2, deriv=0)
        doty = ktk.filters.savgol(tsin, window_length=3, poly_order=2, deriv=1)
        ddoty = ktk.filters.savgol(
            tsin, window_length=3, poly_order=2, deriv=2
        )

    # Remove points that I don't want to compare because they can't be valid
    # (begin and end points, plus points around the original nan)
    y.data["data2"][0:2, 1] = np.nan
    y.data["data2"][-2:, 1] = np.nan
    y.data["data2"][8:13, 1] = np.nan

    tokeep = ~y.isnan("data2")

    assert np.all(np.abs(y.data["data1"][tokeep] - time[tokeep] ** 2) < tol)
    assert np.all(np.abs(doty.data["data1"][tokeep] - 2 * time[tokeep]) < tol)
    assert np.all(np.abs(ddoty.data["data1"][tokeep] - 2) < tol)

    assert np.all(np.abs(y.data["data2"][tokeep, 0] - time[tokeep] ** 2) < tol)
    assert np.all(
        np.abs(doty.data["data2"][tokeep, 0] - 2 * time[tokeep]) < tol
    )
    assert np.all(np.abs(ddoty.data["data2"][tokeep, 0] - 2) < tol)

    assert np.all(np.abs(y.data["data2"][tokeep, 1] - time[tokeep] ** 4) < tol)
    assert np.all(
        np.abs(doty.data["data2"][tokeep, 1] - 4 * (time[tokeep] ** 3)) < tol
    )
    assert np.all(
        np.abs(ddoty.data["data2"][tokeep, 1] - 12 * time[tokeep] ** 2) < tol
    )


def test_smooth():
    """Test smooth."""
    # Let define a TimeSeries with some data inside:
    data = np.array(
        [
            0.7060,
            0.0318,
            0.2769,
            0.0462,
            0.0971,
            0.8235,
            0.6948,
            0.3171,
            0.9502,
            0.0344,
            0.4387,
            0.3816,
            0.7655,
            0.7952,
            0.1869,
            0.4898,
            0.4456,
            0.6463,
            0.7094,
            0.7547,
            0.2760,
            0.6797,
            0.6551,
            0.1626,
            0.1190,
            0.4984,
            0.9597,
            0.3404,
            0.5853,
            0.2238,
        ]
    )

    ts = ktk.TimeSeries()
    ts.data["data"] = data
    ts.time = np.linspace(0, 1, len(data))

    # Now we smooth this function using a moving average on 5 samples.
    y = ktk.filters.smooth(ts, 5)

    tol = 1e-10  # Numerical tolerance

    # Test that if filters well
    assert np.abs(np.mean(ts.data["data"][4:8] - y.data["data"][6] < tol))

    # Test if it filters at all
    assert np.abs(np.mean(ts.data["data"][6] - y.data["data"][6] > tol))

    # Test if it works with nan
    ts.data["data"][9] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = ktk.filters.smooth(ts, 5)

    # Test that if filters well
    assert np.abs(np.mean(ts.data["data"][4:8] - y.data["data"][6] < tol))
    # Test if it filters at all
    assert np.abs(np.mean(ts.data["data"][6] - y.data["data"][6] > tol))


def test_butter():
    """Test butter."""
    # Let define a TimeSeries with a sinusoidal signal at 1 Hz, with an
    # amplitude of 1.
    ts = ktk.TimeSeries(time=np.linspace(0, 30, 1000))
    ts.data["data"] = np.sin(2 * np.pi * ts.time)

    # Add an ndimensional data with NaNs
    ts.data["data2"] = np.hstack(
        [ts.data["data"][:, np.newaxis], ts.time[:, np.newaxis]]
    )
    ts.data["data2"][400, 1] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # We filter at 1 Hz, with an order 1
        new_ts = ktk.filters.butter(ts, 1, order=1)
        new_ts_highpass = ktk.filters.butter(ts, 1, order=1, btype="highpass")

    # Verify that the new RMS value is the half ot the first
    data1 = ts.data["data"][300:700]
    data2 = new_ts.data["data"][300:700]
    data3 = new_ts_highpass.data["data"][300:700]

    assert (
        np.abs(np.sqrt(np.sum(data2**2)) - np.sqrt(np.sum(data1**2)) / 2)
        < 0.001
    )

    assert (
        np.abs(np.sqrt(np.sum(data3**2)) - np.sqrt(np.sum(data1**2)) / 2)
        < 0.001
    )

    # Verify that filtering a ramp gives a ramp, that filtering
    # an ndimensional data words, and that NaNs in data work.
    tokeep = ~ts.isnan("data2")
    assert np.all(
        np.abs(new_ts.data["data2"][tokeep, 0] - new_ts.data["data"][tokeep])
        < 1e-3
    )
    tokeep[0:100] = False
    tokeep[-100:] = False
    assert np.all(
        np.abs(new_ts.data["data2"][tokeep, 1] - ts.time[tokeep]) < 1e-3
    )

    # Verify that keeping only a narrow range of requencies from a white noise
    # resemble at sinusoidal signal of that midfrequency
    ts = ktk.TimeSeries(time=np.linspace(0, 30, 1000))
    np.random.seed(0)
    ts.data["data"] = np.random.rand(1000)
    new_ts = ktk.filters.butter(ts, fc=[3, 3.5], btype="bandpass")
    # new_ts.plot()  # Checked visually once, ensure that it doesn't change
    assert (
        np.abs(np.sum(new_ts.data["data"] ** 2) - 1.5161649322350133) < 1e-12
    )


def test_median():
    """Test median filter."""
    ts = ktk.TimeSeries(time=np.arange(0, 0.5, 0.1))

    # Test on 1-dimensional data (from doctstring)
    ts.data["data1"] = np.array([10.0, 11.0, 11.0, 20.0, 14.0, 15.0])

    # Test on 2-dimensional data
    ts.data["data2"] = np.array(
        [
            [0.0, 10.0],
            [0.0, 11.0],
            [1.0, 11.0],
            [1.0, 20.0],
            [2.0, 14.0],
            [2.0, 15.0],
        ]
    )

    ts = ktk.filters.median(ts)

    assert np.all(
        np.abs(
            ts.data["data1"] - np.array([10.0, 11.0, 11.0, 14.0, 15.0, 15.0])
        )
        < 1e-16
    )

    assert np.all(
        np.abs(
            ts.data["data2"]
            - np.array(
                [
                    [0.0, 10.0],
                    [0.0, 11.0],
                    [1.0, 11.0],
                    [1.0, 14.0],
                    [2.0, 15.0],
                    [2.0, 15.0],
                ]
            )
        )
        < 1e-16
    )


def test_deriv():
    """Test the deriv filter."""
    ts = ktk.TimeSeries(time=np.arange(0, 0.5, 0.1))
    ts.data["data"] = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    ts.data["data2"] = np.array(
        [
            [0.0, 0.0],  # ndimensional array
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ]
    )

    # First derivative
    ts1 = ktk.filters.deriv(ts)

    assert np.all(
        np.abs(ts1.time - np.array([0.05, 0.15, 0.25, 0.35])) < 1e-12
    )
    assert np.all(
        np.abs(ts1.data["data"] - np.array([0.0, 10.0, 0.0, -10.0])) < 1e-12
    )
    assert np.all(
        np.abs(
            ts1.data["data2"]
            - np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 0.0], [-10.0, 0.0]])
        )
        < 1e-12
    )

    # Second derivative
    ts2 = ktk.filters.deriv(ts, n=2)

    assert np.all(np.abs(ts2.time - np.array([0.1, 0.2, 0.3])) < 1e-12)
    assert np.all(
        np.abs(ts2.data["data"] - np.array([100.0, -100.0, -100.0])) < 1e-12
    )
    assert np.all(
        np.abs(
            ts2.data["data2"]
            - np.array([[100.0, 0.0], [-100.0, 0.0], [-100.0, 0.0]])
        )
        < 1e-12
    )


def test_validate_input():
    ts = ktk.TimeSeries(
        time=np.array([0, 0.1, 0.2]), data={"Data": np.array([0, 0.1, 0.2])}
    )
    ts1 = ktk.filters.smooth(ts, window_length=3)
    # Assert no error

    ts.time = np.array([0, 0.2, 0.1])
    try:
        ts2 = ktk.filters.smooth(ts, window_length=3)
        raise AssertionError("This should have raised a ValueError")
    except ValueError:
        pass


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
