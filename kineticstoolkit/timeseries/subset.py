#!/usr/bin/env python3
#
# Copyright 2020-2025 Félix Chénier

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Provide subset and merging methods for TimeSeries."""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2025 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import warnings
from copy import deepcopy

import numpy as np

from kineticstoolkit.exceptions import TimeSeriesMergeConflictError
from kineticstoolkit.typing_ import TYPE_CHECKING, check_param

if TYPE_CHECKING:
    from kineticstoolkit import TimeSeries


def get_subset(self, data_keys: str | list[str]) -> "TimeSeries":
    """
    Return a subset of the TimeSeries.

    This method returns a TimeSeries that contains only selected data
    keys. Events and info are also copied in the new TimeSeries.

    Parameters
    ----------
    data_keys
        The data keys to extract from the TimeSeries.

    Returns
    -------
    TimeSeries
        The TimeSeries, minus the unspecified data keys.

    Raises
    ------
    KeyError
        If one or more data keys could not be found in the TimeSeries
        data.

    See Also
    --------
    ktk.TimeSeries.merge

    Example
    -------
        >>> ts = ktk.TimeSeries(time=np.arange(10))
        >>> ts = ts.add_data("signal1", ts.time)
        >>> ts = ts.add_data("signal2", ts.time**2)
        >>> ts = ts.add_data("signal3", ts.time**3)
        >>> ts.data.keys()
        dict_keys(['signal1', 'signal2', 'signal3'])

        >>> ts2 = ts.get_subset(["signal1", "signal3"])
        >>> ts2.data.keys()
        dict_keys(['signal1', 'signal3'])

    """
    try:
        check_param("data_keys", data_keys, str)
    except TypeError:
        try:
            check_param("data_keys", data_keys, list, contents_type=str)
        except TypeError as e:
            raise TypeError(
                "data_keys must be a string or a list of strings."
            ) from e
    self._check_well_shaped()

    if isinstance(data_keys, str):
        data_keys = [data_keys]

    ts = self.copy(
        copy_time=True, copy_data=False, copy_info=True, copy_events=True
    )

    for key in data_keys:
        try:
            ts.data[key] = self.data[key].copy()
        except KeyError as e:
            raise KeyError(
                f"The key '{key}' could not be found among the "
                f"{len(self.data)} data entries of the TimeSeries"
            ) from e

    return ts


def _merge_resample_if_needed(
    ts_in: "TimeSeries", ts_out: "TimeSeries", resample: bool
) -> None:
    """Resample ts_in if needed."""
    if (ts_out.time.shape == ts_in.time.shape) and np.all(
        ts_out.time == ts_in.time
    ):
        must_resample = False
    else:
        must_resample = True

    if must_resample is True and resample is False:
        raise ValueError(
            "Time attributes do not match, resampling is required."
        )

    if must_resample is True:
        ts_in.resample(ts_out.time, in_place=True)


def _merge_data(
    ts_in: "TimeSeries",
    ts_out: "TimeSeries",
    data_keys: list[str],
    on_conflict: str,
    overwrite: bool,
) -> None:
    """Merge data from ts_in into ts_out."""
    for key in data_keys:
        if key not in ts_out.data:
            # No conflict
            ts_out.add_data(key, ts_in.data[key], in_place=True)
        elif on_conflict == "error":
            # Conflict, and we need to raise
            raise TimeSeriesMergeConflictError(
                f"The key '{key}' exists in both TimeSeries's data. "
            )
        elif on_conflict == "warning":
            # Conflict, and we need to warn
            if overwrite:
                ts_out.add_data(
                    key, ts_in.data[key], overwrite=True, in_place=True
                )
                warnings.warn(
                    f"The key '{key}' exists in both TimeSeries's data. "
                    "According to the overwrite=True "
                    "parameter, its prior value has been overwritten "
                    "by the new value. Use on_conflict='mute' to mute "
                    "this warning."
                )
            else:
                warnings.warn(
                    f"The key '{key}' exists in both TimeSeries's data. "
                    "According to the overwrite=False "
                    "parameter, its prior value has been preserved. "
                    "Use on_conflict='mute' to mute this warning."
                )
        # Conflict, and we need to not warn.
        elif overwrite:
            ts_out.add_data(
                key, ts_in.data[key], overwrite=True, in_place=True
            )


def _merge_info(
    ts_in: "TimeSeries",
    ts_out: "TimeSeries",
    data_keys: list[str],
    on_conflict: str,
    overwrite: bool,
):
    """Merge info from ts_in into ts_out."""
    for outer_key in ts_in.info:
        for inner_key in ts_in.info[outer_key]:
            if outer_key not in ts_out.info:
                # No conflict
                ts_out.add_info(
                    outer_key,
                    inner_key,
                    ts_in.info[outer_key][inner_key],
                    in_place=True,
                )
            elif inner_key not in ts_out.info[outer_key]:
                # No conflict
                ts_out.add_info(
                    outer_key,
                    inner_key,
                    ts_in.info[outer_key][inner_key],
                    in_place=True,
                )
            elif (
                ts_out.info[outer_key][inner_key]
                == ts_in.info[outer_key][inner_key]
            ):
                # Duplicate data, but it's the same, so there's no
                # conflict and thus nothing to do.
                pass
            elif on_conflict == "error":
                # Conflict, and we need to raise
                raise TimeSeriesMergeConflictError(
                    f"The key '{inner_key}' exists in both "
                    f"TimeSeries's attribute info[{outer_key}]."
                )

            elif on_conflict == "warning":
                # Conflict, and we need to warn
                if overwrite:
                    ts_out.add_info(
                        outer_key,
                        inner_key,
                        ts_in.info[outer_key][inner_key],
                        overwrite=True,
                        in_place=True,
                    )
                    warnings.warn(
                        f"The key '{inner_key}' exists in both "
                        f"TimeSeries's attribute info[{outer_key}]. "
                        "According to the overwrite=True "
                        "parameter, its prior value has been overwritten "
                        "by the new value. Use on_conflict='mute' to mute "
                        "this warning."
                    )
                else:
                    warnings.warn(
                        f"The key '{inner_key}' exists in both "
                        f"TimeSeries's attribute info[{outer_key}]. "
                        "According to the overwrite=False "
                        "parameter, its prior value has been preserved. "
                        "Use on_conflict='mute' to mute this warning."
                    )

            # Conflict, and we need to not warn.
            elif overwrite:
                ts_out.add_info(
                    outer_key,
                    inner_key,
                    ts_in.info[outer_key][inner_key],
                    overwrite=True,
                    in_place=True,
                )


def merge(
    self,
    ts: "TimeSeries",
    data_keys: str | list[str] | None = None,
    *,
    resample: bool = False,
    merge_events: bool = True,
    merge_info: bool = True,
    overwrite: bool = False,
    on_conflict: str = "warning",
    in_place: bool = False,
) -> "TimeSeries":
    """
    Merge the TimeSeries with another TimeSeries.

    Parameters
    ----------
    ts
        The TimeSeries to merge into the current TimeSeries.
    data_keys
        Optional. The data keys to merge from ts. If left empty, all the
        data keys are merged.
    resample
        Optional. Set to True to resample the source TimeSeries to the
        target one using a linear interpolation. If the time attributes are
        not equivalent and resample is False, an exception is raised. To
        resample using other methods than linear interpolation, please
        resample the source TimeSeries manually before, using
        TimeSeries.resample. Default is False.
    merge_events
        Optional. Set to True to also merge events. Default is True.
    merge_info
        Optional. Set to True to also merge info. Default is True.
    overwrite
        Optional. Select what to do if a data or info key from the source
        TimeSeries already exists in the destination TimeSeries. True to
        overwrite the already existing value, False to ignore the new value.
        Default is False.
    on_conflict
        Optional. Select the warning level when a data or info key
        from the source TimeSeries already exists in the destination
        TimeSeries. May take the following values:
        "mute": No warning;
        "warning": Warns that duplicate keys were found and how the
        conflict has been resolved following the `overwrite` parameter.
        "error": Raises a TimeSeriesMergeConflictError.
        Default is "warning".
    in_place
        Optional. True to modify and return the original TimeSeries. False
        to return a modified copy of the TimeSeries while leaving the
        original TimeSeries intact. Default is False.

    Returns
    -------
    TimeSeries
        The merged TimeSeries.

    Raises
    ------
    TimeSeriesMergeConflictError
        If a data or info key from the source TimeSeries already exists in
        the destination TimeSeries and on_conflict is set to "error".

    See Also
    --------
    ktk.TimeSeries.get_subset
    ktk.TimeSeries.resample

    """
    if data_keys is None:
        data_keys = []
    try:
        check_param("data_keys", data_keys, str)
    except TypeError:
        try:
            data_keys = list(data_keys)
            check_param("data_keys", data_keys, list, contents_type=str)
        except TypeError as e:
            raise TypeError(
                "data_keys must be a string or a list of strings."
            ) from e
    check_param("resample", resample, bool)
    check_param("overwrite", overwrite, bool)
    check_param("on_conflict", on_conflict, str)
    if on_conflict not in ["mute", "warning", "error"]:
        raise ValueError(
            "Parameter on_conflict must be either 'mute', 'warning' or "
            "'error'."
        )
    check_param("in_place", in_place, bool)
    self._check_well_shaped()
    ts._check_well_shaped()
    # --

    ts_out = self if in_place else self.copy()
    ts = ts.copy()
    if len(data_keys) == 0:
        data_keys = list(ts.data.keys())
    elif isinstance(data_keys, str):
        data_keys = [data_keys]

    if len(ts_out.time) == 0:
        ts_out.time = deepcopy(ts.time)

    # Check if resampling is needed
    _merge_resample_if_needed(ts, ts_out, resample)

    # Merge data
    _merge_data(ts, ts_out, data_keys, on_conflict, overwrite)

    # Merge info
    if merge_info:
        _merge_info(ts, ts_out, data_keys, on_conflict, overwrite)

    # Merge events
    if merge_events:
        for event in ts.events:
            ts_out.add_event(
                event.time, event.name, in_place=True, unique=True
            )

    return ts_out
