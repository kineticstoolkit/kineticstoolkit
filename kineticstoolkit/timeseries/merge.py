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
"""Implement the TimeSeries merge method."""

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


def _merge(
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
    """Implement TimeSeries.merge."""
    if data_keys is None:
        data_keys = []
    try:
        check_param("data_keys", data_keys, str)
    except TypeError:
        try:
            data_keys = list(data_keys)
            check_param("data_keys", data_keys, list, contents_type=str)
        except TypeError:
            raise TypeError("data_keys must be a string or a list of strings.")
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
