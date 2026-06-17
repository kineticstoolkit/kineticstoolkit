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
"""Provide graphical user interface methods for TimeSeries."""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2025 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import warnings
from copy import deepcopy
from typing import Any

import limitedinteraction as li
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from kineticstoolkit.gui import button_dialog, message
from kineticstoolkit.tools import check_interactive_backend
from kineticstoolkit.typing_ import TYPE_CHECKING, check_param

if TYPE_CHECKING:
    from kineticstoolkit import TimeSeries


WINDOW_PLACEMENT = {"top": 50, "right": 0}
MAX_CURVES_TO_TRY_BEST_LEGEND_PLACEMENT = 20


def ui_edit_events(
    self,
    name: str | list[str] = [],
    data_keys: str | list[str] = [],
    legend: bool = True,
    max_lines: int = 40,
) -> "TimeSeries":  # pragma: no cover
    """
    Edit events interactively.

    Parameters
    ----------
    name
        Optional. The name of the event(s) to add. May be a string
        or a list of strings. These events appear on their own buttons
        "add `name`". Event names can also be defined interactively.
    data_keys
        Optional. A signal name of list of signal name to be plotted,
        similar to the data_keys argument of ktk.TimeSeries.plot.
    legend
        Optional. True to plot a legend, False otherwise. Default is True.
    max_lines
        Optional. The maximal number of lines to plot. Default is 40. A
        warning is issued if plotting all the data would require more
        lines.

    Returns
    -------
    TimeSeries
        The TimeSeries with the modified events. If the operation was
        cancelled by the user, this is the original TimeSeries.

    Warning
    -------
    This function, which has been introduced in 0.6, is still experimental
    and may change signature or behaviour in the future.

    See Also
    --------
    ktk.TimeSeries.add_event
    ktk.TimeSeries.rename_event
    ktk.TimeSeries.remove_event
    ktk.TimeSeries.trim_events

    Note
    ----
    Matplotlib must be in interactive mode for this function to work.

    """
    check_interactive_backend()

    try:
        check_param("name", name, str)
    except TypeError:
        try:
            check_param("name", name, list, contents_type=str)
        except TypeError:
            raise TypeError("name must be a string or a list of strings.")
    try:
        check_param("data_keys", data_keys, str)
    except TypeError:
        try:
            check_param("data_keys", data_keys, list, contents_type=str)
        except TypeError:
            raise TypeError("data_keys must be a string or a list of strings.")
    check_param("legend", legend, bool)
    check_param("max_lines", max_lines, int)
    self._check_well_shaped()
    self._check_not_empty_time()
    self._check_not_empty_data()

    def add_this_event(ts: "TimeSeries", name: str) -> "TimeSeries":
        message("Place the event on the figure.", **WINDOW_PLACEMENT)
        this_time = plt.ginput(1)[0][0]
        ts = ts.add_event(this_time, name)
        message("")
        return ts

    def get_event_index(ts: "TimeSeries") -> int:
        message("Select an event on the figure.", **WINDOW_PLACEMENT)
        this_time = plt.ginput(1)[0][0]
        event_times = np.array([event.time for event in ts.events])
        message("")
        return int(np.argmin(np.abs(event_times - this_time)))

    # Set Matplotlib interactive mode
    isinteractive = plt.isinteractive()
    plt.ion()

    ts = self.copy()

    if isinstance(name, str):
        event_names = [name]
    else:
        event_names = deepcopy(name)

    fig = plt.figure()
    ts.plot(
        data_keys,
        _raise_on_no_data=True,
        legend=legend,
        max_lines=max_lines,
    )

    while True:
        # Populate the choices to the user
        choices = [f"Add '{s}'" for s in event_names]

        choice_index = {}
        choice_index["add"] = len(choices)
        if len(event_names) == 0:
            choices.append("Add event")
        else:
            choices.append("Add event with another name")

        if len(ts.events) > 0:
            choice_index["remove"] = len(choices)
            choices.append("Remove event")

        if len(ts.events) > 0:
            choice_index["remove_all"] = len(choices)
            choices.append("Remove all events")

            choice_index["move"] = len(choices)
            choices.append("Move event")

        choice_index["close"] = len(choices)
        choices.append("Save and close")

        choice_index["cancel"] = len(choices)
        choices.append("Cancel")

        # Show the button dialog
        choice = button_dialog(
            "Move and zoom on the figure,\nthen select an option below.",
            choices,
            **WINDOW_PLACEMENT,
        )

        # Execute
        if choice < choice_index["add"]:
            ts = add_this_event(ts, event_names[choice])

        elif choice == choice_index["add"]:
            event_names.append(
                li.input_dialog(
                    "Please enter the event name:", **WINDOW_PLACEMENT
                )
            )
            # Add this event name to the list of recently added events
            if len(event_names) > 5:
                event_names = event_names[-5:]

            # Add the event
            ts = add_this_event(ts, event_names[-1])

        elif ("remove" in choice_index) and (choice == choice_index["remove"]):
            event_index = get_event_index(ts)
            try:
                ts.events.pop(event_index)
            except IndexError:
                li.button_dialog(
                    "No event was removed.",
                    choices=["OK"],
                    icon="error",
                    **WINDOW_PLACEMENT,
                )

        elif ("remove_all" in choice_index) and (
            choice == choice_index["remove_all"]
        ):
            if (
                li.button_dialog(
                    "Do you really want to remove all events from this "
                    "TimeSeries?",
                    ["Yes, remove all events", "No"],
                    icon="alert",
                    **WINDOW_PLACEMENT,
                )
                == 0
            ):
                ts.events = []

        elif ("move" in choice_index) and (choice == choice_index["move"]):
            event_index = get_event_index(ts)
            event_name = ts.events[event_index].name
            try:
                ts.events.pop(event_index)
                ts = add_this_event(ts, event_name)
            except IndexError:
                li.button_dialog(
                    "Could not move this event.",
                    choices=["OK"],
                    icon="error",
                    **WINDOW_PLACEMENT,
                )

        elif ("close" in choice_index) and (choice == choice_index["close"]):
            plt.close(fig)
            if not isinteractive:
                plt.ioff()
            return ts

        elif (choice == -1) or (
            ("cancel" in choice_index) and (choice == choice_index["cancel"])
        ):
            plt.close(fig)
            if not isinteractive:
                plt.ioff()
            return self.copy()

        # Refresh
        ts.remove_duplicate_events(in_place=True)
        axes = plt.axis()
        plt.cla()
        ts.plot(
            data_keys,
            legend=legend,
            max_lines=max_lines,
            _raise_on_no_data=True,
        )
        plt.axis(axes)


def _ui_sync_one_timeseries(
    ts: "TimeSeries",
    data_keys: str | list[str],
    legend: bool,
    max_lines: int,
    fig: mpl.figure.Figure,
) -> "TimeSeries":
    """Provide GUI for syncing only one TimeSeries."""
    ts.plot(data_keys, legend=legend, max_lines=max_lines)
    choice = button_dialog(
        "Please zoom on the time zero and press Next.",
        ["Cancel", "Next"],
        **WINDOW_PLACEMENT,
    )
    if choice != 1:
        plt.close(fig)
        return ts

    message("Click on the sync event.", **WINDOW_PLACEMENT)
    click = plt.ginput(1)
    message("")
    plt.close(fig)
    return ts.shift(-click[0][0])


def _ui_sync_plot_two_timeseries(
    ts1: "TimeSeries",
    data_keys: str | list[str],
    ts2: "TimeSeries",
    data_keys2: str | list[str],
    legend: bool,
    max_lines: int,
    axes: list[Any],
    fig: mpl.figure.Figure,
) -> None:
    """Plot the two TimeSeries to be synced."""
    if len(axes) == 0:
        axes.append(fig.add_subplot(2, 1, 1))
        axes.append(fig.add_subplot(2, 1, 2, sharex=axes[0]))

    plt.sca(axes[0])
    axes[0].cla()
    ts1.plot(data_keys, legend=legend, max_lines=max_lines)
    plt.title("First TimeSeries (ts1)")
    plt.grid(True)
    plt.tight_layout()

    plt.sca(axes[1])
    axes[1].cla()
    ts2.plot(data_keys2, legend=legend, max_lines=max_lines)
    plt.title("Second TimeSeries (ts2)")
    plt.grid(True)
    plt.tight_layout()


def _ui_sync_two_timeseries(
    ts1: "TimeSeries",
    data_keys: str | list[str],
    ts2: "TimeSeries",
    data_keys2: str | list[str],
    legend: bool,
    max_lines: int,
    fig: mpl.figure.Figure,
) -> tuple["TimeSeries", "TimeSeries"]:
    """Provide GUI for syncing two TimeSeries."""
    finished = False
    # list of axes:
    axes: list[Any] = []

    while finished is False:
        _ui_sync_plot_two_timeseries(
            ts1, data_keys, ts2, data_keys2, legend, max_lines, axes, fig
        )

        choices = [
            "Zero ts1 only, using ts1",
            "Zero ts2 only, using ts2",
            "Zero both TimeSeries, using ts1",
            "Zero both TimeSeries, using ts2",
            "Sync both TimeSeries on a common event",
            "Finished",
        ]

        choice = button_dialog(
            "Please select an option.",
            choices=choices,
            **WINDOW_PLACEMENT,
        )

        if choice == choices.index("Zero ts1 only, using ts1"):
            message("Click on the time zero in ts1.", **WINDOW_PLACEMENT)
            click_1 = plt.ginput(1)
            message("")

            ts1 = ts1.shift(-click_1[0][0])

        elif choice == choices.index("Zero ts2 only, using ts2"):
            message("Click on the time zero in ts2.", **WINDOW_PLACEMENT)
            click_1 = plt.ginput(1)
            message("")

            ts2 = ts2.shift(-click_1[0][0])

        elif choice == choices.index("Zero both TimeSeries, using ts1"):
            message("Click on the time zero in ts1.", **WINDOW_PLACEMENT)
            click_1 = plt.ginput(1)
            message("")

            ts1 = ts1.shift(-click_1[0][0])
            ts2 = ts2.shift(-click_1[0][0])

        elif choice == choices.index("Zero both TimeSeries, using ts2"):
            message("Click on the time zero in ts2.", **WINDOW_PLACEMENT)
            click_2 = plt.ginput(1)
            message("")

            ts1 = ts1.shift(-click_2[0][0])
            ts2 = ts2.shift(-click_2[0][0])

        elif choice == choices.index("Sync both TimeSeries on a common event"):
            message("Click on the sync event in ts1.", **WINDOW_PLACEMENT)
            click_1 = plt.ginput(1)
            message(
                "Now click on the same event in ts2.",
                **WINDOW_PLACEMENT,
            )
            click_2 = plt.ginput(1)
            message("")

            ts1 = ts1.shift(-click_1[0][0])
            ts2 = ts2.shift(-click_2[0][0])

        elif (
            choice == choices.index("Finished") or choice < -1
        ):  # OK or closed figure, quit.
            plt.close(fig)
            finished = True

    return (ts1, ts2)


def ui_sync(
    self,
    data_keys: str | list[str] | None = None,
    ts2: "TimeSeries | None" = None,
    data_keys2: str | list[str] | None = None,
    legend: bool = True,
    max_lines: int = 40,
) -> "TimeSeries":  # pragma: no cover
    """
    Synchronize one or two TimeSeries by shifting their time.

    If this method is called on only one TimeSeries, an interactive
    interface asks the user to click on the time to set to zero.

    If another TimeSeries is given, an interactive interface allows
    synchronizing both TimeSeries together.

    Parameters
    ----------
    data_keys
        Optional. The data keys to plot. If empty, all data is plotted.
    ts2
        Optional. A second TimeSeries to be synced to the first one. This
        TimeSeries is modified in place.
    data_keys2
        Optional. The data keys from the second TimeSeries to plot. If
        empty, all data is plotted.
    legend
        Optional. True to plot a legend, False otherwise. Default is True.
    max_lines
        Optional. The maximal number of lines to plot. Default is 40.
        A warning is issued if plotting all the data would require more
        lines.

    Returns
    -------
    TimeSeries
        The TimeSeries after synchronization.

    Warning
    -------
    This function, which has been introduced in 0.1, is still experimental
    and may change signature or behaviour in the future.

    See Also
    --------
    ktk.TimeSeries.shift

    Notes
    -----
    Matplotlib must be in interactive mode for this method to work.

    """
    check_interactive_backend()

    if data_keys is None:
        data_keys = []
    if data_keys2 is None:
        data_keys2 = []

    try:
        check_param("data_keys", data_keys, str)
    except TypeError:
        try:
            check_param("data_keys", data_keys, list, contents_type=str)
        except TypeError as e:
            raise TypeError(
                "data_keys must be a string or a list of strings."
            ) from e
    try:
        check_param("data_keys2", data_keys2, str)
    except TypeError:
        try:
            check_param("data_keys2", data_keys2, list, contents_type=str)
        except TypeError as e:
            raise TypeError(
                "data_keys2 must be a string or a list of strings."
            ) from e
    check_param("legend", legend, bool)
    check_param("max_lines", max_lines, int)

    self._check_well_shaped()
    self._check_not_empty_time()
    self._check_not_empty_data()

    if ts2 is not None:
        ts2._check_well_shaped()
        ts2._check_not_empty_time()
        ts2._check_not_empty_data()

    ts1 = self.copy()

    fig = plt.figure("ktk.TimeSeries.ui_sync")

    if ts2 is None:
        # Synchronize ts1 only
        ts1 = _ui_sync_one_timeseries(ts1, data_keys, legend, max_lines, fig)

    else:  # Sync two TimeSeries together
        ts1, new_ts2 = _ui_sync_two_timeseries(
            ts1, data_keys, ts2, data_keys2, legend, max_lines, fig
        )
        ts2.time = new_ts2.time
        ts2.data = new_ts2.data
        ts2.info = new_ts2.info
        ts2.events = new_ts2.events

    return ts1


def _plot_curves(
    ts: "TimeSeries", max_lines: int, args, kwargs
) -> mpl.axes._axes.Axes:
    """Plot the curves of the input TimeSeries on the given axes."""
    df = ts.to_dataframe()
    labels = df.columns.to_list()

    axes = plt.gca()
    # Don't know why I need to disable mypy on these lines.
    axes.set_prop_cycle(
        mpl.cycler(linewidth=[1, 2, 3, 4])  # type: ignore
        * mpl.cycler(linestyle=["-", "--", "-.", ":"])  # type: ignore
        * plt.rcParams["axes.prop_cycle"]
    )

    for i_label, label in enumerate(labels):
        if i_label >= max_lines:
            warnings.warn(
                f"Only {max_lines} of {len(labels)} lines have been "
                "plotted. Increase max_lines to plot more lines."
            )
            break
        axes.plot(
            df.index.to_numpy(),
            df[label].to_numpy(),
            *args,
            label=label,
            **kwargs,
        )
    return axes


def _plot_units(ts: "TimeSeries") -> None:
    """Plot the units of the input TimeSeries on the current figure."""
    unit_set = set()
    for outer in ts.info:
        for inner in ts.info[outer]:
            if inner == "Unit" and outer != "Time":
                unit_set.add(ts.info[outer][inner])
    # Plot this list
    unit_str = ""
    for unit in unit_set:
        if len(unit_str) > 0:
            unit_str += ", "
        unit_str += unit

    plt.ylabel(unit_str)


def _plot_events(ts: "TimeSeries", event_names: bool) -> None:
    """Plot the events of the input TimeSeries on the current figure."""
    n_events = len(ts.events)
    event_times = []
    for event in ts.events:
        event_times.append(event.time)

    if len(ts.events) > 0:
        a = plt.axis()
        min_y = a[2]
        max_y = a[3]
        event_line_x = np.zeros(3 * n_events)
        event_line_y = np.zeros(3 * n_events)

        for i_event in range(0, n_events):
            event_line_x[3 * i_event] = event_times[i_event]
            event_line_x[3 * i_event + 1] = event_times[i_event]
            event_line_x[3 * i_event + 2] = np.nan

            event_line_y[3 * i_event] = min_y
            event_line_y[3 * i_event + 1] = max_y
            event_line_y[3 * i_event + 2] = np.nan

        plt.plot(event_line_x, event_line_y, ":k")

        if event_names:
            occurrences = {}  # type: dict[str, int]

            for event in ts.events:
                if event.name == "_":
                    name = "_"
                elif event.name in occurrences:
                    occurrences[event.name] += 1
                    name = f"{event.name} {occurrences[event.name]}"
                else:
                    occurrences[event.name] = 0
                    name = f"{event.name} 0"

                plt.text(
                    event.time,
                    max_y,
                    name,
                    rotation="vertical",
                    horizontalalignment="center",
                    fontsize="small",
                )


def plot(
    self,
    data_keys: str | list[str] | None = None,
    *args,
    event_names: bool = True,
    legend: bool = True,
    max_lines: int = 40,
    **kwargs,
) -> None:
    """
    Plot the TimeSeries in the current matplotlib figure.

    Parameters
    ----------
    data_keys
        The data keys to plot. If left empty, all data is plotted.
    event_names
        Optional. True to plot the event names on top of the event lines.
    legend
        Optional. True to plot a legend, False otherwise. Default is True.
    max_lines
        Optional. The maximal number of lines to plot. Default is 40. A
        warning is issued if plotting all the data would require more
        lines.

    Note
    ----
    Additional positional and keyboard arguments are passed to
    matplotlib's ``pyplot.plot`` function::

        ts.plot(["Forces"], "--")

    plots the forces using a dashed line style.

    Example
    -------
    For a TimeSeries ``ts`` with data keys being "Forces", "Moments" and
    "Angle"::

        ts.plot()

    plots all data (Forces, Moments and Angle), whereas::

        ts.plot(["Forces", "Moments"])

    plots only the forces and moments, without plotting the angle.

    """
    if data_keys is None:
        data_keys = []
    try:
        check_param("data_keys", data_keys, str)
    except TypeError:
        try:
            check_param("data_keys", data_keys, list, contents_type=str)
        except TypeError as e:
            raise TypeError(
                "data_keys must be a string or a list of strings."
            ) from e
    check_param("event_names", event_names, bool)
    check_param("legend", legend, bool)
    check_param("max_lines", max_lines, int)
    self._check_well_shaped()

    # Private argument _raise_on_no_data: Raise an EmptyDataSeriesError
    # instead of warning when no data is available to plot.
    if "_raise_on_no_data" in kwargs:
        raise_on_no_data = kwargs.pop("_raise_on_no_data")
    else:
        raise_on_no_data = False

    if len(data_keys) == 0:
        # Plot all
        ts = self.copy()
    else:
        ts = self.get_subset(data_keys)

    if raise_on_no_data:
        self._check_not_empty_time()
        self._check_not_empty_data()

    # Plot the curves
    axes = _plot_curves(ts, max_lines, args, kwargs)

    # Add labels
    plt.xlabel("Time (" + ts._get_time_unit() + ")")

    # Make unique list of units
    _plot_units(ts)

    # Plot the events
    _plot_events(ts, event_names)

    if legend and len(ts.data) > 0:
        if len(ts.data) < MAX_CURVES_TO_TRY_BEST_LEGEND_PLACEMENT:
            legend_location = "best"
        else:
            legend_location = "upper right"

        axes.legend(
            loc=legend_location, ncol=1 + int(len(ts.data) / max_lines)
        )  # Max MAX_CURVES_TO_TRY_BEST_LEGEND_PLACEMENT
