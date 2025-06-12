#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2025 Félix Chénier

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Provides objects to find events from videos.

Warning
-------
This is currently in development, the whole interface could change at any time.

This module requires OpenCV which is not a dependency (yet) of Kinetics
Toolkit.

Installing OpenCV
-----------------
- Using conda: `conda install -c conda-forge opencv`
- Using pip: `pip install opencv-python`

"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2025 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation, widgets
from kineticstoolkit import TimeSeries
from kineticstoolkit.exceptions import warn_once
import time


WINDOW_PLACEMENT = {"top": 50, "right": 0}


def read_video(
    filename: str,
    width: int | None = None,
    height: int | None = None,
    downsample: int = 1,
    max_memory: int = 1000,
) -> TimeSeries:
    """
    Read a video file as a TimeSeries.

    Warning
    -------
    This feature is experimental and will probably change in the future.

    Reads a video file using opencv, and returns a TimeSeries with one data
    key ("Video") that contains the video as an NxHxWx3 array where N is the
    number of images in the video, H is the video height, W is the video
    width, and 3 are the RGB channels.

    Parameters
    ----------
    filename
        The name of the video file
    width
        Optional. Scale down the video to a given width in pixels.
        Aspect ratio is maintained unless both width and height are set.
    height
        Optional. Scale down the video to a given height in pixels.
        Aspect ratio is maintained unless both width and height are set.
    downsample
        Optional. Downsample the video by a given integer. For instance, a
        240 FPS video with a downsample parameter of 2 will results in a
        120 FPS TimeSeries. Default is 1.
    max_memory
        Prevent the output TimeSeries from exceeding a maximal RAM size, in
        Mb. Default is 1000.

    Returns
    -------
    TimeSeries

    """
    warn_once(
        "Function read_video is experimental and will probably change in the "
        "future."
    )
    video = cv2.VideoCapture(filename)

    video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = int(video.get(cv2.CAP_PROP_FPS))

    new_video_length = video_length // downsample
    new_video_fps = video_fps // downsample

    if (width is not None) and (height is None) and (width < video_width):
        new_video_width = int(width)
        new_video_height = int(video_height * (width / video_width))
    elif (width is None) and (height is not None) and (height < video_height):
        new_video_height = int(height)
        new_video_width = int(video_width * (height / video_height))
    elif (
        (width is not None)
        and (height is not None)
        and ((height < video_height) | (width < video_width))
    ):
        new_video_height = int(height)
        new_video_width = int(width)
    else:
        new_video_width = int(video_width)
        new_video_height = int(video_height)

    estimated_memory_size = int(
        new_video_length * new_video_width * new_video_height * 3 / 1e6
    )
    if estimated_memory_size > max_memory:
        raise ValueError(
            "The resulting TimeSeries' RAM usage would be about "
            f"{estimated_memory_size} Mb, which exceed "
            f"the maximal RAM usage of {max_memory} Mb "
            "defined by the max_memory parameter. You can either increase "
            "this value or reduce the memory required by the video by "
            "resizing or downsampling it using the according "
            "read_video function parameters."
        )

    frames = np.empty(
        (new_video_length, new_video_height, new_video_width, 3),
        dtype=np.uint8,
    )

    for i in tqdm(range(video_length)):
        read, frame = video.read()

        if not read:
            break

        if i % downsample != 0:
            continue

        if new_video_width != video_width:
            frame = cv2.resize(
                frame,
                (new_video_width, new_video_height),
                interpolation=cv2.INTER_LINEAR,
            )
        frames[i // downsample] = frame[:, :, -1::-1]  # BGR to RGB

    return TimeSeries(
        time=np.arange(new_video_length) / new_video_fps,
        data={"Video": frames},
    )


def ui_edit_events(
    ts_video: TimeSeries,
    ts_data: TimeSeries,
    video_key: str = "Video",
    data_keys: str | list[str] = [],
    event_source: str = "merge",
    in_place: bool = False,
) -> list[TimeSeries]:
    """
    Use an interactive interface to edit time and events in a Video TimeSeries.

    Warning
    -------
    This feature is experimental and will probably change in the future.

    Parameters
    ----------
    ts_video
        The TimeSeries that contains the video, normally obtained using
        the read_video() function.
    ts_data
        Optional. TimeSeries.
    video_key
        Optional. The data key in ts_video that contains the video. Default is
        "Video".
    data_keys
        Optional. The data key(s) in ts_video to plot. Default is [], which
        means that every data of the TimeSeries is plotted.
    event_source
        Optional. Defines which TimeSeries to read the event from. Can be::
            - "merge" (default): The editor includes the events from both
              video_ts and data_ts
            - "video": The editor includes only the events from video_ts
            - "data_ts": The editor includes only the events from data_ts
    in_place
        Optional. True to modify the original TimeSeries, False to return a
        copy.

    Returns
    -------
    list[TimeSeries]
        The list [video_ts, data_ts] which are now synchronized and which
        share the events.

    """
    warn_once(
        "Function ui_edit_events is experimental and will probably change in "
        "the future."
    )

    if not in_place:
        ts_video = ts_video.copy()
        ts_data = ts_data.copy()

    video = Video(
        ts_video=ts_video,
        ts_data=ts_data,
        video_key=video_key,
        data_keys=data_keys,
        event_source=event_source,
    )

    while video._closed is False:
        plt.pause(0.1)

    return [ts_video, ts_data]


class Video:
    """
    Launch an interactive video player to edit events (Work in progress).

    Warning
    -------
    This feature is experimental and will probably change in the future.

    Attributes
    ----------
    ts_video : TimeSeries
        The TimeSeries that contains the video, normally obtained using
        the read_video() function.
    ts_data : TimeSeries
        Optional. TimeSeries.
    video_key : str
        Optional. The data key in ts_video that contains the video. Default is
        "Video".
    data_keys : str | list[str]
        Optional. The data key(s) in ts_video to plot. Default is [], which
        means that every data of the TimeSeries is plotted.
    event_source: str
        Optional. Defines which TimeSeries to read the event from. Can be::
            - "merge" (default): The editor includes the events from both
              video_ts and data_ts
            - "video": The editor includes only the events from video_ts
            - "data_ts": The editor includes only the events from data_ts

    """

    def __init__(
        self,
        ts_video: TimeSeries,
        ts_data: TimeSeries | None,
        *,
        video_key: str = "Video",
        data_keys: str | list[str] = [],
        event_source: str = "merge",
    ):

        self._closed = False

        self._video_key = video_key
        self._data_keys = data_keys

        # Create the Video TimeSeries
        self._ts_video = ts_video

        # Create the Data TimeSeries
        if ts_data is None:
            self._ts_data = TimeSeries(
                time=ts_video.time, events=ts_video.events
            )
        else:
            self._ts_data = ts_data

        # Select which events to keep or merge
        if event_source == "merge":
            for event in self._ts_video.events:
                self._ts_data.add_event(
                    event.time, event.name, unique=True, in_place=True
                )
            for event in self._ts_data.events:
                self._ts_video.add_event(
                    event.time, event.name, unique=True, in_place=True
                )
        elif event_source == "data":
            self._ts_video.events = []
            for event in self._ts_data.events:
                self._ts_video.add_event(event.time, event.name, in_place=True)
        elif event_source == "video":
            self._ts_data.events = []
            for event in self._ts_video.events:
                self._ts_data.add_event(event.time, event.name, in_place=True)
        else:
            raise ValueError(
                "event_source must be 'merge', 'data' or 'video'."
            )

        # Extract information from the video
        self._video_length = self._ts_video.data[video_key].shape[0]
        self._video_height = self._ts_video.data[video_key].shape[1]
        self._video_width = self._ts_video.data[video_key].shape[2]

        # Ensure we work with bytes, in case someone did operations on it and
        # now we have floats.
        self._ts_video.data[video_key] = np.clip(
            self._ts_video.data[video_key], 0, 255
        ).astype(np.uint8)

        self._create_gui()

        # Public properties
        self.current_index = 0

        # Last things
        self._refresh()

    # Properties
    @property
    def current_index(self) -> int:
        """Read/write current_index."""
        return self._current_index

    @current_index.setter
    def current_index(self, value: int):
        """Set current_index value."""
        self._current_index = value % self._video_length
        self._refresh()

    @property
    def current_time(self) -> float:
        """Read/write current_time."""
        return self._ts_video.time[self._current_index]

    @current_time.setter
    def current_time(self, value: float):
        """Set current_time value."""
        index = int(np.argmin(np.abs(self._ts_video.time - value)))
        self.current_index = index

    def _create_gui(self):
        """Create all the GUI elements."""
        # Remove default keymaps for left and right
        if "left" in plt.rcParams["keymap.back"]:
            plt.rcParams["keymap.back"].remove("left")
        if "right" in plt.rcParams["keymap.forward"]:
            plt.rcParams["keymap.forward"].remove("right")

        # Create the figure
        # fmt: off
        self._figure, self._axes = plt.subplot_mosaic(
            [
                ["LblNextIndex",  "LblNextIndex", "Video",    "LblEvent",     "LblEvent"],
                ["LblNextIndex",  "LblNextIndex", "Video",    "LblEvent",     "LblEvent"],
                ["BtnPrevIndex",  "BtnNextIndex", "Video",    "TxtEventName", "TxtEventName"],
                ["LblNextEvent",  "LblNextEvent", "Video",    "BtnAddEvent",  "BtnRemoveEvent"],
                ["LblNextEvent",  "LblNextEvent", "Video",    "RSpace1",      "RSpace1"],
                ["BtnPrevEvent",  "BtnNextEvent", "Video",    "RSpace1",      "RSpace1"],
                ["LSpace",        "LSpace",       "Video",    "LblZero",      "LblZero"],
                ["LSpace",        "LSpace",       "Video",    "LblZero",      "LblZero"],
                ["LSpace",        "LSpace",       "Video",    "BtnZeroVideo", "BtnZeroVideo"],
                ["LSpace",        "LSpace",       "Video",    "BtnZeroData",  "BtnZeroData"],
                ["BtnPlay",       "BtnPlay",      "Video",    "RSpace2",      "RSpace2"],
                ["Space1",        "Space1",       "Space1",   "Space1",       "Space1"],
                ["Data",          "Data",         "Data",     "Data",         "Data"],
            ],
            width_ratios=[1, 1, 20, 1, 1],
            height_ratios=[1,1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1.5, 6],
        )
        # fmt: on

        # Remove frames and ticks for all for now
        for key in self._axes:
            self._axes[key].set_frame_on(False)
            self._axes[key].set_xticks([])
            self._axes[key].set_yticks([])

        # Create the video part
        self._axes["Video"].set_frame_on(True)
        self._axes_image = self._axes["Video"].imshow(
            self._ts_video.data[self._video_key][0],
            interpolation="none",
            resample=False,
        )

        # Create the data part
        self._axes_data = self._axes["Data"]
        self._create_refresh_data()

        # Create a state like in the Player
        self._state = {"SystemTimeOnLastUpdate": time.time()}
        self._running = False

        # Create the anim timer
        self._anim = animation.FuncAnimation(
            self._figure,
            self._on_timer,  # type: ignore
            interval=33,
            cache_frame_data=False,
        )  # 30 ips

        # Create the GUI elements

        self._axes["LblNextIndex"].text(
            0.5,
            0,
            "Next Index\n(←/→/Scroll)",
            horizontalalignment="center",
            verticalalignment="bottom",
        )
        self._axes["BtnPrevIndex"].set_frame_on(True)
        self._btn_prev_index = widgets.Button(self._axes["BtnPrevIndex"], "◀︎")
        self._btn_prev_index.on_clicked(self._on_previous_index)
        self._axes["BtnNextIndex"].set_frame_on(True)
        self._btn_next_index = widgets.Button(self._axes["BtnNextIndex"], "▶︎")
        self._btn_next_index.on_clicked(self._on_next_index)

        self._axes["LblNextEvent"].text(
            0.5,
            0,
            "Next Event\n(Shift + ←/→)",
            horizontalalignment="center",
            verticalalignment="bottom",
        )
        self._axes["BtnPrevEvent"].set_frame_on(True)
        self._btn_prev_event = widgets.Button(self._axes["BtnPrevEvent"], "◀︎◀︎")
        self._btn_prev_event.on_clicked(self._on_previous_event)
        self._axes["BtnNextEvent"].set_frame_on(True)
        self._btn_next_event = widgets.Button(self._axes["BtnNextEvent"], "▶︎▶︎")
        self._btn_next_event.on_clicked(self._on_next_event)

        self._axes["LblZero"].text(
            0.5,
            0,
            "Sync (zero)",
            horizontalalignment="center",
            verticalalignment="bottom",
        )
        self._axes["BtnZeroVideo"].set_frame_on(True)
        self._btn_zero_video = widgets.Button(
            self._axes["BtnZeroVideo"], "Video (z)"
        )
        self._btn_zero_video.on_clicked(self._on_zero_video)
        self._axes["BtnZeroData"].set_frame_on(True)
        self._btn_zero_data = widgets.Button(
            self._axes["BtnZeroData"], "Data (Shift + z)"
        )
        self._btn_zero_data.on_clicked(self._on_zero_data)

        self._axes["LblEvent"].text(
            0.5,
            0,
            "Add/Remove",
            horizontalalignment="center",
            verticalalignment="bottom",
        )
        self._axes["TxtEventName"].set_frame_on(True)
        self._txt_event_name = widgets.TextBox(
            self._axes["TxtEventName"], label="", initial="event", color="w"
        )
        self._axes["BtnAddEvent"].set_frame_on(True)
        self._btn_add_event = widgets.Button(self._axes["BtnAddEvent"], "+")
        self._btn_add_event.on_clicked(self._on_add_event)
        self._axes["BtnRemoveEvent"].set_frame_on(True)
        self._btn_remove_event = widgets.Button(
            self._axes["BtnRemoveEvent"], "−"
        )
        self._btn_remove_event.on_clicked(self._on_remove_event)

        # Connect callbacks to the figure
        self._figure.canvas.mpl_connect("key_press_event", self._on_key)
        # self._figure.canvas.mpl_connect("key_release_event", self._on_release)
        self._figure.canvas.mpl_connect("scroll_event", self._on_scroll)
        self._figure.canvas.mpl_connect("pick_event", self._on_pick)
        self._figure.canvas.mpl_connect("close_event", self._on_close)

        # Give a decent size to the figure
        self._figure.set_size_inches(15, 8)

    # %% Callbacks

    def _on_next_index(self, event):
        self.current_index += 1

    def _on_previous_index(self, event):
        self.current_index -= 1

    def _on_next_event(self, event):
        for i, one_event in enumerate(self._ts_video.events):
            if one_event.time > self.current_time:
                self.current_time = one_event.time
                break
        self._create_refresh_data()
        self._refresh()

    def _on_previous_event(self, event):
        for i, one_event in enumerate(self._ts_video.events[-1::-1]):
            if one_event.time < self.current_time:
                self.current_time = one_event.time
                break
        self._create_refresh_data()
        self._refresh()

    def _on_zero_video(self, event):
        shift = -self.current_time
        self._ts_video.shift(shift, in_place=True)
        self._create_refresh_data()
        self.current_time = 0
        self._refresh()

    def _on_zero_data(self, event):
        shift = -self.current_time
        self._ts_data.shift(shift, in_place=True)
        self._create_refresh_data()
        self.current_time = 0
        self._refresh()

    def _on_add_event(self, event):
        self._ts_data.add_event(
            self.current_time,
            self._txt_event_name.text,
            in_place=True,
            unique=True,
        )
        self._ts_video.add_event(
            self.current_time,
            self._txt_event_name.text,
            in_place=True,
            unique=True,
        )
        self._create_refresh_data()
        self._refresh()

    def _on_remove_event(self, event):
        for i, one_event in enumerate(self._ts_video.events):
            if (
                np.isclose(one_event.time, self.current_time)
                and one_event.name == self._txt_event_name.text
            ):
                self._ts_data.events.pop(i)
                self._ts_video.events.pop(i)
                break
        self._create_refresh_data()
        self._refresh()

    def _on_key(self, event):
        if event.key == "right":
            self._on_next_index(event)

        elif event.key == "left":
            self._on_previous_index(event)

        elif event.key == "shift+right":
            self._on_next_event(event)

        elif event.key == "shift+left":
            self._on_previous_event(event)

        elif event.key == " ":
            if self._running:
                self._running = False
                self._anim.event_source.stop()
            else:
                self._state["SystemTimeOnLastUpdate"] = time.time()
                self._running = True
                self._anim.event_source.start()

        elif event.key == "+":
            self._on_add_event(event)

        elif event.key == "-":
            self._on_remove_event(event)

        elif event.key == "z":
            self._on_zero_video(event)

        elif event.key == "Z":
            self._on_zero_data(event)

    def _on_pick(self, event):
        if event.mouseevent.button == 1:
            index = event.ind[0]
            self.current_index = index

    def _on_scroll(self, event):  # pragma: no cover
        if event.button == "up":
            self.current_index -= 1
        elif event.button == "down":
            self.current_index += 1

    def _on_timer(self, _) -> None:  # pragma: no cover
        """Implement callback for the animation timer object."""
        if self._running is True:
            # We check self._running because we can enter this callback
            # even if the animation has been deactivated. This is because the
            # recommended way to deactivate a timer is to unreference it,
            # however the garbage collector may take time deleting the timer
            # and we will end up with still a few timer callbacks. Checking
            # self._running makes sure that we effectively stop.
            current_system_time = time.time()
            current_index = self.current_index
            self.current_time += 1.0 * (
                time.time() - self._state["SystemTimeOnLastUpdate"]  # type: ignore
            )
            # Type ignored because mypy considers
            # self._state["SystemTimeOnLastUpdate"] as an "object"
            if current_index == self.current_index:
                # The time wasn't enough to advance a frame. Articifically
                # advance a frame.
                self.current_index += 1
            self._state["SystemTimeOnLastUpdate"] = current_system_time
        else:
            self._anim.event_source.stop()

    def _refresh(self):
        self._axes_image.set_data(
            self._ts_video.data["Video"][self._current_index]
        )
        #        self._axes.set_title(
        #            f"{self._current_index} ({self.current_time:.2f} s.)"
        #        )

        # Update line
        cursor_data = self._axes_index_line.get_data()
        self._axes_index_line.set_data(
            [
                self._ts_video.time[self._current_index],
                self._ts_video.time[self._current_index],
            ],
            [cursor_data[1][0], cursor_data[1][1]],
        )

        # Update cursor
        cursor_data = self._axes_index_cursor.get_data()
        self._axes_index_cursor.set_data(
            [
                self._ts_video.time[self._current_index],
            ],
            [cursor_data[1][0]],
        )

        self._figure.canvas.draw_idle()

    def _create_refresh_data(self):
        plt.sca(self._axes_data)
        plt.cla()

        # Plot data
        self._ts_data.plot(self._data_keys, legend=False)
        plt.xlabel("")
        plt.ylabel("")

        # Plot red line
        current_axes = plt.axis()
        self._axes_index_line = plt.plot(
            [0, 0], [current_axes[2], current_axes[3]], "r"
        )[0]

        # Plot clickable timeline
        plt.plot(
            self._ts_video.time,
            (0 * self._ts_video.time)
            + current_axes[2]
            - 0.1 * (current_axes[3] - current_axes[2]),
            ".",
            color=(0.8, 0.8, 0.8),
            picker=True,
            pickradius=2,
        )

        # Plot red cursor
        self._axes_index_cursor = plt.plot(
            [0],
            [
                current_axes[2] - 0.1 * (current_axes[3] - current_axes[2]),
            ],
            "dr",
            markersize=8,
        )[0]

        plt.axis(
            [
                min(self._ts_video.time[0], self._ts_data.time[0]),
                max(self._ts_video.time[-1], self._ts_data.time[-1]),
                current_axes[2] - 0.15 * (current_axes[3] - current_axes[2]),
                current_axes[3],
            ]
        )

    def _on_close(self, event):
        self._closed = True
