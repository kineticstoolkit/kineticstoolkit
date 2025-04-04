#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
__copyright__ = "Copyright (C) 2020-2025 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation
from kineticstoolkit import TimeSeries
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


def ui_edit_events(ts: TimeSeries, in_place: bool = False) -> TimeSeries:
    """
    Use an interactive interface to edit time and events in a Video TimeSeries.

    Parameters
    ----------
    ts
        TimeSeries, with at least one data key named "Video".
    in_place
        Optional. True to modify the original TimeSeries, False to return a
        copy.

    Returns
    -------
    TimeSeries
        The TimeSeries with the modified time and events.

    """
    if not in_place:
        ts = ts.copy()

    video = Video(ts)
    while video._closed is False:
        plt.pause(0.1)
    return ts


class Video:
    """
    WIP - Launch an interactive video player.

    For now, the TimeSeries must contain only one video labelled with the
    "Video" data key.

    """

    def __init__(self, video_ts: TimeSeries):
        # Extract information from the video
        data_key = "Video"

        self._closed = False

        self._video_ts = video_ts
        self._video_length = video_ts.data[data_key].shape[0]
        self._video_height = video_ts.data[data_key].shape[1]
        self._video_width = video_ts.data[data_key].shape[2]

        # Ensure we work with bytes, in case someone did operations on it and
        # now we have floats.
        self._video_ts.data[data_key] = np.clip(
            self._video_ts.data[data_key], 0, 255
        ).astype(np.uint8)

        # Remove default keymaps for left and right
        if "left" in plt.rcParams["keymap.back"]:
            plt.rcParams["keymap.back"].remove("left")
        if "right" in plt.rcParams["keymap.forward"]:
            plt.rcParams["keymap.forward"].remove("right")

        # Create the figure
        self._figure, axes = plt.subplots(
            4, 1, gridspec_kw={"height_ratios": [10, 0.25, 1, 0.5]}
        )

        # Remove all the second subplot, it's only a white space
        axes[1].set_frame_on(False)
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        # Remove all the last subplot, it's only for the text
        axes[3].set_frame_on(False)
        axes[3].set_xticks([])
        axes[3].set_yticks([])
        axes[3].text(
            0,
            0,
            "(←/→/Scroll) Navigate          "
            "(Shift←/Shift→) Navigate to event          "
            "(a/x) Add/Remove event          "
            "(z) Zero time          "
            "(q) Quit",
        )

        # Create the video part
        self._axes = axes[0]
        self._axes.set_xticks([])
        self._axes.set_yticks([])
        self._axes_image = self._axes.imshow(
            video_ts.data[data_key][0], interpolation="none", resample=False
        )

        # Create the data part
        self._axes_event_bar = axes[2]

        data_ts = TimeSeries(time=video_ts.time, events=video_ts.events)

        self._data_ts = data_ts

        self._create_refresh_event_bar()

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

        # Connect callbacks to the figure
        self._figure.canvas.mpl_connect("key_press_event", self._on_key)
        # self._figure.canvas.mpl_connect("key_release_event", self._on_release)
        self._figure.canvas.mpl_connect("scroll_event", self._on_scroll)
        self._figure.canvas.mpl_connect("pick_event", self._on_pick)
        self._figure.canvas.mpl_connect("close_event", self._on_close)
        self._figure.canvas.mpl_connect("resize_event", self._on_resize)

        # Public properties
        self.current_index = 0

        # Last things
        self._refresh()
        self._figure.tight_layout()

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
        return self._video_ts.time[self._current_index]

    @current_time.setter
    def current_time(self, value: float):
        """Set current_time value."""
        index = int(np.argmin(np.abs(self._video_ts.time - value)))
        self.current_index = index

    def _on_key(self, event):
        if event.key == "right":
            self.current_index += 1

        elif event.key == "left":
            self.current_index -= 1

        elif event.key == "shift+right":
            for i, one_event in enumerate(self._video_ts.events):
                if one_event.time > self.current_time:
                    self.current_time = one_event.time
                    break
            self._create_refresh_event_bar()
            self._refresh()

        elif event.key == "shift+left":
            for i, one_event in enumerate(self._video_ts.events[-1::-1]):
                if one_event.time < self.current_time:
                    self.current_time = one_event.time
                    break
            self._create_refresh_event_bar()
            self._refresh()

        elif event.key == " ":
            if self._running:
                self._running = False
                self._anim.event_source.stop()
            else:
                self._state["SystemTimeOnLastUpdate"] = time.time()
                self._running = True
                self._anim.event_source.start()

        elif event.key == "a":
            self._data_ts.add_event(
                self.current_time, "event", in_place=True, unique=True
            )
            self._video_ts.add_event(
                self.current_time, "event", in_place=True, unique=True
            )
            self._create_refresh_event_bar()
            self._refresh()

        elif event.key == "x":
            for i, one_event in enumerate(self._video_ts.events):
                if np.isclose(one_event.time, self.current_time):
                    self._data_ts.events.pop(i)
                    self._video_ts.events.pop(i)
                    break
            self._create_refresh_event_bar()
            self._refresh()

        elif event.key == "z":
            shift = -self.current_time
            self._data_ts.shift(shift, in_place=True)
            self._video_ts.shift(shift, in_place=True)
            self._create_refresh_event_bar()
            self._refresh()

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
            self._video_ts.data["Video"][self._current_index]
        )
        self._axes.set_title(
            f"{self._current_index} ({self.current_time:.2f} s.)"
        )
        self._axes_index_cursor.set_data(
            [self._video_ts.time[self._current_index]], [0]
        )
        self._figure.canvas.draw()

    def _create_refresh_event_bar(self):
        plt.sca(self._axes_event_bar)
        plt.cla()
        self._data_ts.plot()
        plt.axis([self._video_ts.time[0], self._video_ts.time[-1], -0.1, 1])
        plt.plot(
            self._video_ts.time,
            0 * self._video_ts.time,
            ".",
            color=(0.8, 0.8, 0.8),
            picker=True,
            pickradius=2,
        )
        self._axes_event_bar.set_yticks([])
        plt.box(False)

        self._axes_index_cursor = plt.plot(0, 0, "dr")[0]

    def _on_close(self, event):
        self._closed = True

    def _on_resize(self, event):
        self._figure.tight_layout()
