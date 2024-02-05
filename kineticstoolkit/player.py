#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020 Félix Chénier

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
Provides the Player class to visualize points and frames in 3d.

The Player class is accessible directly from the toplevel Kinetics Toolkit
namespace (i.e., ktk.Player).
"""
from __future__ import annotations

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


from kineticstoolkit.timeseries import TimeSeries
from kineticstoolkit.decorators import deprecated
from kineticstoolkit.tools import check_interactive_backend
import kineticstoolkit._repr
import kineticstoolkit.geometry as geometry

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import animation
import numpy as np
from numpy import sin, cos
import time
from copy import deepcopy
from typing import Any
from kineticstoolkit.typing_ import typecheck, ArrayLike
import warnings

# To fit the new viewpoint on selecting a new point
import scipy.optimize as optim

REPR_HTML_MAX_DURATION = 10  # Max duration for _repr_html
COLORS = ["r", "g", "b", "y", "c", "m", "w"]

HELP_TEXT = """
    ktk.Player help
    ----------------------------------------------------
    KEYBOARD COMMANDS
    show/hide this help : h
    previous index      : left
    next index          : right
    previous second     : shift+left
    next second         : shift+right
    play/pause          : space
    2x playback speed   : +
    0.5x playback speed : -
    toggle track        : t
    toggle perspective  : d (depth)
    ----------------------------------------------------
    MOUSE COMMANDS
    select a point      : left-click
    3d rotate           : left-drag
    pan                 : middle-drag or shift+left-drag
    zoom                : right-drag or wheel
"""


@typecheck
class Player:
    # FIXME! Update this docstring.
    """
    A class that allows visualizing points and frames in 3D.

    `player = ktk.Player(parameters)` creates and launches an interactive
    Player instance. Once the window is open, press `h` to show a help
    overlay.

    Parameters
    ----------
    *ts
        Contains the points and frames to visualize, where each data
        key is either a point position expressed as Nx4 array, or a frame
        expressed as a Nx4x4 array. Multiple TimeSeries can be provided.

    interconnections
        Optional. Each key corresponds to an inerconnection between points,
        where one interconnection is another dict with the following keys:

        - "Links": list of lists strings, where each string is a point
          name. For example, to create a link that spans Point1 and Point2,
          and another link that spans Point3, Point4 and Point5,
          interconnections["Links"] would be::

              [["Point1", "Point2"], ["Point3", "Point4", "Point5"]]

        - "Color": character or tuple (RGB) that represents the color of the
          link. Color must be a valid value for matplotlib's
          plots.

    current_index
        Optional. Sets the inital index number to show.

    point_size
        Optional. Sets the point radius as defined by matplotlib.

    frame_size
        Optional. Sets the frame size in meters.

    up
        Optional. Defines the ground plane by setting which axis is up. May be
        {"x", "y", "z", "-x", "-y", "-z"}. Default is "y".

    zoom
        Optional. Sets the initial camera zoom.

    azimuth
        Optional. Sets the initial camera azimuth in radians.

    elevation
        Optional. Sets the initial camera elevation in radians.

    translation
        Optional. Sets the initial camera translation (panning).

    target
        Optional. Sets the camera target in meters.

    track
        Optional. False to keep the scene static, True to track the last
        selected point when changing index.

    perspective
        Optional. True to draw the scene using perspective, False to draw the
        scene orthogonally.

    Note
    ----
    Matplotlib must be in interactive mode.

    """

    # %% Init and properties getters and setters

    def __init__(
        self,
        *ts: TimeSeries,
        interconnections: dict[str, dict[str, Any]] = {},
        current_index: int = 0,
        current_time: float | None = None,
        playback_speed: float = 1.0,
        up: str = "y",
        zoom: float = 1.0,
        azimuth: float = 0.0,
        elevation: float = 0.2,
        translation: ArrayLike = (0.0, 0.0),
        target: ArrayLike = (0.0, 0.0, 0.0),
        perspective: bool = True,
        track: bool = False,
        point_size: float = 4.0,
        interconnection_width: float = 1.5,
        frame_size: float = 0.1,
        frame_width: float = 3.0,
        grid_size: float = 10.0,
        grid_width: float = 1.0,
        grid_subdivision_size: float = 1.0,
        grid_origin: ArrayLike = (0.0, 0.0, 0.0, 1.0),
        grid_color: ArrayLike = (0.3, 0.3, 0.3),
        background_color: ArrayLike = (0.0, 0.0, 0.0),
        **kwargs,  # Can be "inline_player=True", or older parameter names
    ):
        # Allow older parameter names
        if "segments" in kwargs and interconnections == {}:
            interconnections = kwargs["segments"]
        if "segment_width" in kwargs:
            interconnection_width = kwargs["segment_width"]
        if "current_frame" in kwargs:
            current_index = kwargs["current_frame"]

        # Warn if Matplotlib is not interactive
        check_interactive_backend()

        # Assign properties

        # Empty content for now. We add the final content after all
        # initializations.
        self._contents = TimeSeries(time=[0])
        self._grid = np.array([])
        self._oriented_points = TimeSeries(time=self._contents.time)
        self._oriented_frames = TimeSeries(time=self._contents.time)

        self._interconnections = {}
        self._extended_interconnections = {}

        self._current_index = current_index

        # FIXME! transform to property correctly
        if current_time is not None:
            self.current_time = current_time

        self._playback_speed = playback_speed
        self._up = up
        self._zoom = zoom
        self._azimuth = azimuth
        self._elevation = elevation
        self._translation = translation
        self._target = target
        self._perspective = perspective
        self._track = track
        self._point_size = point_size
        self._interconnection_width = interconnection_width
        self._frame_size = frame_size
        self._frame_width = frame_width
        self._grid_size = grid_size
        self._grid_width = grid_width
        self._grid_subdivision_size = grid_subdivision_size
        self._grid_origin = grid_origin
        self._grid_color = grid_color
        self._background_color = background_color

        self._select_none()
        self.last_selected_point = ""
        self._running = False

        # Init mouse navigation state
        self._state = {
            "ShiftPressed": False,
            "MouseLeftPressed": False,
            "MouseMiddlePressed": False,
            "MouseRightPressed": False,
            "MousePositionOnPress": (0.0, 0.0),
            "MousePositionOnMiddlePress": (0.0, 0.0),
            "MousePositionOnRightPress": (0.0, 0.0),
            "TranslationOnMousePress": (0.0, 0.0),
            "AzimutOnMousePress": 0.0,
            "ElevationOnMousePress": 0.0,
            "SystemTimeOnLastUpdate": time.time(),
        }

        # Create the figure and prepare its contents
        (fig, axes, anim) = self._create_empty_figure()
        self._mpl_objects = {
            "Figure": fig,
            "Axes": axes,
            "Anim": anim,
        }

        # Add the true contents using the public interface so that everything
        # is refreshed automatically
        temp_ts = TimeSeries()
        for one_ts in ts:
            temp_ts.merge(one_ts, in_place=True)

        self.set_contents(temp_ts)
        self.set_interconnections(interconnections)
        self.grid_origin = grid_origin  # Refresh grid

    @property
    def current_index(self) -> int:
        """Get current_index value."""
        return self._current_index

    @current_index.setter
    def current_index(self, value: int):
        """Set current_index value."""
        if value >= len(self._contents.time):
            self._current_index = len(self._contents.time) - 1
            warnings.warn(
                "Index must be lower than the number of samples "
                f"({len(self._contents.time)}), however a value of {value} "
                "has been received. The current index has been set to "
                f"{self._current_index}."
            )
        elif value < 0:
            self._current_index = 0
            warnings.warn(
                f"Index must be higher than 0, however a value of {value} has "
                "been received. The current index has been set to 0."
            )
        else:
            self._current_index = value
        self._fast_refresh()

    # Properties
    @property
    def playback_speed(self) -> float:
        """Get playback_speed value."""
        return self._playback_speed

    @playback_speed.setter
    def playback_speed(self, value: float):
        """Set playback_speed value."""
        self._playback_speed = value

    @property
    def up(self) -> str:
        """Get up value."""
        return self._up

    @up.setter
    def up(self, value: str):
        """Set up value."""
        if value in {"x", "y", "z", "-x", "-y", "-z"}:
            self._up = value
        else:
            raise ValueError(
                'up must be either "x", "y", "z", "-x", "-y", or "-z"}'
            )
        self._orient_contents()
        self.refresh()

    @property
    def zoom(self) -> float:
        """Get zoom value."""
        return self._zoom

    @zoom.setter
    def zoom(self, value: float):
        """Set zoom value."""
        self._zoom = value
        self._fast_refresh()

    @property
    def azimuth(self) -> float:
        """Get azimuth value."""
        return self._azimuth

    @azimuth.setter
    def azimuth(self, value: float):
        """Set azimuth value."""
        self._azimuth = value
        self._fast_refresh()

    @property
    def elevation(self) -> float:
        """Get elevation value."""
        return self._elevation

    @elevation.setter
    def elevation(self, value: float):
        """Set elevation value."""
        self._elevation = value
        self._fast_refresh()

    @property
    def translation(self) -> tuple[float, float]:
        """Get translation value as (x, y)."""
        return tuple(self._translation)

    @translation.setter
    def translation(self, value: ArrayLike):
        """Set translation value using (x, y) or (x, y, ...)."""
        # Store as ndarray[x, y]
        self._translation = np.array(value[0:2])
        self._fast_refresh()

    @property
    def target(self) -> tuple[float, float, float]:
        """Get target value as (x, y, z)."""
        return tuple(self._target)

    @target.setter
    def target(self, value: ArrayLike):
        """Set target value using (x, y, z) or (x, y, z, 1.0)."""
        self._target = np.array(value[0:3])
        self._fast_refresh()

    @property
    def perspective(self) -> bool:
        """Get perspective value."""
        return self._perspective

    @perspective.setter
    def perspective(self, value: bool):
        """Set perspective value."""
        self._perspective = value
        self._fast_refresh()

    @property
    def track(self) -> bool:
        """Get track value."""
        return self._track

    @track.setter
    def track(self, value: bool):
        """Set perspective value."""
        self._track = value
        self._fast_refresh()

    @property
    def point_size(self) -> float:
        """Get point_size value."""
        return self._point_size

    @point_size.setter
    def point_size(self, value: float):
        """Set point_size value."""
        self._point_size = value
        self.refresh()

    @property
    def interconnection_width(self) -> float:
        """Get interconnection_width value."""
        return self._interconnection_width

    @interconnection_width.setter
    def interconnection_width(self, value: float):
        """Set interconnection_width value."""
        self._interconnection_width = value
        self.refresh()

    @property
    def frame_size(self) -> float:
        """Get frame_size value."""
        return self._frame_size

    @frame_size.setter
    def frame_size(self, value: float):
        """Set frame_size value."""
        self._frame_size = value
        self._fast_refresh()

    @property
    def frame_width(self) -> float:
        """Get frame_width value."""
        return self._frame_width

    @frame_width.setter
    def frame_width(self, value: float):
        """Set frame_width value."""
        self._frame_width = value
        self.refresh()

    @property
    def grid_size(self) -> float:
        """Get grid_size value."""
        return self._grid_size

    @grid_size.setter
    def grid_size(self, value: float):
        """Set grid_size value."""
        self._grid_size = value
        self._update_grid()
        self.refresh()

    @property
    def grid_width(self) -> float:
        """Get grid_width value."""
        return self._grid_width

    @grid_width.setter
    def grid_width(self, value: float):
        """Set grid_width value."""
        self._grid_width = value
        self._update_grid()
        self.refresh()

    @property
    def grid_subdivision_size(self) -> float:
        """Get grid_subdivision_size value."""
        return self._grid_subdivision_size

    @grid_subdivision_size.setter
    def grid_subdivision_size(self, value: float):
        """Set grid_subdivision_size value."""
        self._grid_subdivision_size = value
        self._update_grid()
        self.refresh()

    @property
    def grid_origin(self) -> np.ndarray:
        """Get grid_origin value."""
        return self._grid_origin

    @grid_origin.setter
    def grid_origin(self, value: float):
        """Set grid_origin value."""
        self._grid_origin = np.array(value)
        self._update_grid()
        self.refresh()

    @property
    def grid_color(self) -> tuple[float, float, float]:
        """Get grid_color value."""
        return tuple(self._grid_color)

    @grid_color.setter
    def grid_color(self, value: ArrayLike):
        """Set grid_color value."""
        if len(value) != 3:
            raise ValueError("grid_color must be an (R, G, B) tuple.")
        self._grid_color = tuple(value)
        self._update_grid()
        self.refresh()

    @property
    def background_color(self) -> tuple[float, float, float]:
        """Get background_color value."""
        return tuple(self._background_color)

    @background_color.setter
    def background_color(self, value: ArrayLike):
        """Set background_color value."""
        if len(value) != 3:
            raise ValueError("background_color must be an (R, G, B) tuple.")
        self._background_color = tuple(value)
        self.refresh()


    def __dir__(self):
        """Return directory."""
        return ["close"]

    def __str__(self) -> str:
        """Print a textual description of the Player properties."""
        return (
            "ktk.Player with current properties:\n"
            "---\n"
            f"current_index : {self.current_index}\n"
            f"playback_speed : {self.playback_speed:.3f}\n"
            "---\n"
            f"up : '{self.up}'\n"
            f"zoom : {self.zoom:.3f}\n"
            f"azimuth : {self.azimuth:.3f}\n"
            f"elevation : {self.elevation:.3f}\n"
            f"translation : {self.translation}\n"
            f"target : {self.target}\n"
            f"perspective : {self.perspective}\n"
            "---\n"
            f"track : {self.track}\n"
        )

    def __repr__(self) -> str:
        """Generate the class representation."""
        return str(self)

    def _create_empty_figure(self) -> tuple:
        """Create figure and return Figure, Axes and AnimationTimer."""
        # Create the figure and axes
        (fig, ax) = plt.subplots(num=None, figsize=(12, 9))
        fig.set_facecolor("k")

        # Remove the toolbar
        try:  # Try, setVisible method not always there
            fig.canvas.toolbar.setVisible(False)
        except AttributeError:
            pass

        plt.tight_layout()

        # Add the animation timer
        anim = animation.FuncAnimation(
            fig,
            self._on_timer,  # type: ignore
            frames=len(self._contents.time),
            interval=33,
        )  # 30 ips

        # Connect the callback functions
        fig.canvas.mpl_connect("pick_event", self._on_pick)
        fig.canvas.mpl_connect("key_press_event", self._on_key)
        fig.canvas.mpl_connect("key_release_event", self._on_release)
        fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        fig.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        fig.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        fig.canvas.mpl_connect("motion_notify_event", self._on_mouse_motion)

        return (fig, ax, anim)

    # %% Contents getters/setters

    # We use proper setters and getters to be absolutely sure the contents
    # could not be modified without adapting the Player to this new contents.
    # (e.g., rebuild the interconnection plots)

    def get_contents(self) -> TimeSeries:
        """Get contents value."""
        return self._contents

    def set_contents(self, value: TimeSeries) -> None:
        """Set contents value."""
        # Ensure that there is at least one sample so that the Player does not
        # crash and shows nothing instead.
        if len(value.time) > 0:
            self._contents = value.copy()
        else:
            warnings.warn("The provided TimeSeries is empty.")
            self._contents = TimeSeries(time=[0])

        self._orient_contents()
        self._extend_interconnections()
        self.refresh()

    def get_interconnections(self) -> dict[str, dict[str, Any]]:
        """Get interconnections value."""
        return self._interconnections

    def set_interconnections(self, value: dict[str, dict[str, Any]]) -> None:
        """Set interconnections value."""
        self._interconnections = deepcopy(value)
        self._extend_interconnections()
        self.refresh()

    def _extend_interconnections(self) -> None:
        """Update self._extended_interconnections. Does not refresh."""
        # Make a set of all patterns matched by the * in interconnection
        # point names.
        patterns = set()
        keys = list(self._contents.data.keys())
        for body_name in self._interconnections:
            for i_link, link in enumerate(
                self._interconnections[body_name]["Links"]
            ):
                for i_point, point in enumerate(link):
                    if point.startswith("*"):
                        for key in keys:
                            if key.endswith(point[1:]):
                                patterns.add(
                                    key[: (len(key) - len(point) + 1)]
                                )
                    elif point.endswith("*"):
                        for key in keys:
                            if key.startswith(point[:-1]):
                                patterns.add(
                                    key[(len(key) - len(point) + 1) :]
                                )

        # Extend every * to every pattern
        self._extended_interconnections = dict()
        for pattern in patterns:
            for body_name in self._interconnections:
                body_key = f"{pattern}{body_name}"
                self._extended_interconnections[body_key] = dict()
                self._extended_interconnections[body_key][
                    "Color"
                ] = self._interconnections[body_name]["Color"]

                # Go through every link of this segment
                self._extended_interconnections[body_key]["Links"] = []
                for i_link, link in enumerate(
                    self._interconnections[body_name]
                ):
                    self._extended_interconnections[body_key]["Links"].append(
                        [
                            s.replace("*", pattern)
                            for s in self._interconnections[body_name][
                                "Links"
                            ][i_link]
                        ]
                    )

    def _up_rotation(self) -> np.ndarray:
        """Return a 1x4x4 rotation matrix according to the 'up' attribute."""
        if self.up == "x":
            return geometry.create_transforms("z", [90], degrees=True)
        elif self.up == "y":
            return np.eye(4)[np.newaxis]
        elif self.up == "z":
            return geometry.create_transforms("x", [-90], degrees=True)
        elif self.up == "-x":
            return geometry.create_transforms("z", [-90], degrees=True)
        elif self.up == "-y":
            return geometry.create_transforms("z", [-180], degrees=True)
        elif self.up == "-z":
            return geometry.create_transforms("x", [90], degrees=True)
        else:
            raise ValueError(
                "up must be in {'x', 'y', 'z', '-x', '-y', '-z'}."
            )

    def _orient_contents(self) -> None:
        """
        Update self._oriented_points and _oriented_frames

        Rotate everything according to the up input, so that the end result
        is y up:

           |y
           |
           +---- x
          /
        z/

        Also add the global origin to _oriented_frames. Does not refresh.

        """
        self._oriented_points = self._contents.copy(copy_data=False)
        self._oriented_frames = self._contents.copy(copy_data=False)

        contents = self._contents.copy()
        # Add the global reference frame
        origin_name = "Origin"
        while origin_name in contents.data:
            origin_name += "_"
        contents.data[origin_name] = np.repeat(
            np.eye(4)[np.newaxis], len(contents.time), axis=0
        )

        rotation = self._up_rotation()

        # Orient points and frames
        for key in contents.data:
            if contents.data[key].shape[1:] == (4,):
                self._oriented_points.data[
                    key
                ] = geometry.get_global_coordinates(
                    contents.data[key], rotation
                )
            elif contents.data[key].shape[1:] == (4, 4):
                self._oriented_frames.data[
                    key
                ] = geometry.get_global_coordinates(
                    contents.data[key], rotation
                )

    # %% Projection and update

    def _project_to_camera(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Get a 3d --> 2d projection of a list of points.

        The method uses the class's camera variables to project a list of
        3d points onto a 2d canvas.

        Parameters
        ----------
        points_3d
            Nx4 array, where the first dimension is the number of points
            and  the second dimension is (x, y, z, 1).

        Returns
        -------
            Nx2 array, where the first dimension is the number of points and
            the second dimension is (x, y) to be ploted on a 2d graphic.

        """
        # ------------------------------------------------------------
        # Create the rotation matrix to convert the lab's coordinates
        # (x anterior, y up, z right) to the camera coordinates (x right,
        # y up, z deep)

        R = (
            np.array(
                [
                    [2 * self.zoom, 0, 0, 0],
                    [0, 2 * self.zoom, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
            @ np.array(
                [
                    [1, 0, 0, self.translation[0]],  # Pan
                    [0, 1, 0, self.translation[1]],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
            @ np.array(
                [
                    [1, 0, 0, 0],
                    [0, cos(-self.elevation), sin(self.elevation), 0],
                    [0, sin(-self.elevation), cos(-self.elevation), 0],
                    [0, 0, 0, 1],
                ]
            )
            @ np.array(
                [
                    [cos(-self.azimuth), 0, sin(self.azimuth), 0],
                    [0, 1, 0, 0],
                    [sin(-self.azimuth), 0, cos(-self.azimuth), 0],
                    [0, 0, 0, 1],
                ]
            )
            @ np.array(
                [
                    [1, 0, 0, -self.target[0]],  # Rotate around target
                    [0, 1, 0, -self.target[1]],
                    [0, 0, -1, self.target[2]],
                    [0, 0, 0, 1],
                ]
            )
        )

        # Add a first dimension to R and match first dimension of points_3d
        R = np.repeat(R[np.newaxis, :], points_3d.shape[0], axis=0)

        # Rotate points.
        rotated_points_3d = R @ points_3d[:, :, np.newaxis]
        rotated_points_3d = rotated_points_3d[:, :, 0]

        # Apply perspective.
        if self.perspective is True:
            # This uses an ugly magical constant but it works fine for now.
            denom = rotated_points_3d[:, 2] / 10 + 5
            rotated_points_3d[:, 0] = rotated_points_3d[:, 0] / denom
            rotated_points_3d[:, 1] = rotated_points_3d[:, 1] / denom
            with np.errstate(invalid="ignore"):
                to_remove = denom < 1e-12
            rotated_points_3d[to_remove, 0] = np.nan
            rotated_points_3d[to_remove, 1] = np.nan
        else:
            # Scale to match the point of view
            rotated_points_3d /= 5

        # Return only x and y
        return rotated_points_3d[:, 0:2]

    def _update_grid(self) -> None:
        """
        (Re)-create the grid.

        First create a ground plane matrix in the form:
            [
                [x1, 0, z1],
                [x2, 0, z2],
                [nan, nan, nan],
                [x3, 0, z3],
                [x4, 0, z4],
                [nan, nan, nan, nan],
                ...
            ]

        The grid is on the x and z axes, at y=0, as to be shown in Matplotlib.

        Then translate it according to the 'up' attribute and 'grid_origin'.

        A full refresh must be called after to recreate the Matplotlib plot
        using the correct width.

        """
        # Build the grid as an xz plane with y being up.
        temp_grid = []

        # z-to-z lines
        for x in np.arange(
            -self._grid_size / 2,
            self._grid_size / 2 + self._grid_subdivision_size,
            self._grid_subdivision_size,
        ):
            for z in np.arange(
                -self._grid_size / 2,
                self._grid_size / 2 + self._grid_subdivision_size,
                self._grid_subdivision_size,
            ):
                temp_grid.append([x, 0.0, z, 1.0])
            temp_grid.append([np.nan, np.nan, np.nan, np.nan])

        # x-to-x lines
        for z in np.arange(
            -self._grid_size / 2,
            self._grid_size / 2 + self._grid_subdivision_size,
            self._grid_subdivision_size,
        ):
            for x in np.arange(
                -self._grid_size / 2,
                self._grid_size / 2 + self._grid_subdivision_size,
                self._grid_subdivision_size,
            ):
                temp_grid.append([x, 0.0, z, 1.0])
            temp_grid.append([np.nan, np.nan, np.nan, np.nan])

        self._grid = np.array(temp_grid)

        # Translate the grid
        translation = geometry.get_global_coordinates(
            self._grid_origin[np.newaxis], self._up_rotation()
        )[0]
        translation[3] = 0  # Not a position, but a vector
        self._grid += translation

    def _update_points_and_interconnections(self) -> None:
        # Get a Nx4 matrices of every point at the current index
        points = self._oriented_points
        if points is None:
            return
        else:
            n_points = len(points.data)

        points_data = dict()  # Used to draw the points with different colors
        interconnection_points = dict()  # Used to draw the interconnections

        for color in COLORS:
            points_data[color] = np.empty([n_points, 4])
            points_data[color][:] = np.nan

            points_data[color + "s"] = np.empty([n_points, 4])
            points_data[color + "s"][:] = np.nan

        if n_points > 0:
            for i_point, point in enumerate(points.data):
                # Get this point's color
                if (
                    point in points.data_info
                    and "Color" in points.data_info[point]
                ):
                    color = points.data_info[point]["Color"]
                else:
                    color = "w"

                these_coordinates = points.data[point][self.current_index]
                points_data[color][i_point] = these_coordinates
                interconnection_points[point] = these_coordinates

        # Update the points plot
        for color in COLORS:
            # Unselected points
            points_data[color] = self._project_to_camera(points_data[color])
            self._mpl_objects["PointPlots"][color].set_data(
                points_data[color][:, 0], points_data[color][:, 1]
            )

            # Selected points
            points_data[color + "s"] = self._project_to_camera(
                points_data[color + "s"]
            )
            self._mpl_objects["PointPlots"][color + "s"].set_data(
                points_data[color + "s"][:, 0],
                points_data[color + "s"][:, 1],
            )

        # Draw the interconnections
        for interconnection in self._extended_interconnections:
            coordinates = []
            chains = self._extended_interconnections[interconnection]["Links"]

            for chain in chains:
                for point in chain:
                    try:
                        coordinates.append(interconnection_points[point])
                    except KeyError:
                        coordinates.append(np.repeat(np.nan, 4))

                coordinates.append(np.repeat(np.nan, 4))

            np_coordinates = np.array(coordinates)
            np_coordinates = self._project_to_camera(np_coordinates)

            self._mpl_objects["InterconnectionPlots"][
                interconnection
            ].set_data(np_coordinates[:, 0], np_coordinates[:, 1])

    def _fast_refresh(self) -> None:
        """Update plot data, assuming all plots have already been created."""
        self._update_points_and_interconnections()

        # Get three (3N)x4 matrices (for x, y and z lines) for the rigid bodies
        # at the current index
        frames = self._oriented_frames
        n_frames = len(frames.data)
        framex_data = np.empty([n_frames * 3, 4])
        framey_data = np.empty([n_frames * 3, 4])
        framez_data = np.empty([n_frames * 3, 4])

        for i_rigid_body, rigid_body in enumerate(frames.data):
            # Origin
            framex_data[i_rigid_body * 3] = frames.data[rigid_body][
                self.current_index, :, 3
            ]
            framey_data[i_rigid_body * 3] = frames.data[rigid_body][
                self.current_index, :, 3
            ]
            framez_data[i_rigid_body * 3] = frames.data[rigid_body][
                self.current_index, :, 3
            ]
            # Direction
            framex_data[i_rigid_body * 3 + 1] = frames.data[rigid_body][
                self.current_index
            ] @ np.array([self.frame_size, 0, 0, 1])
            framey_data[i_rigid_body * 3 + 1] = frames.data[rigid_body][
                self.current_index
            ] @ np.array([0, self.frame_size, 0, 1])
            framez_data[i_rigid_body * 3 + 1] = frames.data[rigid_body][
                self.current_index
            ] @ np.array([0, 0, self.frame_size, 1])
            # NaN to cut the line between the different frames
            framex_data[i_rigid_body * 3 + 2] = np.repeat(np.nan, 4)
            framey_data[i_rigid_body * 3 + 2] = np.repeat(np.nan, 4)
            framez_data[i_rigid_body * 3 + 2] = np.repeat(np.nan, 4)

        # Update the ground plane
        if len(self._grid) > 0:
            gp = self._project_to_camera(self._grid)
            self._mpl_objects["GridPlot"].set_data(gp[:, 0], gp[:, 1])

        # Create or update the frame plot
        framex_data = self._project_to_camera(framex_data)
        framey_data = self._project_to_camera(framey_data)
        framez_data = self._project_to_camera(framez_data)
        self._mpl_objects["FrameXPlot"].set_data(
            framex_data[:, 0], framex_data[:, 1]
        )
        self._mpl_objects["FrameYPlot"].set_data(
            framey_data[:, 0], framey_data[:, 1]
        )
        self._mpl_objects["FrameZPlot"].set_data(
            framez_data[:, 0], framez_data[:, 1]
        )

        # Update the window title
        try:
            self._mpl_objects["Figure"].canvas.manager.set_window_title(
                f"{self.current_index}/{len(self._contents.time)}: "
                + "%2.2f s." % self._contents.time[self.current_index]
            )
        except AttributeError:
            pass

        self._mpl_objects["Figure"].canvas.draw()

    def _set_new_target(self, target: ArrayLike) -> None:
        """Set new target and adapts translation and zoom consequently."""
        target = np.array(target)
        if np.sum(np.isnan(target)) > 0:
            return
        initial_translation = deepcopy(self.translation)
        initial_zoom = deepcopy(self.zoom)
        initial_target = deepcopy(self.target)

        n_points = len(self._oriented_points.data)
        points = np.empty((n_points, 4))
        for i_point, point in enumerate(self._oriented_points.data):
            points[i_point] = self._oriented_points.data[point][
                self.current_index
            ]

        initial_projected_points = self._project_to_camera(points)
        # Do not consider points that are not in the screen
        initial_projected_points[
            initial_projected_points[:, 0] < -1.5
        ] = np.nan
        initial_projected_points[initial_projected_points[:, 0] > 1.5] = np.nan
        initial_projected_points[
            initial_projected_points[:, 1] < -1.0
        ] = np.nan
        initial_projected_points[initial_projected_points[:, 1] > 1.0] = np.nan
        self.target = target

        def error_function(input):
            self._translation = input[0:2]
            self._zoom = input[2]
            new_projected_points = self._project_to_camera(points)
            error = np.nanmean(
                (initial_projected_points - new_projected_points) ** 2
            )
            return error

        res = optim.minimize(
            error_function, np.hstack((self.translation, self.zoom))
        )
        if res.success is False:
            self._translation = initial_translation
            self._zoom = initial_zoom
            self._target = initial_target

    # ------------------------------------
    # Helper functions
    def _set_index(self, index: int) -> None:
        """Set current index to a given index and update plots."""
        if index >= len(self._contents.time):
            self.current_index = len(self._contents.time) - 1
        elif index < 0:
            self.current_index = 0
        else:
            self.current_index = index

        if self.track is True and self._oriented_points is not None:
            new_target = self._oriented_points.data[self.last_selected_point][
                self.current_index
            ]
            if not np.isnan(np.sum(new_target)):
                self.target = new_target

    def _set_time(self, time: float) -> None:
        """Set current index to a given time and update plots."""
        index = int(np.argmin(np.abs(self._contents.time - time)))
        self._set_index(index)

    def _select_none(self) -> None:
        """Deselect every points."""
        if self._oriented_points is not None:
            for point in self._oriented_points.data:
                try:
                    # Keep 1st character, remove the possible 's'
                    self._oriented_points.data_info[point][
                        "Color"
                    ] = self._oriented_points.data_info[point]["Color"][0]
                except KeyError:
                    self._oriented_points = (
                        self._oriented_points.add_data_info(
                            point, "Color", "w"
                        )
                    )

    # ------------------------------------
    # Callbacks
    def _on_close(self, _) -> None:  # pragma: no cover
        # Release all references to objects
        self.close()

    def _on_timer(self, _) -> None:  # pragma: no cover
        """Implement callback for the animation timer object."""
        if self._running is True:
            # We check self._running because we can enter this callback
            # even if the animation has been deactivated. This is because the
            # recommended way to deactivate a timer is to unreference it,
            # however the garbage collector may take time deleting the timer
            # and we will end up with still a few timer callbacks. Checking
            # self._running makes sure that we effectively stop.
            current_index = self.current_index
            self._set_time(
                self._contents.time[self.current_index]
                + self.playback_speed
                * (time.time() - self._state["SystemTimeOnLastUpdate"])
            )
            if current_index == self.current_index:
                # The time wasn't enough to advance a frame. Articifically
                # advance a frame.
                self._set_index(self.current_index + 1)
            self._state["SystemTimeOnLastUpdate"] = time.time()

            self._fast_refresh()
        else:
            self._mpl_objects["Anim"].event_source.stop()

    def _on_pick(self, event):  # pragma: no cover
        """Implement callback for point selection."""
        if event.mouseevent.button == 1:
            index = event.ind
            selected_point = list(self._oriented_points.data.keys())[index[0]]
            self._mpl_objects["Axes"].set_title(selected_point)

            # Mark selected
            self._select_none()
            self._oriented_points.data_info[selected_point]["Color"] = (
                self._oriented_points.data_info[selected_point]["Color"][0]
                + "s"
            )

            # Set as new target
            self.last_selected_point = selected_point
            self._set_new_target(
                self._oriented_points.data[selected_point][self.current_index]
            )

            self._fast_refresh()

    def _on_key(self, event):  # pragma: no cover
        """Implement callback for keyboard key pressed."""
        if event.key == " ":
            if self._running is False:
                self._state["SystemTimeOnLastUpdate"] = time.time()
                self._running = True
                self._mpl_objects["Anim"].event_source.start()
            else:
                self._running = False
                self._mpl_objects["Anim"].event_source.stop()

        elif event.key == "left":
            self._set_index(self.current_index - 1)

        elif event.key == "shift+left":
            self._set_time(self._contents.time[self.current_index] - 1)

        elif event.key == "right":
            self._set_index(self.current_index + 1)

        elif event.key == "shift+right":
            self._set_time(self._contents.time[self.current_index] + 1)

        elif event.key == "-":
            self.playback_speed /= 2
            self._mpl_objects["Axes"].set_title(
                f"Playback set to {self.playback_speed}x"
            )

        elif event.key == "+":
            self.playback_speed *= 2
            self._mpl_objects["Axes"].set_title(
                f"Playback set to {self.playback_speed}x"
            )

        elif event.key == "h":
            if self._mpl_objects["HelpText"] is None:
                self._mpl_objects["HelpText"] = self._mpl_objects["Axes"].text(
                    -1.5,
                    -1,
                    HELP_TEXT,
                    color=[0, 1, 0],
                    fontfamily="monospace",
                )
            else:
                self._mpl_objects["HelpText"].remove()
                self._mpl_objects["HelpText"] = None

        elif event.key == "d":
            self.perspective = not self.perspective
            if self.perspective is True:
                self._mpl_objects["Axes"].set_title(
                    "Camera set to perspective"
                )
            else:
                self._mpl_objects["Axes"].set_title("Camera set to orthogonal")

        elif event.key == "t":
            self.track = not self.track
            if self.track is True:
                self._mpl_objects["Axes"].set_title("Point tracking activated")
            else:
                self._mpl_objects["Axes"].set_title(
                    "Point tracking deactivated"
                )

        elif event.key == "shift":
            self._state["ShiftPressed"] = True

        self._fast_refresh()

    def _on_release(self, event):  # pragma: no cover
        if event.key == "shift":
            self._state["ShiftPressed"] = False

    def _on_scroll(self, event):  # pragma: no cover
        if event.button == "up":
            self.zoom *= 1.05
        elif event.button == "down":
            self.zoom /= 1.05
        self._fast_refresh()

    def _on_mouse_press(self, event):  # pragma: no cover
        if len(self.last_selected_point) > 0:
            self._set_new_target(
                self._oriented_points.data[self.last_selected_point][
                    self.current_index
                ]
            )

        self._state["TranslationOnMousePress"] = self.translation
        self._state["AzimutOnMousePress"] = self.azimuth
        self._state["ElevationOnMousePress"] = self.elevation
        self._state["ZoomOnMousePress"] = self.zoom
        self._state["MousePositionOnPress"] = (event.x, event.y)
        if event.button == 1:
            self._state["MouseLeftPressed"] = True
        elif event.button == 2:
            self._state["MouseMiddlePressed"] = True
        elif event.button == 3:
            self._state["MouseRightPressed"] = True

    def _on_mouse_release(self, event):  # pragma: no cover
        if event.button == 1:
            self._state["MouseLeftPressed"] = False
        elif event.button == 2:
            self._state["MouseMiddlePressed"] = False
        elif event.button == 3:
            self._state["MouseRightPressed"] = False

    def _on_mouse_motion(self, event):  # pragma: no cover
        # Pan:
        if (
            self._state["MouseLeftPressed"] and self._state["ShiftPressed"]
        ) or self._state["MouseMiddlePressed"]:
            self.translation = (
                self._state["TranslationOnMousePress"][0]
                + (event.x - self._state["MousePositionOnPress"][0])
                / (100 * self.zoom),
                self._state["TranslationOnMousePress"][1]
                + (event.y - self._state["MousePositionOnPress"][1])
                / (100 * self.zoom),
            )
            self._fast_refresh()

        # Rotation:
        elif (
            self._state["MouseLeftPressed"] and not self._state["ShiftPressed"]
        ):
            self.azimuth = (
                self._state["AzimutOnMousePress"]
                - (event.x - self._state["MousePositionOnPress"][0]) / 250
            )
            self.elevation = (
                self._state["ElevationOnMousePress"]
                - (event.y - self._state["MousePositionOnPress"][1]) / 250
            )
            self._fast_refresh()

        # Zoom:
        elif self._state["MouseRightPressed"]:
            self.zoom = (
                self._state["ZoomOnMousePress"]
                + (event.y - self._state["MousePositionOnPress"][1]) / 250
            )
            self._fast_refresh()

    def _to_animation(
        self,
    ):
        """
        Create a matplotlib FuncAnimation for displaying in Jupyter notebooks.

        Parameters
        ----------
        No parameter.

        Returns
        -------
        A FuncAnimation to be displayed by Jupyter notebook.

        """
        mpl.rcParams["animation.html"] = "html5"

        self.playback_speed = 0
        self._mpl_objects["Figure"].set_size_inches(6, 4.5)  # Half size
        self._set_index(0)
        self._running = True
        #        self._mpl_objects["Anim"].frames = stop_index - start_index
        #        self._mpl_objects["Anim"].save_count = stop_index - start_index
        self._mpl_objects["Anim"].event_source.start()
        plt.close(self._mpl_objects["Figure"])
        return self._mpl_objects["Anim"]

    # %% Public methods

    def close(self) -> None:
        """Close the Player and its associated window."""
        plt.close(self._mpl_objects["Figure"])
        self._mpl_objects = {}

    def refresh(self):
        """
        Perform a full refresh of the Player.

        Normally, this function does not need to be called by the user. Use it
        if for an unknown reason, the Player is not refreshed as it should.
        You can also report this need as a bug in the issue tracker:
        https://github.com/kineticstoolkit/kineticstoolkit/issues

        """
        # Clear and rebuild the mpl plots.
        self._mpl_objects["InterconnectionPlots"] = dict()
        self._mpl_objects["PointPlots"] = dict()
        self._mpl_objects["GridPlot"] = None
        self._mpl_objects["FrameXPlot"] = None
        self._mpl_objects["FrameYPlot"] = None
        self._mpl_objects["FrameZPlot"] = None
        self._mpl_objects["GridPlot"] = None
        self._mpl_objects["HelpText"] = None

        self._mpl_objects["Axes"].clear()
        self._mpl_objects["Figure"].set_facecolor(self._background_color)

        # Reset axes properties
        self._mpl_objects["Axes"].set_axis_off()

        # Create the ground plane
        self._mpl_objects["GridPlot"] = self._mpl_objects["Axes"].plot(
            np.nan,
            np.nan,
            linewidth=self._grid_width,
            color=self._grid_color,
        )[0]

        # Create the interconnection plots
        for interconnection in self._extended_interconnections:
            self._mpl_objects["InterconnectionPlots"][
                interconnection
            ] = self._mpl_objects["Axes"].plot(
                np.nan,
                np.nan,
                "-",
                c=self._extended_interconnections[interconnection]["Color"],
                linewidth=self._interconnection_width,
            )[
                0
            ]

        # Create the frame plots
        self._mpl_objects["FrameXPlot"] = self._mpl_objects["Axes"].plot(
            np.nan,
            np.nan,
            c="r",
            linewidth=self.frame_width,
        )[0]
        self._mpl_objects["FrameYPlot"] = self._mpl_objects["Axes"].plot(
            np.nan,
            np.nan,
            c="g",
            linewidth=self.frame_width,
        )[0]
        self._mpl_objects["FrameZPlot"] = self._mpl_objects["Axes"].plot(
            np.nan,
            np.nan,
            c="b",
            linewidth=self.frame_width,
        )[0]

        # Create the point plots
        colors = {
            "r": [1, 0, 0],
            "g": [0, 1, 0],
            "b": [0.3, 0.3, 1],
            "y": [1, 1, 0],
            "m": [1, 0, 1],
            "c": [0, 1, 1],
            "w": [0.8, 0.8, 0.8],
        }

        for color in COLORS:
            self._mpl_objects["PointPlots"][color] = self._mpl_objects[
                "Axes"
            ].plot(
                np.nan,
                np.nan,
                ".",
                c=colors[color],
                markersize=self._point_size,
                pickradius=1.1 * self._point_size,
                picker=True,
            )[
                0
            ]

            self._mpl_objects["PointPlots"][color + "s"] = self._mpl_objects[
                "Axes"
            ].plot(
                np.nan,
                np.nan,
                ".",
                c=colors[color],
                markersize=3 * self._point_size,
            )[
                0
            ]

        # Add the title
        title_obj = plt.title("Player")
        plt.setp(title_obj, color=[0, 1, 0])  # Set a green title

        self._fast_refresh()  # Draw everything once

        # Set limits once it's drawn
        self._mpl_objects["Axes"].set_xlim([-1.5, 1.5])
        self._mpl_objects["Axes"].set_ylim([-1.0, 1.0])

    # %% Deprecated methods
    @deprecated(
        since="0.12",
        until="2024",
        details="This method has been removed because it did not return html5 and "
        "was mainly a hack for representing videos in tutorials. The "
        "supported way to use the Player is interactively.",
    )
    def to_html5(self, **kwargs):
        return self._to_animation()
