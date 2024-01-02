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
Provides the Player class to visualize markers and rigid bodies in 3d.

The Player class is accessible directly from the toplevel Kinetics Toolkit
namespace (i.e., ktk.Player).
"""
from __future__ import annotations


__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"

REPR_HTML_MAX_DURATION = 10  # Max duration for _repr_html

from kineticstoolkit.timeseries import TimeSeries
from kineticstoolkit.decorators import deprecated
from kineticstoolkit.tools import check_interactive_backend
import kineticstoolkit.geometry as geometry

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import animation
import numpy as np
from numpy import sin, cos
import time
import copy
from typing import Any
from numpy.typing import ArrayLike
import warnings

# To fit the new viewpoint on selecting a new marker
import scipy.optimize as optim


class Player:
    # FIXME! Update this docstring.
    """
    A class that allows visualizing markers and rigid bodies in 3D.

    `player = ktk.Player(parameters)` creates and launches an interactive
    Player instance. Once the window is open, press `h` to show a help
    overlay.

    Parameters
    ----------
    *ts
        Contains the markers and rigid bodies to visualize, where each data
        key is either a marker position expressed as Nx4 array, or a frame
        expressed as a Nx4x4 array. Multiple TimeSeries can be provided.

    interconnections
        Optional. Each key corresponds to an inerconnection between markers,
        where one interconnection is another dict with the following keys:

        - "Links": list of lists strings, where each string is a marker
          name. For example, to create a link that spans Marker1 and Marker2,
          and another link that spans Marker3, Marker4 and Marker5,
          interconnections["Links"] would be::

              [["Marker1", "Marker2"], ["Marker3", "Marker4", "Marker5"]]

        - "Color": character or tuple (RGB) that represents the color of the
          link. Color must be a valid value for matplotlib's
          plots.

    current_index
        Optional. Sets the inital index number to show.

    marker_radius
        Optional. Sets the marker radius as defined by matplotlib.

    axis_length
        Optional. Sets the rigid body size in meters.

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
        selected marker when changing index.

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
        group: str = "",
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
        marker_radius: float = 0.008,
        axis_length: float = 0.1,
        axis_width: float = 3.0,
        interconnection_width: float = 1.5,
        grid_color: tuple[float, float, float] | None = (0.8, 0.8, 0.8),
        background_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
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
        self._source_timeseries = dict()
        self._source_timeseries[group] = [_.copy() for _ in ts]
        self._source_interconnections = dict()
        self._source_interconnections[group] = interconnections

        self.current_index = current_index
        if current_time is not None:
            self.current_time = current_time
        self.playback_speed = playback_speed
        self.up = up
        self.zoom = zoom
        self.azimuth = azimuth
        self.elevation = elevation
        self.translation = translation
        self.target = target
        self.perspective = perspective
        self.track = track
        self.marker_radius = marker_radius
        self.axis_length = axis_length
        self.axis_width = axis_width
        self.interconnection_width = interconnection_width
        self.grid_color = grid_color
        self.background_color = background_color

        self.continue_init()  # temp

    @property
    def source_timeseries(self) -> dict[str, list[TimeSeries]]:
        """The source TimeSeries, separated in groups."""
        return self._source_timeseries

    @property
    def merged_timeseries(self) -> TimeSeries:
        """The TimeSeries, merged and prefixed by group name."""
        out = TimeSeries()
        for group in self.source_timeseries:
            for ts in self.source_timeseries[group]:
                for key in ts.data:
                    temp_ts = ts.get_subset(key)
                    if group != "":
                        temp_ts.rename_data(key, f"{group}:{key}", in_place=True)
                    out.merge(temp_ts, in_place=True)
        return out

    @property
    def source_interconnections(self) -> dict[str, dict[str, dict[str, Any]]]:
        """The source interconnections, separated in groups."""
        return self._source_interconnections

    @property
    def merged_interconnections(self) -> dict[str, dict[str, Any]]:
        """The merged interconnections, renamed to match merged_timeseries."""
        out = {}
        for group in self.source_interconnections:
            # Go through every body segment
            for body_name in self.source_interconnections[group]:
                out_key = body_name if group == "" else f"{group}:{body_name}"
                out[out_key] = dict()
                out[out_key]["Color"] = self.source_interconnections[group][
                    body_name
                ]["Color"]
                out[out_key]["Links"] = []

                # Go through every link of this segment
                for i_link, link in enumerate(
                    self.source_interconnections[group][body_name]
                ):
                    out[out_key]["Links"].append(
                        [
                            f"{group}:{s}"
                            for s in self.source_interconnections[group][
                                body_name
                            ]["Links"][i_link]
                        ]
                    )
        return out

    def continue_init(self):
        """temp."""
        ts = self.merged_timeseries

        # ---------------------------------------------------------------
        # Set self.n_indexes and self.time, and verify that we have at least
        # markers or rigid bodies.

        # FIXME! With the Player API, the Player should be
        # functionning without any TimeSeries (just showing nothing).

        self.time = ts.time
        self.n_indexes = len(self.time)

        # ---------------------------------------------------------------
        # Decompose the input TimeSeries' contents into points and frames

        # FIXME! This should be changed to a proper function that is
        # run automatically on each Player.add() or Player.remove() function.
        self._points = TimeSeries()
        self._frames = TimeSeries()

        for key in ts.data:
            if ts.data[key].shape[1:] == (4,):
                self._points.data[key] = ts.data[key]
                if key in ts.data_info:
                    self._points.data_info[key] = ts.data_info[key]

            elif ts.data[key].shape[1:] == (4, 4):
                # Add the frames
                self._frames.data[key] = ts.data[key]
                if key in ts.data_info:
                    self._frames.data_info[key] = ts.data_info[key]

                # Add the frame origins as "markers"
                key_label = f"[{key}] origin"
                self._points.data[key_label] = ts.data[key][:, :, 3]
                if key in ts.data_info:
                    self._points.data_info[key_label] = ts.data_info[
                        key
                    ]

            else:
                warnings.warn(
                    f"The data key {key} has a shape of "
                    f"{ts.data[key].shape}. However, the Player can "
                    "only show points (Nx4) and frames Nx4x4. Therefore, "
                    f"{key} won't be shown in the Player."
                )

        self._select_none()
        self.last_selected_marker = ""

        # Add the origin to the rigid bodies
        self._frames.data["Global"] = np.repeat(
            np.eye(4, 4)[np.newaxis, :, :], self.n_indexes, axis=0
        )

        # Rotate everything according to the up input, so that the end result
        # is y up:
        #
        #    |y
        #    |
        #    +---- x
        #   /
        # z/
        if self.up == "x":
            rotation = geometry.create_transforms("z", [90], degrees=True)
        elif self.up == "y":
            rotation = np.eye(4)[np.newaxis]
        elif self.up == "z":
            rotation = geometry.create_transforms("x", [-90], degrees=True)
        elif self.up == "-x":
            rotation = geometry.create_transforms("z", [-90], degrees=True)
        elif self.up == "-y":
            rotation = geometry.create_transforms("z", [-180], degrees=True)
        elif self.up == "-z":
            rotation = geometry.create_transforms("x", [90], degrees=True)
        else:
            raise ValueError(
                "up must be in {'x', 'y', 'z', '-x', '-y', '-z'}."
            )
        for key in self._points.data:
            self._points.data[key] = geometry.get_global_coordinates(
                self._points.data[key], rotation
            )
        for key in self._frames.data:
            self._frames.data[key] = geometry.get_global_coordinates(
                self._frames.data[key], rotation
            )

        # Camera target
        target1x4 = np.ones((1, 4))
        target1x4[0, 0:3] = self.target
        self.target = geometry.get_global_coordinates(target1x4, rotation)[0]

        self.playback_speed = 1.0  # 0.0 to show all samples
        #  self._anim = None  # Will initialize in _create_figure

        # Init objects
        self._objects = dict()  # type: dict[str, Any]
        self._colors = ["r", "g", "b", "y", "c", "m", "w"]
        self._objects["PlotMarkers"] = dict()
        for color in self._colors:
            self._objects["PlotMarkers"][color] = None  # Not selected
            self._objects["PlotMarkers"][color + "s"] = None  # Selected
        self._objects["PlotRigidBodiesX"] = None
        self._objects["PlotRigidBodiesY"] = None
        self._objects["PlotRigidBodiesZ"] = None
        self._objects["PlotGroundPlane"] = None
        self._objects["PlotInterconnections"] = dict()

        self._objects["Figure"] = None
        self._objects["Axes"] = None
        self._objects["Help"] = None

        # Init mouse navigation state
        self._state = dict()  # type: dict[str, Any]
        self._state["ShiftPressed"] = False
        self._state["MouseLeftPressed"] = False
        self._state["MouseMiddlePressed"] = False
        self._state["MouseRightPressed"] = False
        self._state["MousePositionOnPress"] = (0.0, 0.0)
        self._state["MousePositionOnMiddlePress"] = (0.0, 0.0)
        self._state["MousePositionOnRightPress"] = (0.0, 0.0)
        self._state["TranslationOnMousePress"] = (0.0, 0.0)
        self._state["AzimutOnMousePress"] = 0.0
        self._state["ElevationOnMousePress"] = 0.0
        self._state["SelfTimeOnPlay"] = self.time[0]
        self._state["SystemTimeOnLastUpdate"] = time.time()

        self._help_text = """
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
            select a marker     : left-click
            3d rotate           : left-drag
            pan                 : middle-drag or shift+left-drag
            zoom                : right-drag or wheel
            """

        self._create_figure()
        self._create_interconnections()
        self._create_markers()
        self._create_ground_plane()
        self._first_refresh()

    def __dir__(self):
        """Return directory."""
        return ["close"]

    def _create_figure(self) -> None:
        """Create the player's figure."""
        # Create the figure and axes
        self._objects["Figure"], self._objects["Axes"] = plt.subplots(
            num=None, figsize=(12, 9), facecolor="k", edgecolor="w"
        )

        # Remove the toolbar
        try:  # Try, setVisible method not always there
            self._objects["Figure"].canvas.toolbar.setVisible(False)
        except AttributeError:
            pass

        plt.tight_layout()

        # Add the title
        title_obj = plt.title("Player")
        plt.setp(title_obj, color=[0, 1, 0])  # Set a green title

        # Remove the background for faster plotting
        self._objects["Axes"].set_axis_off()

        # Add the animation timer

        # FIXME! All attributes should be initialized in the constructor.
        # Not everywhere like now.

        self._anim = animation.FuncAnimation(
            self._objects["Figure"],
            self._on_timer,  # type: ignore
            frames=self.n_indexes,
            interval=33,
        )  # 30 ips
        self._running = False

        # Connect the callback functions
        self._objects["Figure"].canvas.mpl_connect("pick_event", self._on_pick)
        self._objects["Figure"].canvas.mpl_connect(
            "key_press_event", self._on_key
        )
        self._objects["Figure"].canvas.mpl_connect(
            "key_release_event", self._on_release
        )
        self._objects["Figure"].canvas.mpl_connect(
            "scroll_event", self._on_scroll
        )
        self._objects["Figure"].canvas.mpl_connect(
            "button_press_event", self._on_mouse_press
        )
        self._objects["Figure"].canvas.mpl_connect(
            "button_release_event", self._on_mouse_release
        )
        self._objects["Figure"].canvas.mpl_connect(
            "motion_notify_event", self._on_mouse_motion
        )

    def _create_interconnections(self) -> None:
        """Create the interconnections plots in the player's figure."""
        for interconnection in self.merged_interconnections:
            self._objects["PlotInterconnections"][
                interconnection
            ] = self._objects["Axes"].plot(
                np.nan,
                np.nan,
                "-",
                c=self.merged_interconnections[interconnection]["Color"],
                linewidth=self.interconnection_width,
            )[
                0
            ]

    def _create_markers(self) -> None:
        """Create the markers plots in the player's figure."""
        colors = {
            "r": [1, 0, 0],
            "g": [0, 1, 0],
            "b": [0.3, 0.3, 1],
            "y": [1, 1, 0],
            "m": [1, 0, 1],
            "c": [0, 1, 1],
            "w": [0.8, 0.8, 0.8],
        }

        for color in self._colors:
            self._objects["PlotMarkers"][color] = self._objects["Axes"].plot(
                np.nan,
                np.nan,
                ".",
                c=colors[color],
                markersize=4,
                pickradius=5,
                picker=True,
            )[0]
        for color in self._colors:
            self._objects["PlotMarkers"][color + "s"] = self._objects[
                "Axes"
            ].plot(np.nan, np.nan, ".", c=colors[color], markersize=12)[0]

    def _create_ground_plane(self) -> None:
        # Create the ground plane matrix
        gp_size = 30  # blocks
        gp_div = 4  # blocks per meter
        gp_x = np.block(
            [
                np.tile(
                    [-gp_size / gp_div, gp_size / gp_div, np.nan], gp_size
                ),
                np.repeat(
                    np.linspace(-gp_size / gp_div, gp_size / gp_div, gp_size),
                    3,
                ),
            ]
        )
        gp_y = np.zeros(6 * gp_size)
        gp_z = np.block(
            [
                np.repeat(
                    np.linspace(-gp_size / gp_div, gp_size / gp_div, gp_size),
                    3,
                ),
                np.tile(
                    [-gp_size / gp_div, gp_size / gp_div, np.nan], gp_size
                ),
            ]
        )
        gp_1 = np.ones(6 * gp_size)
        self._ground_plane = np.hstack(
            [
                gp_x[:, np.newaxis],
                gp_y[:, np.newaxis],
                gp_z[:, np.newaxis],
                gp_1[:, np.newaxis],
            ]
        )

    def _first_refresh(self) -> None:
        """Draw the stuff and set the axis size."""
        self._update_plots()
        plt.axis((-1.5, 1.5, -1.0, 1.0))

    def _get_projection(self, points_3d: np.ndarray) -> np.ndarray:
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

    def _update_markers_and_interconnections(self) -> None:
        # Get a Nx4 matrices of every marker at the current index
        markers = self._points
        if markers is None:
            return
        else:
            n_markers = len(markers.data)

        markers_data = dict()  # Used to draw the markers with different colors
        interconnection_markers = dict()  # Used to draw the interconnections

        for color in self._colors:
            markers_data[color] = np.empty([n_markers, 4])
            markers_data[color][:] = np.nan

            markers_data[color + "s"] = np.empty([n_markers, 4])
            markers_data[color + "s"][:] = np.nan

        if n_markers > 0:
            for i_marker, marker in enumerate(markers.data):
                # Get this marker's color
                if (
                    marker in markers.data_info
                    and "Color" in markers.data_info[marker]
                ):
                    color = markers.data_info[marker]["Color"]
                else:
                    color = "w"

                these_coordinates = markers.data[marker][self.current_index]
                markers_data[color][i_marker] = these_coordinates
                interconnection_markers[marker] = these_coordinates

        # Update the markers plot
        for color in self._colors:
            # Unselected markers
            markers_data[color] = self._get_projection(markers_data[color])
            self._objects["PlotMarkers"][color].set_data(
                markers_data[color][:, 0], markers_data[color][:, 1]
            )

            # Selected markers
            markers_data[color + "s"] = self._get_projection(
                markers_data[color + "s"]
            )
            self._objects["PlotMarkers"][color + "s"].set_data(
                markers_data[color + "s"][:, 0],
                markers_data[color + "s"][:, 1],
            )

        # Draw the interconnections
        for interconnection in self.merged_interconnections:
            coordinates = []
            chains = self.merged_interconnections[interconnection]["Links"]

            for chain in chains:
                for marker in chain:
                    try:
                        coordinates.append(interconnection_markers[marker])
                    except KeyError:
                        coordinates.append(np.repeat(np.nan, 4))

                coordinates.append(np.repeat(np.nan, 4))

            np_coordinates = np.array(coordinates)
            np_coordinates = self._get_projection(np_coordinates)

            self._objects["PlotInterconnections"][
                interconnection
            ].set_data(np_coordinates[:, 0], np_coordinates[:, 1])

    def _update_plots(self) -> None:
        """Update the plots, or draw it if not plot has been drawn before."""
        self._update_markers_and_interconnections()

        # Get three (3N)x4 matrices (for x, y and z lines) for the rigid bodies
        # at the current index
        rigid_bodies = self._frames
        n_rigid_bodies = len(rigid_bodies.data)
        rbx_data = np.empty([n_rigid_bodies * 3, 4])
        rby_data = np.empty([n_rigid_bodies * 3, 4])
        rbz_data = np.empty([n_rigid_bodies * 3, 4])

        for i_rigid_body, rigid_body in enumerate(rigid_bodies.data):
            # Origin
            rbx_data[i_rigid_body * 3] = rigid_bodies.data[rigid_body][
                self.current_index, :, 3
            ]
            rby_data[i_rigid_body * 3] = rigid_bodies.data[rigid_body][
                self.current_index, :, 3
            ]
            rbz_data[i_rigid_body * 3] = rigid_bodies.data[rigid_body][
                self.current_index, :, 3
            ]
            # Direction
            rbx_data[i_rigid_body * 3 + 1] = rigid_bodies.data[rigid_body][
                self.current_index
            ] @ np.array([self.axis_length, 0, 0, 1])
            rby_data[i_rigid_body * 3 + 1] = rigid_bodies.data[rigid_body][
                self.current_index
            ] @ np.array([0, self.axis_length, 0, 1])
            rbz_data[i_rigid_body * 3 + 1] = rigid_bodies.data[rigid_body][
                self.current_index
            ] @ np.array([0, 0, self.axis_length, 1])
            # NaN to cut the line between the different rigid bodies
            rbx_data[i_rigid_body * 3 + 2] = np.repeat(np.nan, 4)
            rby_data[i_rigid_body * 3 + 2] = np.repeat(np.nan, 4)
            rbz_data[i_rigid_body * 3 + 2] = np.repeat(np.nan, 4)

        # Update the ground plane
        gp = self._get_projection(self._ground_plane)
        if self._objects["PlotGroundPlane"] is None:  # Create the plot
            self._objects["PlotGroundPlane"] = self._objects["Axes"].plot(
                gp[:, 0], gp[:, 1], c=[0.3, 0.3, 0.3], linewidth=1
            )[0]
        else:  # Update the plot
            self._objects["PlotGroundPlane"].set_data(gp[:, 0], gp[:, 1])

        # Create or update the rigid bodies plot
        rbx_data = self._get_projection(rbx_data)
        rby_data = self._get_projection(rby_data)
        rbz_data = self._get_projection(rbz_data)
        if self._objects["PlotRigidBodiesX"] is None:  # Create the plot
            self._objects["PlotRigidBodiesX"] = self._objects["Axes"].plot(
                rbx_data[:, 0],
                rbx_data[:, 1],
                c="r",
                linewidth=self.axis_width,
            )[0]
            self._objects["PlotRigidBodiesY"] = self._objects["Axes"].plot(
                rby_data[:, 0],
                rby_data[:, 1],
                c="g",
                linewidth=self.axis_width,
            )[0]
            self._objects["PlotRigidBodiesZ"] = self._objects["Axes"].plot(
                rbz_data[:, 0],
                rbz_data[:, 1],
                c="b",
                linewidth=self.axis_width,
            )[0]
        else:  # Update the plot
            self._objects["PlotRigidBodiesX"].set_data(
                rbx_data[:, 0], rbx_data[:, 1]
            )
            self._objects["PlotRigidBodiesY"].set_data(
                rby_data[:, 0], rby_data[:, 1]
            )
            self._objects["PlotRigidBodiesZ"].set_data(
                rbz_data[:, 0], rbz_data[:, 1]
            )

        # Update the window title
        try:
            self._objects["Figure"].canvas.manager.set_window_title(
                f"{self.current_index}/{self.n_indexes}: "
                + "%2.2f s." % self.time[self.current_index]
            )
        except AttributeError:
            pass

        self._objects["Figure"].canvas.draw()

    def _set_new_target(self, target: ArrayLike) -> None:
        """Set new target and adapts translation and zoom consequently."""
        target = np.array(target)
        if np.sum(np.isnan(target)) > 0:
            return
        initial_translation = copy.deepcopy(self.translation)
        initial_zoom = copy.deepcopy(self.zoom)
        initial_target = copy.deepcopy(self.target)

        n_markers = len(self._points.data)
        markers = np.empty((n_markers, 4))
        for i_marker, marker in enumerate(self._points.data):
            markers[i_marker] = self._points.data[marker][self.current_index]

        initial_projected_markers = self._get_projection(markers)
        # Do not consider markers that are not in the screen
        initial_projected_markers[
            initial_projected_markers[:, 0] < -1.5
        ] = np.nan
        initial_projected_markers[
            initial_projected_markers[:, 0] > 1.5
        ] = np.nan
        initial_projected_markers[
            initial_projected_markers[:, 1] < -1.0
        ] = np.nan
        initial_projected_markers[
            initial_projected_markers[:, 1] > 1.0
        ] = np.nan
        self.target = target

        def error_function(input):
            self.translation = input[0:2]
            self.zoom = input[2]
            new_projected_markers = self._get_projection(markers)
            error = np.nanmean(
                (initial_projected_markers - new_projected_markers) ** 2
            )
            return error

        res = optim.minimize(
            error_function, np.hstack((self.translation, self.zoom))
        )
        if res.success is False:
            self.translation = initial_translation
            self.zoom = initial_zoom
            self.target = initial_target

    # ------------------------------------
    # Helper functions
    def _set_index(self, index: int) -> None:
        """Set current index to a given index and update plots."""
        if index >= self.n_indexes:
            self.current_index = self.n_indexes - 1
        elif index < 0:
            self.current_index = 0
        else:
            self.current_index = index

        if self.track is True and self._points is not None:
            new_target = self._points.data[self.last_selected_marker][
                self.current_index
            ]
            if not np.isnan(np.sum(new_target)):
                self.target = new_target

    def _set_time(self, time: float) -> None:
        """Set current index to a given time and update plots."""
        index = int(np.argmin(np.abs(self.time - time)))
        self._set_index(index)

    def _select_none(self) -> None:
        """Deselect every markers."""
        if self._points is not None:
            for marker in self._points.data:
                try:
                    # Keep 1st character, remove the possible 's'
                    self._points.data_info[marker][
                        "Color"
                    ] = self._points.data_info[marker]["Color"][0]
                except KeyError:
                    self._points = self._points.add_data_info(
                        marker, "Color", "w"
                    )

    def close(self) -> None:
        """Close the Player and its associated window."""
        plt.close(self._objects["Figure"])
        self._objects = {}

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
                self.time[self.current_index]
                + self.playback_speed
                * (time.time() - self._state["SystemTimeOnLastUpdate"])
            )
            if current_index == self.current_index:
                # The time wasn't enough to advance a frame. Articifically
                # advance a frame.
                self._set_index(self.current_index + 1)
            self._state["SystemTimeOnLastUpdate"] = time.time()

            self._update_plots()
        else:
            self._anim.event_source.stop()

    def _on_pick(self, event):  # pragma: no cover
        """Implement callback for marker selection."""
        if event.mouseevent.button == 1:
            index = event.ind
            selected_marker = list(self._points.data.keys())[index[0]]
            self._objects["Axes"].set_title(selected_marker)

            # Mark selected
            self._select_none()
            self._points.data_info[selected_marker]["Color"] = (
                self._points.data_info[selected_marker]["Color"][0] + "s"
            )

            # Set as new target
            self.last_selected_marker = selected_marker
            self._set_new_target(
                self._points.data[selected_marker][self.current_index]
            )

            self._update_plots()

    def _on_key(self, event):  # pragma: no cover
        """Implement callback for keyboard key pressed."""
        if event.key == " ":
            if self._running is False:
                self._state["SystemTimeOnLastUpdate"] = time.time()
                self._state["SelfTimeOnPlay"] = self.time[self.current_index]
                self._running = True
                self._anim.event_source.start()
            else:
                self._running = False
                self._anim.event_source.stop()

        elif event.key == "left":
            self._set_index(self.current_index - 1)

        elif event.key == "shift+left":
            self._set_time(self.time[self.current_index] - 1)

        elif event.key == "right":
            self._set_index(self.current_index + 1)

        elif event.key == "shift+right":
            self._set_time(self.time[self.current_index] + 1)

        elif event.key == "-":
            self.playback_speed /= 2
            self._objects["Axes"].set_title(
                f"Playback set to {self.playback_speed}x"
            )

        elif event.key == "+":
            self.playback_speed *= 2
            self._objects["Axes"].set_title(
                f"Playback set to {self.playback_speed}x"
            )

        elif event.key == "h":
            if self._objects["Help"] is None:
                self._objects["Help"] = self._objects["Axes"].text(
                    -1.5,
                    -1,
                    self._help_text,
                    color=[0, 1, 0],
                    fontfamily="monospace",
                )
            else:
                self._objects["Help"].remove()
                self._objects["Help"] = None

        elif event.key == "d":
            self.perspective = not self.perspective
            if self.perspective is True:
                self._objects["Axes"].set_title("Camera set to perspective")
            else:
                self._objects["Axes"].set_title("Camera set to orthogonal")

        elif event.key == "t":
            self.track = not self.track
            if self.track is True:
                self._objects["Axes"].set_title("Marker tracking activated")
            else:
                self._objects["Axes"].set_title("Marker tracking deactivated")

        elif event.key == "shift":
            self._state["ShiftPressed"] = True

        self._update_plots()

    def _on_release(self, event):  # pragma: no cover
        if event.key == "shift":
            self._state["ShiftPressed"] = False

    def _on_scroll(self, event):  # pragma: no cover
        if event.button == "up":
            self.zoom *= 1.05
        elif event.button == "down":
            self.zoom /= 1.05
        self._update_plots()

    def _on_mouse_press(self, event):  # pragma: no cover
        if len(self.last_selected_marker) > 0:
            self._set_new_target(
                self._points.data[self.last_selected_marker][
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
            self._update_plots()

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
            self._update_plots()

        # Zoom:
        elif self._state["MouseRightPressed"]:
            self.zoom = (
                self._state["ZoomOnMousePress"]
                + (event.y - self._state["MousePositionOnPress"][1]) / 250
            )
            self._update_plots()

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
        self._objects["Figure"].set_size_inches(6, 4.5)  # Half size
        self._set_index(0)
        self._running = True
        #        self._anim.frames = stop_index - start_index
        #        self._anim.save_count = stop_index - start_index
        self._anim.event_source.start()
        plt.close(self._objects["Figure"])
        return self._anim

    @deprecated(
        since="0.12",
        until="2024",
        details="This method has been removed because it did not return html5 and "
        "was mainly a hack for representing videos in tutorials. The "
        "supported way to use the Player is interactively.",
    )
    def to_html5(self, **kwargs):
        return self._to_animation()
