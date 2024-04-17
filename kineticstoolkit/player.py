#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020-2024 Félix Chénier

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

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2024 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


from kineticstoolkit.timeseries import TimeSeries
from kineticstoolkit.tools import check_interactive_backend
import kineticstoolkit.geometry as geometry
from kineticstoolkit._repr import _format_dict_entries

from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from numpy import sin, cos
import time
from copy import deepcopy
from typing import Any
from kineticstoolkit.typing_ import ArrayLike, check_param
import warnings

# To fit the new viewpoint on selecting a new point
import scipy.optimize as optim

REPR_HTML_MAX_DURATION = 10  # Max duration for _repr_html
PALETTE = {
    "k": (0.0, 0.0, 0.0),
    "r": (1.0, 0.0, 0.0),
    "g": (0.0, 1.0, 0.0),
    "b": (0.3, 0.3, 1.0),
    "y": (1.0, 1.0, 0.0),
    "m": (1.0, 0.0, 1.0),
    "c": (0.0, 1.0, 1.0),
    "w": (1.0, 1.0, 1.0),
}


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
    set back/front view : 1/2
    set left/right view : 3/4
    set top/bottom view : 5/6
    set initial view    : 0
    ----------------------------------------------------
    MOUSE COMMANDS
    select a point      : left-click
    3d rotate           : left-drag
    pan                 : middle-drag or shift+left-drag
    zoom                : right-drag or wheel
"""


def _parse_color(
    value: str | tuple[float, float, float]
) -> tuple[float, float, float]:
    """Convert a color specification into a tuple[float, float, float]."""
    if isinstance(value, str):
        try:
            return PALETTE[value]
        except KeyError:
            raise ValueError(
                f"The specified color '{value}' is not recognized."
            )

    # Here, it's a sequence. Cast to tuple, check and return.
    value = tuple(value)  # type: ignore
    check_param("value", value, tuple, length=3, contents_type=float)
    if (
        (value[0] < 0.0)
        or (value[1] < 0.0)
        or (value[2] < 0.0)
        or (value[0] > 1.0)
        or (value[1] > 1.0)
        or (value[2] > 1.0)
    ):
        raise ValueError(
            f"The specified color '{value}' is invalid because each R, G, B "
            "value must be between 0.0 and 1.0."
        )
    return value


class Player:
    """
    A class that allows visualizing points and frames in 3D.

    `player = ktk.Player(parameters)` creates and launches an interactive
    Player instance. Once the window is open, press `h` to show a help
    overlay.

    All of the following parameters are also accessible as read/write
    properties, except the contents and the interconnections that are
    accessible using `get_contents`, `set_contents`, `get_interconnections`
    and `set_interconnections`.

    Parameters
    ----------
    *ts
        Contains the points and frames to visualize, where each data
        key is either a point position expressed as Nx4 array, or a frame
        expressed as a Nx4x4 array. Multiple TimeSeries can be provided.

    interconnections
        Optional. Each key corresponds to an interconnection between points,
        where each interconnection is a nested dict with the following keys:

        - "Links": list of lists strings, where each string is a point
          name. For example, to create a link that connects Point1 to Point2,
          and another link that spans Point3, Point4 and Point5::

              interconnections["Example"]["Links"] = [
                  ["Point1", "Point2"],
                  ["Point3", "Point4", "Point5"]
              ]

          Point names can include wildcards (*) either as a prefix or as a
          suffix. This is useful to apply a single set of interconnections to
          multiple bodies. For instance, if the Player's contents includes
          these points: [Body1_HipR, Body1_HipL, Body1_L5S1, Body2_HipR,
          Body2_HipL, Body2_L5S1], we could link L5S1 and both hips at once
          using::

              interconnections["Pelvis"]["Links"] = [
                  ["*_HipR", "*_HipL", "*_L5S1"]
              ]

        - "Color": character or tuple (RGB) that represents the color of the
          link. These two examples are equivalent::

              interconnections["Pelvis"]["Color"] = 'r'
              interconnections["Pelvis"]["Color"] = (1.0, 0.0, 0.0)

    current_index
        Optional. The current index being shown.

    current_time
        Optional. The current time being shown.

    playback_speed
        Optional. Speed multiplier. Set to 1.0 for normal speed, 1.5 to
        increase playback speed by 50%, etc.
    up
        Optional. Defines the ground plane by setting which axis is up. May be
        {"x", "y", "z", "-x", "-y", "-z"}. Default is "y".

    anterior
        Optional. Defines the anterior direction. May be
        {"x", "y", "z", "-x", "-y", "-z"}. Default is "x".

    zoom
        Optional. Camera zoom multipler.

    azimuth
        Optional. Camera azimuth in radians. If `anterior` is set, then an
        azimuth of 0 corresponds to the right sagittal plane, pi/2 to the
        front frontal plane, -pi/2 to the back frontal plane, etc.

    elevation
        Optional. Camera elevation in radians. Default is 0.2. If `up` is set,
        then a value of 0 corresponds to a purely horizontal view, pi/2 to the
        top transverse plane, -pi/2 to the bottom transverse plane, etc.

    perspective
        Optional. True to draw the scene using perspective, False to draw the
        scene orthogonally.

    pan
        Optional. Camera translation (panning). Default is (0.0, 0.0).

    target
        Optional. Camera target in meters. Default is (0.0, 0.0, 0.0).

    track
        Optional. False to keep the camera static, True to follow the last
        selected point when changing index. Default is False.

    default_point_color
        Optional. Default color for points that do not have a "Color"
        data_info. Can be a character or tuple (RGB) where each RGB color is
        between 0.0 and 1.0. Default is (0.8, 0.8, 0.8).

    point_size
        Optional. Point size as defined by Matplotlib marker size. Default is
        4.0.

    interconnection_width
        Optional. Width of the interconnections as defined by Matplotlib line
        width. Default is 1.5.

    frame_size
        Optional. Length of the frame axes in meters. Default is 0.1.

    frame_width
        Optional. Width of the frame axes as defined by Matplotlib line width.
        Default is 3.0.

    grid_size
        Optional. Length of one side of the grid in meters. Default is 10.0.

    grid_subdivision_size
        Optional. Length of one subdivision of the grid in meters. Default is
        1.0.

    grid_width
        Optional. Width of the grid lines as defined by Matplotlib line width.
        Default is 1.0.

    grid_origin
        Optional. Origin of the grid in meters. Default is (0.0, 0.0, 0.0).

    grid_color
        Optional. Color of the grid. Can be a character or tuple (RGB) where
        each RGB color is between 0.0 and 1.0. Default is (0.3, 0.3, 0.3).

    background_color
        Optional. Background color. Can be a character or tuple (RGB) where
        each RGB color is between 0.0 and 1.0. Default is (0.0, 0.0, 0.0).

    Note
    ----
    Matplotlib must be in interactive mode.

    """

    # %% Init and properties getters and setters

    # Internal variables - for mypy
    _being_constructed: bool
    _contents: TimeSeries
    _oriented_points: TimeSeries
    _oriented_frames: TimeSeries
    _oriented_target: tuple[float, float, float]
    _interconnections: dict[str, dict[str, Any]]
    _extended_interconnections: dict[str, dict[str, Any]]
    _colors: set[tuple[float, float, float]]  # A list of all point colors
    _selected_points: list[str]  # List of point names
    _last_selected_point: str
    _current_index: int
    _current_time: float
    _playback_speed: float
    _up: str
    _anterior: str
    _zoom: float
    _azimuth: float
    _elevation: float
    _perspective: bool
    _initial_elevation: float
    _initial_azimuth: float
    _initial_perspective: bool
    _pan: np.ndarray
    _target: np.ndarray
    _track: bool
    _default_point_color: tuple[float, float, float]
    _point_size: float
    _interconnection_width: float
    _frame_size: float
    _frame_width: float
    _grid_size: float
    _grid_subdivision_size: float
    _grid_width: float
    _grid_origin: np.ndarray
    _grid_color: tuple[float, float, float]
    _background_color: tuple[float, float, float]
    _title_text: str

    def __init__(
        self,
        *ts: TimeSeries,
        interconnections: dict[str, dict[str, Any]] = {},
        current_index: int = 0,
        current_time: float | None = None,
        playback_speed: float = 1.0,
        up: str = "y",
        anterior: str = "x",
        zoom: float = 1.0,
        azimuth: float = 0.0,
        elevation: float = 0.2,
        pan: tuple[float, float] = (0.0, 0.0),
        target: tuple[float, float, float] = (0.0, 0.0, 0.0),
        perspective: bool = True,
        track: bool = False,
        default_point_color: str | tuple[float, float, float] = (
            0.8,
            0.8,
            0.8,
        ),
        point_size: float = 4.0,
        interconnection_width: float = 1.5,
        frame_size: float = 0.1,
        frame_width: float = 3.0,
        grid_size: float = 10.0,
        grid_subdivision_size: float = 1.0,
        grid_width: float = 1.0,
        grid_origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        grid_color: str | tuple[float, float, float] = (
            0.3,
            0.3,
            0.3,
        ),
        background_color: str | tuple[float, float, float] = (
            0.0,
            0.0,
            0.0,
        ),
        **kwargs,  # Can be "inline_player=True", or older parameter names
    ):
        # Allow older parameter names
        if "segments" in kwargs and interconnections == {}:
            interconnections = kwargs["segments"]
        if "segment_width" in kwargs:
            interconnection_width = kwargs["segment_width"]
        if "current_frame" in kwargs:
            current_index = kwargs["current_frame"]
        if "translation" in kwargs:
            pan = kwargs["translation"]
        if "marker_radius" in kwargs:
            point_size = kwargs["marker_radius"]
        if "axis_length" in kwargs:
            frame_size = kwargs["axis_length"]
        if "axis_width" in kwargs:
            frame_width = kwargs["axis_width"]

        check_param("ts", ts, tuple, contents_type=TimeSeries)
        # The other parameters are checked by the property setters.

        # Warn if Matplotlib is not interactive
        check_interactive_backend()

        # Assign properties

        # Empty content for now. We set the final content after all
        # initializations.
        self._being_constructed = True

        self._contents = TimeSeries(time=[0])
        self._grid = np.array([])
        self._oriented_points = TimeSeries(time=self._contents.time)
        self._oriented_frames = TimeSeries(time=self._contents.time)
        self._oriented_target = (0.0, 0.0, 0.0)

        self._interconnections = interconnections  # Just to put stuff for now
        self._extended_interconnections = interconnections  # idem
        self._colors = set()  # idem
        self._selected_points = []
        self._last_selected_point = ""

        # Assign standard properties
        self.current_index = current_index

        self.playback_speed = playback_speed

        self.up = up  # We set directly because setters need both being set
        self.anterior = anterior
        self.zoom = zoom
        self.azimuth = azimuth
        self.elevation = elevation
        self.perspective = perspective
        self._initial_elevation = elevation
        self._initial_azimuth = azimuth
        self._initial_perspective = perspective
        self.pan = pan
        self.target = target
        self.track = track
        self.default_point_color = default_point_color
        self.point_size = point_size
        self.interconnection_width = interconnection_width
        self.frame_size = frame_size
        self.frame_width = frame_width
        self.grid_size = grid_size
        self.grid_width = grid_width
        self.grid_subdivision_size = grid_subdivision_size
        self.grid_origin = grid_origin
        self.grid_color = grid_color
        self.background_color = background_color
        self.title_text = ""

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
            "PanOnMousePress": (0.0, 0.0),
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
        self._being_constructed = False

        temp_ts = TimeSeries()
        for one_ts in ts:
            temp_ts.merge(one_ts, in_place=True)

        self.set_contents(temp_ts)
        self.set_interconnections(interconnections)
        self.grid_origin = grid_origin  # Refresh grid

        # Now that everything is loaded, we can set the current time if
        # needed.
        if current_time is not None:
            self.current_time = current_time

    @property
    def contents(self):
        """Use get_contents or set_contents instead."""
        raise AttributeError(
            "Please use Player.get_contents() and Player.set_contents() to "
            "read and write contents."
        )

    @contents.setter
    def contents(self, value):
        """Use get_contents or set_contents instead."""
        raise AttributeError(
            "Please use Player.get_contents() and Player.set_contents() to "
            "read and write contents."
        )

    @property
    def interconnections(self):
        """Use get_interconnections or set_interconnections instead."""
        raise AttributeError(
            "Please use Player.get_interconnections() and "
            "Player.set_interconnections() to read and write interconnections."
        )

    @interconnections.setter
    def interconnections(self, value):
        """Use get_interconnections or set_interconnections instead."""
        raise AttributeError(
            "Please use Player.get_interconnections() and "
            "Player.set_interconnections() to read and write interconnections."
        )

    @property
    def current_index(self) -> int:
        """Read/write current_index."""
        return self._current_index

    @current_index.setter
    def current_index(self, value: int):
        """Set current_index value."""
        try:
            self._current_index = value % len(self._contents.time)
        except AttributeError:  # No self._contents.time
            self._current_index = 0

        if not self._being_constructed:
            if self.track is True and self._oriented_points is not None:
                new_target = self._oriented_points.data[
                    self._last_selected_point
                ][self.current_index]
                if not np.isnan(np.sum(new_target)):
                    self.target = new_target

            self._fast_refresh()

    @property
    def current_time(self) -> float:
        """Read/write current_time."""
        return self._contents.time[self._current_index]

    @current_time.setter
    def current_time(self, value: float):
        """Set current_time value."""
        check_param("current_time", value, float)
        index = int(np.argmin(np.abs(self._contents.time - value)))
        self.current_index = index

    # Properties
    @property
    def playback_speed(self) -> float:
        """Read/write playback_speed."""
        return self._playback_speed

    @playback_speed.setter
    def playback_speed(self, value: float):
        """Set playback_speed value."""
        check_param("playback_speed", value, float)
        self._playback_speed = value

    @property
    def up(self) -> str:
        """Read/write up."""
        return self._up

    @up.setter
    def up(self, value: str):
        """Set up value."""
        check_param("up", value, str)
        if value in {"x", "y", "z", "-x", "-y", "-z"}:
            self._up = value
        else:
            raise ValueError(
                'up must be either "x", "y", "z", "-x", "-y", or "-z"}'
            )

        if not self._being_constructed:
            if self._up[-1] == self._anterior[-1]:
                # up and anterior cannot be the same axis.
                if value[-1] != "x":
                    self._anterior = "x"
                else:
                    self._anterior = "y"

            self._orient_contents()
            self._refresh()

    @property
    def anterior(self) -> str:
        """Read/write anterior."""
        return self._anterior

    @anterior.setter
    def anterior(self, value: str):
        """Set anterior value."""
        check_param("anterior", value, str)
        if value in {"x", "y", "z", "-x", "-y", "-z"}:
            self._anterior = value
        else:
            raise ValueError(
                'anterior must be either "x", "y", "z", "-x", "-y", or "-z"}'
            )

        if not self._being_constructed:
            if self._anterior[-1] == self._up[-1]:
                # up and anterior cannot be the same axis.
                if value[-1] != "y":
                    self._up = "y"
                else:
                    self._up = "z"

            self._orient_contents()
            self._refresh()

    @property
    def zoom(self) -> float:
        """Read/write zoom."""
        return self._zoom

    @zoom.setter
    def zoom(self, value: float):
        """Set zoom value."""
        check_param("zoom", value, float)
        self._zoom = value
        if not self._being_constructed:
            self._fast_refresh()

    @property
    def azimuth(self) -> float:
        """Read/write azimuth."""
        return self._azimuth

    @azimuth.setter
    def azimuth(self, value: float):
        """Set azimuth value."""
        check_param("azimuth", value, float)
        self._azimuth = value
        if not self._being_constructed:
            self._fast_refresh()

    @property
    def elevation(self) -> float:
        """Read/write elevation."""
        return self._elevation

    @elevation.setter
    def elevation(self, value: float):
        """Set elevation value."""
        check_param("elevation", value, float)
        self._elevation = value
        if not self._being_constructed:
            self._fast_refresh()

    @property
    def pan(self):
        """Read/write pan as (x, y)."""
        return (self._pan[0], self._pan[1])

    @pan.setter
    def pan(self, value):
        """Set pan value using (x, y) or (x, y, ...)."""
        value = tuple(value)
        check_param("pan", value, tuple, contents_type=float)
        self._pan = np.array(value)[0:2]
        if not self._being_constructed:
            self._fast_refresh()

    @property
    def target(self):
        """Read/write target as (x, y, z)."""
        return tuple(self._target)

    @target.setter
    def target(self, value):
        """Set target value using (x, y, z) or (x, y, z, 1.0)."""
        value = tuple(value)
        check_param("target", value, tuple, contents_type=float)
        self._target = np.array(value)[0:3]

        if not self._being_constructed:
            self._orient_contents()
            self._fast_refresh()

    @property
    def perspective(self) -> bool:
        """Read/write perspective."""
        return self._perspective

    @perspective.setter
    def perspective(self, value: bool):
        """Set perspective value."""
        check_param("perspective", value, bool)
        self._perspective = value
        if not self._being_constructed:
            self._fast_refresh()

    @property
    def track(self) -> bool:
        """Read/write track."""
        return self._track

    @track.setter
    def track(self, value: bool):
        """Set perspective value."""
        check_param("track", value, bool)
        self._track = value
        if not self._being_constructed:
            self._fast_refresh()

    @property
    def default_point_color(self):
        """Read/write default_point_color."""
        return self._default_point_color

    @default_point_color.setter
    def default_point_color(self, value):
        """Set default_point_color value."""
        self._default_point_color = _parse_color(value)
        if not self._being_constructed:
            self._refresh()

    @property
    def point_size(self) -> float:
        """Read/write point_size."""
        return self._point_size

    @point_size.setter
    def point_size(self, value: float):
        """Set point_size value."""
        check_param("point_size", value, float)
        self._point_size = value
        if not self._being_constructed:
            self._refresh()

    @property
    def interconnection_width(self) -> float:
        """Read/write interconnection_width."""
        return self._interconnection_width

    @interconnection_width.setter
    def interconnection_width(self, value: float):
        """Set interconnection_width value."""
        check_param("interconnection_width", value, float)
        self._interconnection_width = value
        if not self._being_constructed:
            self._refresh()

    @property
    def frame_size(self) -> float:
        """Read/write frame_size."""
        return self._frame_size

    @frame_size.setter
    def frame_size(self, value: float):
        """Set frame_size value."""
        check_param("frame_size", value, float)
        self._frame_size = value
        if not self._being_constructed:
            self._fast_refresh()

    @property
    def frame_width(self) -> float:
        """Read/write frame_width."""
        return self._frame_width

    @frame_width.setter
    def frame_width(self, value: float):
        """Set frame_width value."""
        check_param("frame_width", value, float)
        self._frame_width = value
        if not self._being_constructed:
            self._refresh()

    @property
    def grid_size(self) -> float:
        """Read/write grid_size."""
        return self._grid_size

    @grid_size.setter
    def grid_size(self, value: float):
        """Set grid_size value."""
        check_param("grid_size", value, float)
        self._grid_size = value
        if not self._being_constructed:
            self._update_grid()
            self._refresh()

    @property
    def grid_width(self) -> float:
        """Read/write grid_width."""
        return self._grid_width

    @grid_width.setter
    def grid_width(self, value: float):
        """Set grid_width value."""
        check_param("grid_width", value, float)
        self._grid_width = value
        if not self._being_constructed:
            self._update_grid()
            self._refresh()

    @property
    def grid_subdivision_size(self) -> float:
        """Read/write grid_subdivision_size."""
        return self._grid_subdivision_size

    @grid_subdivision_size.setter
    def grid_subdivision_size(self, value: float):
        """Set grid_subdivision_size value."""
        check_param("grid_subdivision_size", value, float)
        self._grid_subdivision_size = value
        if not self._being_constructed:
            self._update_grid()
            self._refresh()

    @property
    def grid_origin(self):
        """Read/write grid_origin."""
        return tuple(self._grid_origin)

    @grid_origin.setter
    def grid_origin(self, value):
        """Set grid_origin value."""
        value = tuple(value)
        check_param("grid_subdivision_size", value, tuple, contents_type=float)
        self._grid_origin = np.array(value)[0:3]
        if not self._being_constructed:
            self._update_grid()
            self._refresh()

    @property
    def grid_color(self):
        """Read/write grid_color."""
        return self._grid_color

    @grid_color.setter
    def grid_color(self, value):
        """Set grid_color value."""
        self._grid_color = _parse_color(value)
        if not self._being_constructed:
            self._update_grid()
            self._refresh()

    @property
    def background_color(self):
        """Read/write background_color."""
        return self._background_color

    @background_color.setter
    def background_color(self, value):
        """Set background_color value."""
        self._background_color = _parse_color(value)
        if not self._being_constructed:
            self._refresh()

    @property
    def title_text(self) -> str:
        """Read/write the text info on top of the figure."""
        return self._title_text

    @title_text.setter
    def title_text(self, value: str):
        """Set title_text."""
        check_param("title_text", value, str)
        self._title_text = value
        if not self._being_constructed:
            self._mpl_objects["Axes"].set_title(value, pad=-20)

    def __dir__(self):
        """Return directory."""
        return ["play", "pause", "set_view", "close", "to_image", "to_video"]

    def __str__(self) -> str:
        """Print a textual description of the Player properties."""
        return "ktk.Player with properties:\n" + _format_dict_entries(
            {
                "current_index": self.current_index,
                "current_time": self.current_time,
                "playback_speed": self.playback_speed,
                "up": self.up,
                "anterior": self.anterior,
                "zoom": self.zoom,
                "azimuth": self.azimuth,
                "elevation": self.elevation,
                "perspective": self.perspective,
                "pan": self.pan,
                "target": self.target,
                "track": self.track,
                "default_point_color": self.default_point_color,
                "point_size": self.point_size,
                "interconnection_width": self.interconnection_width,
                "frame_size": self.frame_size,
                "frame_width": self.frame_width,
                "grid_size": self.grid_size,
                "grid_width": self.grid_width,
                "grid_subdivision_size": self.grid_subdivision_size,
                "grid_origin": self.grid_origin,
                "grid_color": self.grid_color,
                "background_color": self.background_color,
                "title_text": self.title_text,
            }
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
            fig.canvas.toolbar.setVisible(False)  # type: ignore
        except AttributeError:
            pass

        plt.tight_layout()

        # Connect the callback functions
        fig.canvas.mpl_connect("pick_event", self._on_pick)
        fig.canvas.mpl_connect("key_press_event", self._on_key)
        fig.canvas.mpl_connect("key_release_event", self._on_release)
        fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        fig.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        fig.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        fig.canvas.mpl_connect("motion_notify_event", self._on_mouse_motion)

        # Create the animation
        anim = animation.FuncAnimation(
            fig,
            self._on_timer,  # type: ignore
            interval=33,
            cache_frame_data=False,
        )  # 30 ips

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
        check_param("value", value, TimeSeries)
        # First reset index to 0 to be sure that we won't end up out of bounds
        self._current_index = 0

        # Ensure that there is at least one sample so that the Player does not
        # crash and shows nothing instead.
        if len(value.time) > 0:
            self._contents = value.copy()
        else:
            self._contents = TimeSeries(time=[0])

        self._orient_contents()
        self._extend_interconnections()
        self._refresh()

    def get_interconnections(self) -> dict[str, dict[str, Any]]:
        """Get interconnections value."""
        return self._interconnections

    def set_interconnections(self, value: dict[str, dict[str, Any]]) -> None:
        """Set interconnections value."""
        check_param("value", value, dict, key_type=str, contents_type=dict)
        self._interconnections = deepcopy(value)
        self._extend_interconnections()
        self._refresh()

    def _extend_interconnections(self) -> None:
        """Update self._extended_interconnections. Does not refresh."""
        # Make a set of all patterns matched by the * in interconnection
        # point names.
        patterns = {"__NO_WILD_CARD_DEFAULT_PATTERN__"}
        keys = list(self._contents.data.keys())
        for body_name in self._interconnections:
            for i_link, link in enumerate(
                self._interconnections[body_name]["Links"]
            ):
                for i_point, point in enumerate(link):
                    if point.startswith("*") and point.endswith("*"):
                        raise ValueError(
                            f"Point {point} found in interconnections. "
                            "Only one wildcard can be used, either as a "
                            "prefix or as a suffix."
                        )
                    elif point.startswith("*"):
                        for key in keys:
                            if key.endswith(point[1:]):
                                patterns.add(
                                    key[: (len(key) - len(point) + 1)]
                                )
                    elif point.endswith("*"):
                        for key in keys:
                            if key.startswith(point[:-1]):
                                patterns.add(key[(len(point) - 1) :])
        # Extend every * to every pattern
        self._extended_interconnections = dict()
        for pattern in patterns:
            for body_name in self._interconnections:
                body_key = f"{pattern}{body_name}"
                self._extended_interconnections[body_key] = dict()
                self._extended_interconnections[body_key]["Color"] = (
                    self._interconnections[body_name]["Color"]
                )

                # Go through every link of this segment
                self._extended_interconnections[body_key]["Links"] = []
                for i_link, link in enumerate(
                    self._interconnections[body_name]["Links"]
                ):
                    self._extended_interconnections[body_key]["Links"].append(
                        [
                            s.replace("*", pattern)
                            for s in self._interconnections[body_name][
                                "Links"
                            ][i_link]
                        ]
                    )

    def _general_rotation(self) -> np.ndarray:
        """Return a 1x4x4 rotation matrix from up and anterior attributes."""
        # Create a frame based on these specs
        if self.up == "x":
            up = [[1, 0, 0, 0]]
        elif self.up == "y":
            up = [[0, 1, 0, 0]]
        elif self.up == "z":
            up = [[0, 0, 1, 0]]
        elif self.up == "-x":
            up = [[-1, 0, 0, 0]]
        elif self.up == "-y":
            up = [[0, -1, 0, 0]]
        elif self.up == "-z":
            up = [[0, 0, -1, 0]]
        else:
            raise ValueError(
                "up must be in {'x', 'y', 'z', '-x', '-y', '-z'}."
            )

        if self.anterior == "x":
            anterior = [[1, 0, 0, 0]]
        elif self.anterior == "y":
            anterior = [[0, 1, 0, 0]]
        elif self.anterior == "z":
            anterior = [[0, 0, 1, 0]]
        elif self.anterior == "-x":
            anterior = [[-1, 0, 0, 0]]
        elif self.anterior == "-y":
            anterior = [[0, -1, 0, 0]]
        elif self.anterior == "-z":
            anterior = [[0, 0, -1, 0]]
        else:
            raise ValueError(
                "anterior must be in {'x', 'y', 'z', '-x', '-y', '-z'}."
            )

        inverse_transform = geometry.create_frames(
            origin=[[0, 0, 0, 1]], x=anterior, xy=up
        )
        return geometry.inv(inverse_transform)

    def _orient_contents(self) -> None:
        """
        Update, self._oriented_points, _oriented_frames and _oriented_target

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

        rotation = self._general_rotation()

        # Orient points and frames
        for key in contents.data:
            if contents.data[key].shape[1:] == (4,):
                self._oriented_points.data[key] = (
                    geometry.get_global_coordinates(
                        contents.data[key], rotation
                    )
                )
            elif contents.data[key].shape[1:] == (4, 4):
                self._oriented_frames.data[key] = (
                    geometry.get_global_coordinates(
                        contents.data[key], rotation
                    )
                )

        oriented_target = geometry.get_global_coordinates(
            np.array(
                [[self._target[0], self._target[1], self._target[2], 1.0]]
            ),
            rotation,
        )[0, 0:3]
        self._oriented_target = (
            oriented_target[0],
            oriented_target[1],
            oriented_target[2],
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
                    [1, 0, 0, self.pan[0]],  # Pan
                    [0, 1, 0, self.pan[1]],
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
                    [1, 0, 0, -self._oriented_target[0]],
                    [0, 1, 0, -self._oriented_target[1]],
                    [0, 0, -1, self._oriented_target[2]],
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
            [
                [
                    self._grid_origin[0],
                    self._grid_origin[1],
                    self._grid_origin[2],
                    1.0,
                ]
            ],
            self._general_rotation(),
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

        for color in self._colors:
            # Reset unselected points
            points_data[(color, False)] = np.empty([n_points, 4])
            points_data[(color, False)][:] = np.nan
            # Reset selected points
            points_data[(color, True)] = np.empty([n_points, 4])
            points_data[(color, True)][:] = np.nan

        if n_points > 0:
            for i_point, point in enumerate(points.data):
                # Get this point's color
                if (
                    point in points.data_info
                    and "Color" in points.data_info[point]
                ):
                    color = _parse_color(points.data_info[point]["Color"])
                else:
                    color = self.default_point_color

                these_coordinates = points.data[point][self.current_index]
                interconnection_points[point] = these_coordinates

                # Assign to unselected(False) or selected(True) points_data
                if point in self._selected_points:
                    points_data[(color, True)][i_point] = these_coordinates
                else:
                    points_data[(color, False)][i_point] = these_coordinates

        # Update the points plot
        for color in self._colors:
            # Unselected points
            points_data[(color, False)] = self._project_to_camera(
                points_data[(color, False)]
            )
            self._mpl_objects["PointPlots"][(color, False)].set_data(
                points_data[(color, False)][:, 0],
                points_data[(color, False)][:, 1],
            )

            # Selected points
            points_data[(color, True)] = self._project_to_camera(
                points_data[(color, True)]
            )
            self._mpl_objects["PointPlots"][(color, True)].set_data(
                points_data[(color, True)][:, 0],
                points_data[(color, True)][:, 1],
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

    def _refresh(self):
        """
        Perform a full refresh of the Player.

        Normally, this function does not need to be called by the user. Use it
        if the Player is not refreshed as it should. You may report this need
        as a bug in the issue tracker:
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

        # ----------------------
        # Create the point plots
        # ----------------------
        # List all colors in contents
        self._colors = set()
        for key in self._contents.data:
            try:
                color = self._contents.data_info[key]["Color"]
            except KeyError:  # Default color
                color = self._default_point_color
            self._colors.add(_parse_color(color))

        # Create all required point plots
        for color in self._colors:
            # Unselected points
            self._mpl_objects["PointPlots"][
                (color, False)
            ] = self._mpl_objects["Axes"].plot(
                np.nan,
                np.nan,
                ".",
                c=color,
                markersize=self._point_size,
                pickradius=1.1 * self._point_size,
                picker=True,
            )[
                0
            ]

            # Selected points
            self._mpl_objects["PointPlots"][(color, True)] = self._mpl_objects[
                "Axes"
            ].plot(
                np.nan,
                np.nan,
                ".",
                c=color,
                markersize=3 * self._point_size,
            )[
                0
            ]

        # Add the title
        title_obj = plt.title("", fontfamily="monospace")
        plt.setp(title_obj, color=[0, 1, 0])  # Set a green title

        self._fast_refresh()  # Draw everything once

        # Set limits once it's drawn
        self._mpl_objects["Axes"].set_xlim([-1.5, 1.5])
        self._mpl_objects["Axes"].set_ylim([-1.0, 1.0])

    def _set_new_target(self, target: ArrayLike) -> None:
        """Set new target and adapts pan and zoom consequently."""
        # Save the current view
        if np.sum(np.isnan(target)) > 0:
            return
        initial_pan = deepcopy(self.pan)
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
        initial_projected_points[initial_projected_points[:, 0] < -1.5] = (
            np.nan
        )
        initial_projected_points[initial_projected_points[:, 0] > 1.5] = np.nan
        initial_projected_points[initial_projected_points[:, 1] < -1.0] = (
            np.nan
        )
        initial_projected_points[initial_projected_points[:, 1] > 1.0] = np.nan

        def error_function(input):
            self._pan = input[0:2]
            self._zoom = input[2]
            new_projected_points = self._project_to_camera(points)
            error = np.nanmean(
                (initial_projected_points - new_projected_points) ** 2
            )
            return error

        # Set the new target
        self._target = np.array(target)
        self._orient_contents()

        # Try to find a camera pan/zoom so that the view is similar
        res = optim.minimize(error_function, np.hstack((self.pan, self.zoom)))
        if res.success is False:
            self.pan = initial_pan
            self.zoom = initial_zoom
            self.target = initial_target

        self._fast_refresh()

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
            current_system_time = time.time()
            current_index = self.current_index
            self.current_time += self.playback_speed * (
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
            self._mpl_objects["Anim"].event_source.stop()

    def _on_pick(self, event):  # pragma: no cover
        """Implement callback for point selection."""
        if event.mouseevent.button == 1:
            index = event.ind
            selected_point = list(self._oriented_points.data.keys())[index[0]]
            self.title_text = selected_point

            # Mark selected
            self._selected_points = [selected_point]

            # Set as new target
            self._last_selected_point = selected_point
            self._set_new_target(
                self._contents.data[selected_point][self.current_index]
            )

            self._fast_refresh()

    def _on_key(self, event):  # pragma: no cover
        """Implement callback for keyboard key pressed."""
        if event.key == " ":
            if self._running is False:
                self.play()
            else:
                self.pause()

        elif event.key == "left":
            self.current_index -= 1

        elif event.key == "shift+left":
            self.current_time -= 1

        elif event.key == "right":
            self.current_index += 1

        elif event.key == "shift+right":
            self.current_time += 1

        elif event.key == "-":
            self.playback_speed /= 2
            self.title_text = f"Playback set to {self.playback_speed}x"

        elif event.key == "+":
            self.playback_speed *= 2
            self.title_text = f"Playback set to {self.playback_speed}x"

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
                self.title_text = "Camera set to perspective"
            else:
                self.title_text = "Camera set to orthogonal"

        elif event.key == "t":
            self.track = not self.track
            if self.track is True:
                self.title_text = "Point tracking activated"
            else:
                self.title_text = "Point tracking deactivated"

        elif event.key == "1":
            self.set_view("back")
            self.title_text = "Back view, orthogonal"
        elif event.key == "2":
            self.set_view("front")
            self.title_text = "Front view, orthogonal"
        elif event.key == "3":
            self.set_view("left")
            self.title_text = "Left view, orthogonal"
        elif event.key == "4":
            self.set_view("right")
            self.title_text = "Right view, orthogonal"
        elif event.key == "5":
            self.set_view("top")
            self.title_text = "Top view, orthogonal"
        elif event.key == "6":
            self.set_view("bottom")
            self.title_text = "Bottom view, orthogonal"
        elif event.key == "0":
            self.set_view("initial")
            self.title_text = "Initial view"

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
        self._state["PanOnMousePress"] = self.pan
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
        self._fast_refresh()

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
            self.pan = (
                self._state["PanOnMousePress"][0]
                + (event.x - self._state["MousePositionOnPress"][0])
                / (100 * self.zoom),
                self._state["PanOnMousePress"][1]
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

    def _to_animation(self):
        """
        Create a matplotlib FuncAnimation for displaying in Jupyter notebooks.

        This also closes the figure so that Jupyter does not show both
        the animation and the figure.

        Parameters
        ----------
        No parameter.

        Returns
        -------
        A FuncAnimation to be displayed by Jupyter notebook.

        """
        try:
            from IPython.display import Video
        except ModuleNotFoundError:
            raise RuntimeError(
                "This function must be run in an IPython session."
            )

        self._mpl_objects["Figure"].set_size_inches(6, 4.5)  # Half size
        self._mpl_objects["Figure"].tight_layout()
        self.to_video("temp.mp4", show_progress_bar=False)
        return Video(
            "temp.mp4", embed=True, html_attributes="controls loop autoplay"
        )

    # %% Public methods
    def play(self) -> None:
        """Start the animation."""
        self._state["SystemTimeOnLastUpdate"] = time.time()
        self._running = True
        self._mpl_objects["Anim"].event_source.start()

    def pause(self) -> None:
        """Pause the animation."""
        self._running = False
        self._mpl_objects["Anim"].event_source.stop()

    def set_view(self, plane: str) -> None:
        """
        Set the current view to an orthogonal view in a given plane.

        Ensure that the player's `up` and `anterior` properties are set to the
        correct axes beforehand. By default, `up` is "y" and `anterior` is "x".

        Parameters
        ----------
        plane
            Can be either "front", "back", "right", "left", "top", "bottom" or
            "initial". In the latter case, the view is reset to the initial
            view at Player creation.

        """
        check_param("plane", plane, str)

        if plane.lower() == "initial":
            self.elevation = self._initial_elevation
            self.azimuth = self._initial_azimuth
            self.perspective = self._initial_perspective
            return

        # Set a "from rotation" matrix following x anterior, y up and z right
        if plane.lower() == "front":
            from_rot = geometry.create_transforms(
                "YXZ", [[90, 0, 0]], degrees=True
            )
            self.perspective = False
        elif plane.lower() == "back":
            from_rot = geometry.create_transforms(
                "YXZ", [[-90, 0, 0]], degrees=True
            )
            self.perspective = False
        elif plane.lower() == "top":
            from_rot = geometry.create_transforms(
                "YXZ", [[0, 90, 0]], degrees=True
            )
            self.perspective = False
        elif plane.lower() == "bottom":
            from_rot = geometry.create_transforms(
                "YXZ", [[0, -90, 0]], degrees=True
            )
            self.perspective = False
        elif plane.lower() == "right":
            from_rot = geometry.create_transforms(
                "YXZ", [[0, 0, 0]], degrees=True
            )
            self.perspective = False
        elif plane.lower() == "left":
            from_rot = geometry.create_transforms(
                "YXZ", [[180, 0, 0]], degrees=True
            )
            self.perspective = False
        else:
            raise ValueError(
                "Parameter plane can be either "
                '"front", "back", "right", "left", "top" or "bottom".'
            )

        # Ignore gimbal lock warnings, gimbal locks are ok since SciPy
        # behaviour is well documented in those case (3rd angle set to 0).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            angles = geometry.get_angles(from_rot, "YXZ")

        self.elevation = angles[0, 1]
        self.azimuth = angles[0, 0]

    def close(self) -> None:
        """Close the Player and its associated window."""
        plt.close(self._mpl_objects["Figure"])
        self._mpl_objects = {}

    def to_image(self, filename: str) -> None:
        """
        Save the current view to an image file.

        Any format supported by Matplotlib can be used.

        Parameters
        ----------
        filename
            Name of the image file to save (e.g., "file.png", "file.jpeg",
            "file.pdf", "file.svg", "file.tiff")

        Returns
        -------
        None

        """
        check_param("filename", filename, str)

        self._mpl_objects["Figure"].savefig(filename)

    def to_video(
        self,
        filename: str,
        *,
        fps: int | None = None,
        downsample: int = 1,
        show_progress_bar: bool = True,
    ) -> None:
        """
        Save the current view to an MP4 video file.

        Parameters
        ----------
        filename
            Name of the video file to save.
        fps
            Optional. Frames per second of the output video. Default is None,
            which means that fps matches the current playback speed of the
            Player. This attribute does not affect the number of images in
            the output video; it only affects the playback speed of the output
            video.
        downsample
            Optional. Use it to reduce the file size on acquisitions at high
            sample rates. Default is 1, which means that the video is not
            downsampled. In this case, each index is exported as one frame of
            the output video. A value of 2 divides the number of frames by 2,
            which means that every other index is skipped. A value of 3 divides
            the number of frames by 3, etc.
        show_progress_bar
            Optional. True to show a progress bar while creating the video
            file.

        Returns
        -------
        None

        """
        check_param("filename", filename, str)
        check_param("fps", fps, (int, None))
        check_param("downsample", downsample, int)
        check_param("show_progress_bar", show_progress_bar, bool)

        if downsample < 1:
            raise ValueError(
                "Parameter downsample must be stricly higher than 0."
            )

        n_samples = int(len(self._contents.time) / downsample)

        # We create a specific animation and callback, since all processing
        # will be done offline. We set a very long delay between frames but
        # this is just so that the animation didn't advance by itself by the
        # time recording has started.
        def advance(args):
            self.current_index = args * downsample
            self.title_text = (
                f"{self.current_index}/{(n_samples - 1) * downsample}: "
                f"{self.current_time:.3f} s."
            )

        anim = animation.FuncAnimation(
            self._mpl_objects["Figure"],
            advance,  # type: ignore
            frames=n_samples,
            interval=1e6,
        )  # 30 ips

        if fps is None:
            fps = int(
                self._contents.get_sample_rate()
                * self.playback_speed
                / downsample
            )
            if np.isnan(fps):
                fps = 30
        writervideo = animation.FFMpegWriter(fps=fps)

        self.pause()
        self.current_index = 0

        if show_progress_bar:
            progress_bar = tqdm(n_samples - 1)
            update_progress_bar = lambda i, n: progress_bar.update(1)
        else:
            update_progress_bar = lambda i, n: None

        anim.save(
            filename, writer=writervideo, progress_callback=update_progress_bar
        )
        anim.event_source.stop()

        if show_progress_bar:
            progress_bar.close()

        self.title_text = ""
        self.current_index = 0
