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

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


from kineticstoolkit.timeseries import TimeSeries
from kineticstoolkit.decorators import directory

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import animation
import numpy as np
from numpy import sin, cos
import time
import copy
from typing import Union, Sequence, Dict, Any, Optional

# To fit the new viewpoint on selecting a new marker
import scipy.optimize as optim


class Player:
    """
    A class that allows visualizing markers and rigid bodies in 3D.

    player = ktk.Player(parameters) creates and launches a Player
    instance.

    Parameters
    ----------
    *ts
        Contains the markers and rigid bodies to visualize, where each data
        key is either a marker position expressed as Nx4 array (N=time),
        or a rigid body pose expressed as a Nx4x4 array. Multiple TimeSeries
        can be provided, e.g., ktk.Player(markers, rigid_bodies)

    interconnections
        Optional. Each key corresponds to an inerconnection between markers,
        where one interconnection is another dict with the following keys:

        - Links: list of list of 2 str, where each str is a marker
          name. For example, to link Marker1 to Marker2 and
          Marker1 to Marker3, Links would be:
          [['Marker1', 'Marker2'], ['Marker1', 'Marker3']]

        - Color: character or tuple that represents the color of the
          link. Color must be a valid value for matplotlib's
          plots.

    current_index
        Optional. Sets the inital index number to show.

    marker_radius
        Optional. Sets the marker radius as defined by matplotlib.

    axis_length
        Optional. Sets the rigid body size in meters.

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

    """

    def __init__(self,
                 *ts: TimeSeries,
                 interconnections: Dict[str, Dict[str, Any]] = {},
                 current_index: int = 0,
                 marker_radius: float = 0.008,
                 axis_length: float = 0.1,
                 axis_width: float = 3.0,
                 interconnection_width: float = 1.5,
                 zoom: float = 1.0,
                 azimuth: float = 0.0,
                 elevation: float = 0.2,
                 translation: Union[Sequence[float], np.ndarray] = (0.0, 0.0),
                 target: Union[Sequence[float], np.ndarray] = (0.0, 0.0, 0.0),
                 track: bool = False,
                 perspective: bool = True,
                 **kwargs):

        # Allow older variable names
        if 'segments' in kwargs and interconnections == {}:
            interconnections = kwargs['segments']
        if 'segment_width' in kwargs:
            interconnection_width = kwargs['segment_width']
        if 'current_frame' in kwargs:
            current_index = kwargs['current_frame']

        # ---------------------------------------------------------------
        # Set self.n_indexes and self.time, and verify that we have at least
        # markers or rigid bodies.
        self.time = ts[0].time
        self.n_indexes = len(ts[0].time)

        # ---------------------------------------------------------------
        # Assign the markers and rigid bodies
        self.markers = TimeSeries()
        self.rigid_bodies = TimeSeries()

        for one_ts in ts:
            for key in one_ts.data:
                if one_ts.data[key].shape[1:] == (4,):
                    self.markers.data[key] = one_ts.data[key]
                    if key in one_ts.data_info:
                        self.markers.data_info[key] = one_ts.data_info[key]

                elif one_ts.data[key].shape[1:] == (4, 4):

                    # Add the frames
                    self.rigid_bodies.data[key] = one_ts.data[key]
                    if key in one_ts.data_info:
                        self.rigid_bodies.data_info[key] = \
                            one_ts.data_info[key]

                    # Add the frame origins as "markers"
                    key_label = f'[{key}] origin'
                    self.markers.data[key_label] = one_ts.data[key][:, :, 3]
                    if key in one_ts.data_info:
                        self.markers.data_info[key_label] = (
                            one_ts.data_info[key]
                        )

                else:
                    raise ValueError('TimeSeries data must be of shape Nx4 '
                                     '(markers) or Nx4x4 (rigid bodies)')
        self._select_none()
        self.last_selected_marker = ''

        # Add the origin to the rigid bodies
        self.rigid_bodies.data['Global'] = np.repeat(
            np.eye(4, 4)[np.newaxis, :, :], self.n_indexes, axis=0)

        # ---------------------------------------------------------------
        # Assign the interconnections
        self.interconnections = interconnections

        # ---------------------------------------------------------------
        # Other initalizations
        self.current_index = current_index
        self.marker_radius = marker_radius
        self.axis_length = axis_length
        self.axis_width = axis_width
        self.interconnection_width = interconnection_width
        self.zoom = zoom
        self.azimuth = azimuth
        self.elevation = elevation
        self.target = target
        self.track = track
        self.translation = translation
        self.perspective = perspective
        self.playback_speed = 1.0  # 0.0 to show all samples
        #  self.anim = None  # Will initialize in _create_figure

        self.objects = dict()  # type: Dict[str, Any]
        self._colors = ['r', 'g', 'b', 'y', 'c', 'm', 'w']
        self.objects['PlotMarkers'] = dict()
        for color in self._colors:
            self.objects['PlotMarkers'][color] = None  # Not selected
            self.objects['PlotMarkers'][color + 's'] = None  # Selected
        self.objects['PlotRigidBodiesX'] = None
        self.objects['PlotRigidBodiesY'] = None
        self.objects['PlotRigidBodiesZ'] = None
        self.objects['PlotGroundPlane'] = None
        self.objects['PlotInterconnections'] = dict()

        self.objects['Figure'] = None
        self.objects['Axes'] = None
        self.objects['Help'] = None

        self.state = dict()  # type: Dict[str, Any]
        self.state['ShiftPressed'] = False
        self.state['MouseLeftPressed'] = False
        self.state['MouseMiddlePressed'] = False
        self.state['MouseRightPressed'] = False
        self.state['MousePositionOnPress'] = (0.0, 0.0)
        self.state['MousePositionOnMiddlePress'] = (0.0, 0.0)
        self.state['MousePositionOnRightPress'] = (0.0, 0.0)
        self.state['TranslationOnMousePress'] = (0.0, 0.0)
        self.state['AzimutOnMousePress'] = 0.0
        self.state['ElevationOnMousePress'] = 0.0
        self.state['SelfTimeOnPlay'] = self.time[0]
        self.state['SystemTimeOnLastUpdate'] = time.time()

        self._help_text = '''
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
            '''

        self._create_figure()
        self._create_interconnections()
        self._create_markers()
        self._create_ground_plane()
        self._first_refresh()

    def __dir__(self):
        return directory(Player.__dict__)

    def _create_figure(self) -> None:
        """Create the player's figure."""
        # Create the figure and axes
        self.objects['Figure'], self.objects['Axes'] = plt.subplots(
            num=None,
            figsize=(12, 9),
            facecolor='k',
            edgecolor='w')

        # Remove the toolbar
        try:  # Try, setVisible method not always there
            self.objects['Figure'].canvas.toolbar.setVisible(False)
        except AttributeError:
            pass

        plt.tight_layout()

        # Add the title
        title_obj = plt.title('Player')
        plt.setp(title_obj, color=[0, 1, 0])  # Set a green title

        # Remove the background for faster plotting
        self.objects['Axes'].set_axis_off()

        # Add the animation timer
        self.anim = animation.FuncAnimation(self.objects['Figure'],
                                            self._on_timer,
                                            interval=33)  # 30 ips
        self.running = False

        # Connect the callback functions
        self.objects['Figure'].canvas.mpl_connect(
            'pick_event', self._on_pick)
        self.objects['Figure'].canvas.mpl_connect(
            'key_press_event', self._on_key)
        self.objects['Figure'].canvas.mpl_connect(
            'key_release_event', self._on_release)
        self.objects['Figure'].canvas.mpl_connect(
            'scroll_event', self._on_scroll)
        self.objects['Figure'].canvas.mpl_connect(
            'button_press_event', self._on_mouse_press)
        self.objects['Figure'].canvas.mpl_connect(
            'button_release_event', self._on_mouse_release)
        self.objects['Figure'].canvas.mpl_connect(
            'motion_notify_event', self._on_mouse_motion)

    def _create_interconnections(self) -> None:
        """Create the interconnections plots in the player's figure."""
        if self.interconnections is not None:
            for interconnection in self.interconnections:
                self.objects['PlotInterconnections'][interconnection] = \
                    self.objects['Axes'].plot(
                        np.nan, np.nan, '-',
                        c=self.interconnections[interconnection]['Color'],
                        linewidth=self.interconnection_width)[0]

    def _create_markers(self) -> None:
        """Create the markers plots in the player's figure."""
        colors = {
            'r': [1, 0, 0],
            'g': [0, 1, 0],
            'b': [0.3, 0.3, 1],
            'y': [1, 1, 0],
            'm': [1, 0, 1],
            'c': [0, 1, 1],
            'w': [0.8, 0.8, 0.8]}

        for color in self._colors:
            self.objects['PlotMarkers'][color] = self.objects['Axes'].plot(
                np.nan, np.nan, '.',
                c=colors[color], markersize=4, pickradius=5, picker=True)[0]
        for color in self._colors:
            self.objects['PlotMarkers'][color + 's'] = \
                self.objects['Axes'].plot(
                    np.nan, np.nan, '.',
                    c=colors[color], markersize=12)[0]

    def _create_ground_plane(self) -> None:
        # Create the ground plane matrix
        gp_size = 30  # blocks
        gp_div = 4  # blocks per meter
        gp_x = np.block([
            np.tile([-gp_size / gp_div, gp_size / gp_div, np.nan], gp_size),
            np.repeat(
                np.linspace(-gp_size / gp_div, gp_size / gp_div, gp_size), 3)])
        gp_y = np.zeros(6 * gp_size)
        gp_z = np.block([
            np.repeat(
                np.linspace(-gp_size / gp_div, gp_size / gp_div, gp_size), 3),
            np.tile([-gp_size / gp_div, gp_size / gp_div, np.nan], gp_size)])
        gp_1 = np.ones(6 * gp_size)
        self._ground_plane = np.hstack([gp_x[:, np.newaxis],
                                        gp_y[:, np.newaxis],
                                        gp_z[:, np.newaxis],
                                        gp_1[:, np.newaxis]])

    def _first_refresh(self) -> None:
        """Draw the stuff and set the axis size."""
        self._update_plots()
        plt.axis([-1.5, 1.5, -1, 1])

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

        R = (np.array([[2 * self.zoom, 0, 0, 0],
                       [0, 2 * self.zoom, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]]) @
             np.array([[1, 0, 0, self.translation[0]],  # Pan
                       [0, 1, 0, self.translation[1]],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]]) @
             np.array([[1, 0, 0, 0],
                       [0, cos(-self.elevation), sin(self.elevation), 0],
                       [0, sin(-self.elevation), cos(-self.elevation), 0],
                       [0, 0, 0, 1]]) @
             np.array([[cos(-self.azimuth), 0, sin(self.azimuth), 0],
                       [0, 1, 0, 0],
                       [sin(-self.azimuth), 0, cos(-self.azimuth), 0],
                       [0, 0, 0, 1]]) @
             np.array([[1, 0, 0, -self.target[0]],  # Rotate around target
                       [0, 1, 0, -self.target[1]],
                       [0, 0, -1, self.target[2]],
                       [0, 0, 0, 1]]))

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
            with np.errstate(invalid='ignore'):
                to_remove = (denom < 1E-12)
            rotated_points_3d[to_remove, 0] = np.nan
            rotated_points_3d[to_remove, 1] = np.nan
        else:
            # Scale to match the point of view
            rotated_points_3d /= 5

        # Return only x and y
        return rotated_points_3d[:, 0:2]

    def _update_markers_and_interconnections(self) -> None:
        # Get a Nx4 matrices of every marker at the current index
        markers = self.markers
        if markers is None:
            return
        else:
            n_markers = len(markers.data)

        markers_data = dict()  # Used to draw the markers with different colors
        interconnection_markers = dict()  # Used to draw the interconnections

        for color in self._colors:
            markers_data[color] = np.empty([n_markers, 4])
            markers_data[color][:] = np.nan

            markers_data[color + 's'] = np.empty([n_markers, 4])
            markers_data[color + 's'][:] = np.nan

        if n_markers > 0:
            for i_marker, marker in enumerate(markers.data):

                # Get this marker's color
                if (marker in markers.data_info and
                        'Color' in markers.data_info[marker]):
                    color = markers.data_info[marker]['Color']
                else:
                    color = 'w'

                these_coordinates = markers.data[marker][self.current_index]
                markers_data[color][i_marker] = these_coordinates
                interconnection_markers[marker] = these_coordinates

        # Update the markers plot
        for color in self._colors:
            # Unselected markers
            markers_data[color] = self._get_projection(
                markers_data[color])
            self.objects['PlotMarkers'][color].set_data(
                markers_data[color][:, 0],
                markers_data[color][:, 1])

            # Selected markers
            markers_data[color + 's'] = self._get_projection(
                markers_data[color + 's'])
            self.objects['PlotMarkers'][color + 's'].set_data(
                markers_data[color + 's'][:, 0],
                markers_data[color + 's'][:, 1])

        # Draw the interconnections
        if self.interconnections is not None:
            for interconnection in self.interconnections:
                n_links = len(self.interconnections[interconnection]['Links'])
                coordinates = np.empty((3 * n_links, 4))
                coordinates[:] = np.nan
                for i_link in range(n_links):
                    marker1 = self.interconnections[
                        interconnection]['Links'][i_link][0]
                    marker2 = self.interconnections[
                        interconnection]['Links'][i_link][1]
                    if marker1 in interconnection_markers:
                        coordinates[3 * i_link] = (
                            interconnection_markers[marker1]
                        )
                    else:
                        coordinates[3 * i_link] = np.nan

                    if marker2 in interconnection_markers:
                        coordinates[3 * i_link + 1] = (
                            interconnection_markers[marker2]
                        )
                    else:
                        coordinates[3 * i_link + 1] = np.nan

                coordinates = self._get_projection(coordinates)

                self.objects['PlotInterconnections'][interconnection].set_data(
                    coordinates[:, 0], coordinates[:, 1])

    def _update_plots(self) -> None:
        """Update the plots, or draw it if not plot has been drawn before."""
        self._update_markers_and_interconnections()

        # Get three (3N)x4 matrices (for x, y and z lines) for the rigid bodies
        # at the current index
        rigid_bodies = self.rigid_bodies
        n_rigid_bodies = len(rigid_bodies.data)
        rbx_data = np.empty([n_rigid_bodies * 3, 4])
        rby_data = np.empty([n_rigid_bodies * 3, 4])
        rbz_data = np.empty([n_rigid_bodies * 3, 4])

        for i_rigid_body, rigid_body in enumerate(rigid_bodies.data):
            # Origin
            rbx_data[i_rigid_body * 3] = (
                rigid_bodies.data[rigid_body][self.current_index, :, 3])
            rby_data[i_rigid_body * 3] = (
                rigid_bodies.data[rigid_body][self.current_index, :, 3])
            rbz_data[i_rigid_body * 3] = (
                rigid_bodies.data[rigid_body][self.current_index, :, 3])
            # Direction
            rbx_data[i_rigid_body * 3 + 1] = (
                rigid_bodies.data[rigid_body][self.current_index] @
                np.array([self.axis_length, 0, 0, 1]))
            rby_data[i_rigid_body * 3 + 1] = (
                rigid_bodies.data[rigid_body][self.current_index] @
                np.array([0, self.axis_length, 0, 1]))
            rbz_data[i_rigid_body * 3 + 1] = (
                rigid_bodies.data[rigid_body][self.current_index] @
                np.array([0, 0, self.axis_length, 1]))
            # NaN to cut the line between the different rigid bodies
            rbx_data[i_rigid_body * 3 + 2] = np.repeat(np.nan, 4)
            rby_data[i_rigid_body * 3 + 2] = np.repeat(np.nan, 4)
            rbz_data[i_rigid_body * 3 + 2] = np.repeat(np.nan, 4)

        # Update the ground plane
        gp = self._get_projection(self._ground_plane)
        if self.objects['PlotGroundPlane'] is None:  # Create the plot
            self.objects['PlotGroundPlane'] = self.objects['Axes'].plot(
                gp[:, 0], gp[:, 1], c=[0.3, 0.3, 0.3], linewidth=1)[0]
        else:  # Update the plot
            self.objects['PlotGroundPlane'].set_data(gp[:, 0], gp[:, 1])

        # Create or update the rigid bodies plot
        rbx_data = self._get_projection(rbx_data)
        rby_data = self._get_projection(rby_data)
        rbz_data = self._get_projection(rbz_data)
        if self.objects['PlotRigidBodiesX'] is None:  # Create the plot
            self.objects['PlotRigidBodiesX'] = self.objects['Axes'].plot(
                rbx_data[:, 0], rbx_data[:, 1], c='r',
                linewidth=self.axis_width)[0]
            self.objects['PlotRigidBodiesY'] = self.objects['Axes'].plot(
                rby_data[:, 0], rby_data[:, 1], c='g',
                linewidth=self.axis_width)[0]
            self.objects['PlotRigidBodiesZ'] = self.objects['Axes'].plot(
                rbz_data[:, 0], rbz_data[:, 1], c='b',
                linewidth=self.axis_width)[0]
        else:  # Update the plot
            self.objects['PlotRigidBodiesX'].set_data(
                rbx_data[:, 0], rbx_data[:, 1])
            self.objects['PlotRigidBodiesY'].set_data(
                rby_data[:, 0], rby_data[:, 1])
            self.objects['PlotRigidBodiesZ'].set_data(
                rbz_data[:, 0], rbz_data[:, 1])

        # Update the window title
        try:
            self.objects['Figure'].canvas.manager.set_window_title(
                f'{self.current_index}/{self.n_indexes}: ' +
                '%2.2f s.' % self.time[self.current_index])
        except AttributeError:
            pass

        self.objects['Figure'].canvas.draw()

    def _set_new_target(self,
                        target: Union[Sequence[float], np.ndarray]) -> None:
        """Set new target and adapts translation and zoom consequently."""
        if np.sum(np.isnan(target)) > 0:
            return
        initial_translation = copy.deepcopy(self.translation)
        initial_zoom = copy.deepcopy(self.zoom)
        initial_target = copy.deepcopy(self.target)

        n_markers = len(self.markers.data)
        markers = np.empty((n_markers, 4))
        for i_marker, marker in enumerate(self.markers.data):
            markers[i_marker] = self.markers.data[marker][self.current_index]

        initial_projected_markers = self._get_projection(markers)
        # Do not consider markers that are not in the screen
        initial_projected_markers[
            initial_projected_markers[:, 0] < -1.5] = np.nan
        initial_projected_markers[
            initial_projected_markers[:, 0] > 1.5] = np.nan
        initial_projected_markers[
            initial_projected_markers[:, 1] < -1.0] = np.nan
        initial_projected_markers[
            initial_projected_markers[:, 1] > 1.0] = np.nan
        self.target = target

        def error_function(input):
            self.translation = input[0:2]
            self.zoom = input[2]
            new_projected_markers = self._get_projection(markers)
            error = np.nanmean((initial_projected_markers -
                                new_projected_markers) ** 2)
            return error

        res = optim.minimize(error_function, np.hstack((self.translation,
                                                        self.zoom)))
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

        if (self.track is True and
                self.markers is not None):
            new_target = self.markers.data[
                self.last_selected_marker][self.current_index]
            if not np.isnan(np.sum(new_target)):
                self.target = new_target

    def _set_time(self, time: float) -> None:
        """Set current index to a given time and update plots."""
        index = int(np.argmin(np.abs(self.time - time)))
        self._set_index(index)

    def _select_none(self) -> None:
        """Deselect every markers."""
        if self.markers is not None:
            for marker in self.markers.data:
                try:
                    # Keep 1st character, remove the possible 's'
                    self.markers.data_info[marker]['Color'] = \
                        self.markers.data_info[marker]['Color'][0]
                except KeyError:
                    self.markers = self.markers.add_data_info(
                        marker, 'Color', 'w')

    def close(self) -> None:
        """Close the Player and its associated window."""
        plt.close(self.objects['Figure'])
        self.objects = {}

    # ------------------------------------
    # Callbacks
    def _on_close(self, _) -> None:  # pragma: no cover
        # Release all references to objects
        self.close()

    def _on_timer(self, _) -> None:  # pragma: no cover
        if self.running is True:
            # We check self.running because the garbage collector may take time
            # before deleting the animation timer, and unreferencing the
            # animation timer is the recommended way to deactivate a timer.
            """Callback for the animation timer object."""
            current_index = self.current_index
            self._set_time(self.time[self.current_index] +
                           self.playback_speed * (
                           time.time() -
                           self.state['SystemTimeOnLastUpdate']))
            if current_index == self.current_index:
                # The time wasn't enough to advance a frame. Articifically
                # advance a frame.
                self._set_index(self.current_index + 1)
            self.state['SystemTimeOnLastUpdate'] = time.time()

            self._update_plots()
        else:
            self.anim.event_source.stop()

    def _on_pick(self, event):  # pragma: no cover
        """Callback for marker selection."""
        if event.mouseevent.button == 1:
            index = event.ind
            selected_marker = list(self.markers.data.keys())[index[0]]
            self.objects['Axes'].set_title(selected_marker)

            # Mark selected
            self._select_none()
            self.markers.data_info[selected_marker]['Color'] = \
                self.markers.data_info[selected_marker]['Color'][0] + 's'

            # Set as new target
            self.last_selected_marker = selected_marker
            self._set_new_target(
                self.markers.data[selected_marker][self.current_index])
            marker_position = self.markers.data[selected_marker][
                self.current_index]

            self._update_plots()

    def _on_key(self, event):  # pragma: no cover
        """Callback for keyboard key pressed."""
        if event.key == ' ':
            if self.running is False:
                self.state['SystemTimeOnLastUpdate'] = time.time()
                self.state['SelfTimeOnPlay'] = self.time[self.current_index]
                self.running = True
                self.anim.event_source.start()
            else:
                self.running = False
                self.anim.event_source.stop()

        elif event.key == 'left':
            self._set_index(self.current_index - 1)

        elif event.key == 'shift+left':
            self._set_time(self.time[self.current_index] - 1)

        elif event.key == 'right':
            self._set_index(self.current_index + 1)

        elif event.key == 'shift+right':
            self._set_time(self.time[self.current_index] + 1)

        elif event.key == '-':
            self.playback_speed /= 2
            self.objects['Axes'].set_title(
                f'Playback set to {self.playback_speed}x')

        elif event.key == '+':
            self.playback_speed *= 2
            self.objects['Axes'].set_title(
                f'Playback set to {self.playback_speed}x')

        elif event.key == 'h':
            if self.objects['Help'] is None:
                self.objects['Help'] = self.objects['Axes'].text(
                    -1.5, -1, self._help_text, color=[0, 1, 0],
                    fontfamily='monospace')
            else:
                self.objects['Help'].remove()
                self.objects['Help'] = None

        elif event.key == 'd':
            self.perspective = not self.perspective
            if self.perspective is True:
                self.objects['Axes'].set_title(
                    f'Camera set to perspective')
            else:
                self.objects['Axes'].set_title(
                    f'Camera set to orthogonal')

        elif event.key == 't':
            self.track = not self.track
            if self.track is True:
                self.objects['Axes'].set_title(
                    f'Marker tracking activated')
            else:
                self.objects['Axes'].set_title(
                    f'Marker tracking deactivated')

        elif event.key == 'shift':
            self.state['ShiftPressed'] = True

        self._update_plots()

    def _on_release(self, event):  # pragma: no cover
        if event.key == 'shift':
            self.state['ShiftPressed'] = False

    def _on_scroll(self, event):  # pragma: no cover
        if event.button == 'up':
            self.zoom *= 1.05
        elif event.button == 'down':
            self.zoom /= 1.05
        self._update_plots()

    def _on_mouse_press(self, event):  # pragma: no cover

        if len(self.last_selected_marker) > 0:
            self._set_new_target(
                self.markers.data[
                    self.last_selected_marker][self.current_index])

        self.state['TranslationOnMousePress'] = self.translation
        self.state['AzimutOnMousePress'] = self.azimuth
        self.state['ElevationOnMousePress'] = self.elevation
        self.state['ZoomOnMousePress'] = self.zoom
        self.state['MousePositionOnPress'] = (event.x, event.y)
        if event.button == 1:
            self.state['MouseLeftPressed'] = True
        elif event.button == 2:
            self.state['MouseMiddlePressed'] = True
        elif event.button == 3:
            self.state['MouseRightPressed'] = True

    def _on_mouse_release(self, event):  # pragma: no cover
        if event.button == 1:
            self.state['MouseLeftPressed'] = False
        elif event.button == 2:
            self.state['MouseMiddlePressed'] = False
        elif event.button == 3:
            self.state['MouseRightPressed'] = False

    def _on_mouse_motion(self, event):  # pragma: no cover
        # Pan:
        if ((self.state['MouseLeftPressed'] and self.state['ShiftPressed']) or
                self.state['MouseMiddlePressed']):
            self.translation = (
                self.state['TranslationOnMousePress'][0] +
                (event.x - self.state['MousePositionOnPress'][0]) /
                (100 * self.zoom),
                self.state['TranslationOnMousePress'][1] +
                (event.y - self.state['MousePositionOnPress'][1]) /
                (100 * self.zoom))
            self._update_plots()

        # Rotation:
        elif self.state['MouseLeftPressed'] and not self.state['ShiftPressed']:
            self.azimuth = self.state['AzimutOnMousePress'] - \
                (event.x - self.state['MousePositionOnPress'][0]) / 250
            self.elevation = self.state['ElevationOnMousePress'] - \
                (event.y - self.state['MousePositionOnPress'][1]) / 250
            self._update_plots()

        # Zoom:
        elif self.state['MouseRightPressed']:
            self.zoom = self.state['ZoomOnMousePress'] + \
                (event.y - self.state['MousePositionOnPress'][1]) / 250
            self._update_plots()

    def to_html5(self,
                 start_index: Optional[int] = None,
                 stop_index: Optional[int] = None,
                 start_time: Optional[int] = None,
                 stop_time: Optional[int] = None,
                 **kwargs):
        """
        Create an html5 video for displaying in Jupyter notebooks.

        Parameters
        ----------
        start_index
            Start index in the exported video. Default is the beginning.

        stop_index
            Stop index in the exported video. Negative numbers to count from
            the end. Default is the end.

        start_time
            Start time in the exported video. Default is the beginning.
            Ignored if start_index is set.

        stop_time
            Stop time in the exported video. Default is the end.
            Ignored if stop_index is set.

        Returns
        -------
        A matplotlib animation object for displaying as html5.

        Warnings
        --------
        This function, which has been introduced in 0.4, is still experimental
        and may change signature or behaviour in the future.

        """
        # Allow old arguments
        if 'start_frame' in kwargs:
            start_index = kwargs['start_frame']
        if 'stop_frame' in kwargs:
            stop_index = kwargs['stop_frame']

        mpl.rcParams['animation.html'] = 'html5'

        if start_index is None:
            if start_time is None:
                start_index = 0
            else:
                start_index = int(np.argmin(np.abs(self.time - start_time)))

        if stop_index is None:
            if stop_time is None:
                stop_index = -1
            else:
                stop_index = int(np.argmin(np.abs(self.time - stop_time)))

        if stop_index < 0:
            stop_index = self.n_indexes + stop_index + 1

        self.playback_speed = 0
        self.objects['Figure'].set_size_inches(6, 4.5)  # Half size
        self._set_index(start_index)
        self.running = True
        self.anim.save_count = stop_index - start_index
        self.anim.event_source.start()
        plt.close(self.objects['Figure'])
        return self.anim
