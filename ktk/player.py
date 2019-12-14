#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 16:17:10 2019

@author: felix
"""

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import time
from ktk._timeseries import TimeSeries


class Player:

    def __init__(self, markers=None, rigid_bodies=None, segments=None,
                 sample=0, marker_radius=0.008, rigid_body_size=0.1):

        # ---------------------------------------------------------------
        # Set self.n_frames, and verify that we have at least markers or
        # rigid bodies.
        if markers is not None:
            self.n_frames = len(markers.time)
        elif rigid_bodies is not None:
            self.n_frames = len(rigid_bodies.time)
        else:
            raise(ValueError('Either markers or rigid_bodies must be set.'))

        # ---------------------------------------------------------------
        # Assign the markers
        self.markers = markers

        # ---------------------------------------------------------------
        # Assign the rigid bodies
        if rigid_bodies is not None:
            self.rigid_bodies = rigid_bodies.copy()
        else:
            self.rigid_bodies = TimeSeries(time=markers.time)

        # Add the origin to the rigid bodies
        self.rigid_bodies.data['Global'] = np.repeat(
                np.eye(4, 4)[np.newaxis, :, :], self.n_frames, axis=0)

        # ---------------------------------------------------------------
        # Assign the segments
        self.segments = segments

        # ---------------------------------------------------------------
        # Other initalizations
        self.current_frame = sample
        self.marker_radius = marker_radius
        self.rigid_body_size = rigid_body_size
        self.running = False
        self.last_update = time.time()
        self.zoom = 1.0
        self.azimuth = 0.0
        self.elevation = 0.0
        self.target = (0.0, 0.0, 0.0)

        self.objects = dict()
        self.objects['PlotMarkers'] = None
        self.objects['PlotRigidBodiesX'] = None
        self.objects['PlotRigidBodiesY'] = None
        self.objects['PlotRigidBodiesZ'] = None
        self.objects['PlotGroundPlane'] = None
        self.objects['Figure'] = None
        self.objects['Axes'] = None

        self.state = dict()
        self.state['ShiftPressed'] = False
        self.state['MouseLeftPressed'] = False
        self.state['MouseMiddlePressed'] = False
        self.state['MouseRightPressed'] = False
        self.state['MousePositionOnPress'] = (0.0, 0.0)
        self.state['MousePositionOnMiddlePress'] = (0.0, 0.0)
        self.state['MousePositionOnRightPress'] = (0.0, 0.0)
        self.state['TargetOnMousePress'] = (0.0, 0.0, 0.0)
        self.state['AzimutOnMousePress'] = 0.0
        self.state['ElevationOnMousePress'] = 0.0

        self._create_figure()

    def _create_figure(self):
        """Create the player's figure."""
        # Create the figure and axes
        self.objects['Figure'], self.objects['Axes'] = plt.subplots(num=None,
                    figsize=(12, 9),
                    facecolor='k',
                    edgecolor='w')

        # Remove the toolbar
        self.objects['Figure'].canvas.toolbar.setVisible(False)
        plt.tight_layout()

        # Add the title
        title_obj = plt.title('Player')
        plt.setp(title_obj, color=[0, 1, 0])  # Set a green title

        # Remove the background for faster plotting
        self.objects['Axes'].set_axis_off()

        # Draw the markers
        self._update_plots()

        plt.axis([-1.5, 1.5, -1, 1])

        # Start the animation timer
        self.anim = animation.FuncAnimation(self.objects['Figure'],
                                       self._timer_event,
                                       interval=33)  # 30 ips

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

    def _update_plots(self):
        """Update the plots, or draw it if not plot has been drawn before."""

        def get_perspective(x, y, z):
            """Return x and y to plot, considering perspective."""
            # This uses ugly magical constants but it works fine for now.
            denom = z/10+2
            x = x / denom
            y = y / denom
            to_remove = (denom < 1E-10)
            x[to_remove] = np.nan
            y[to_remove] = np.nan
            return x, y

        # Get a Nx4 matrix of every marker at the current frame
        markers = self.markers
        n_markers = len(markers.data)
        markers_data = np.empty([n_markers, 4])
        for i_marker, marker in enumerate(markers.data):
            markers_data[i_marker] = (markers.data[marker][self.current_frame])

        # Get three (3N)x4 matrices (for x, y and z lines) for the rigid bodies
        # at the current frame
        rigid_bodies = self.rigid_bodies
        n_rigid_bodies = len(rigid_bodies.data)
        rbx_data = np.empty([n_rigid_bodies * 3, 4])
        rby_data = np.empty([n_rigid_bodies * 3, 4])
        rbz_data = np.empty([n_rigid_bodies * 3, 4])

        for i_rigid_body, rigid_body in enumerate(rigid_bodies.data):
            # Origin
            rbx_data[i_rigid_body * 3] = (
                    rigid_bodies.data[rigid_body][self.current_frame, :, 3])
            rby_data[i_rigid_body * 3] = (
                    rigid_bodies.data[rigid_body][self.current_frame, :, 3])
            rbz_data[i_rigid_body * 3] = (
                    rigid_bodies.data[rigid_body][self.current_frame, :, 3])
            # Direction
            rbx_data[i_rigid_body * 3 + 1] = (
                    rigid_bodies.data[rigid_body][self.current_frame] @
                    np.array([self.rigid_body_size, 0, 0, 1]))
            rby_data[i_rigid_body * 3 + 1] = (
                    rigid_bodies.data[rigid_body][self.current_frame] @
                    np.array([0, self.rigid_body_size, 0, 1]))
            rbz_data[i_rigid_body * 3 + 1] = (
                    rigid_bodies.data[rigid_body][self.current_frame] @
                    np.array([0, 0, self.rigid_body_size, 1]))
            # NaN to cut the line between the different rigid bodies
            rbx_data[i_rigid_body * 3 + 2] = np.repeat(np.nan, 4)
            rby_data[i_rigid_body * 3 + 2] = np.repeat(np.nan, 4)
            rbz_data[i_rigid_body * 3 + 2] = np.repeat(np.nan, 4)

        # Create the rotation matrix to convert the lab's coordinates
        # (x anterior, y up, z right) to the camera coordinates (x right,
        # y up, z deep)
        centroid = np.nanmean(markers_data, axis=0)
        if np.isnan(np.sum(centroid)):  # No markers, use the rigid bodies
            centroid = np.nanmean(rbx_data, axis=0)

        R = (self.zoom *
             np.array([[1, 0, 0, self.target[0]],  # Pan
                       [0, 1, 0, self.target[1]],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]]) @
             np.array([[1, 0, 0, 0],
                       [0, np.cos(self.elevation), -np.sin(self.elevation), 0],
                       [0, np.sin(self.elevation), np.cos(self.elevation), 0],
                       [0, 0, 0, 1]]) @
             np.array([[np.cos(self.azimuth), 0, -np.sin(self.azimuth), 0],
                       [0, 1, 0, 0],
                       [np.sin(self.azimuth), 0, np.cos(self.azimuth), 0],
                       [0, 0, 0, 1]]) @
             np.array([[1, 0, 0, -centroid[0]],  # Rotate around centroid
                       [0, 1, 0, -centroid[1]],
                       [0, 0, -1, -centroid[2]],
                       [0, 0, 0, 1]]))

        markers_data = (R @ markers_data.T).T
        rbx_data = (R @ rbx_data.T).T
        rby_data = (R @ rby_data.T).T
        rbz_data = (R @ rbz_data.T).T

        # Create the ground plane matrix
        gp_size = 30  # blocks
        gp_div = 4  # blocks per meter
        gp_x = np.block([
                np.tile([-gp_size/gp_div, gp_size/gp_div, np.nan], gp_size),
                np.repeat(
                        np.linspace(-gp_size/gp_div, gp_size/gp_div, gp_size),
                        3)])
        gp_y = np.zeros(6 * gp_size)
        gp_z = np.block([
                np.repeat(
                        np.linspace(-gp_size/gp_div, gp_size/gp_div, gp_size),
                        3),
                np.tile([-gp_size/gp_div, gp_size/gp_div, np.nan], gp_size)])
        gp_1 = np.ones(6 * gp_size)
        gp = R @ np.block([[gp_x], [gp_y], [gp_z], [gp_1]])

        # Create or update the plots
        # ----------------------------------------

        # Create or update the ground plane plot
        x, y = get_perspective(gp[0, :], gp[1, :], gp[2, :])
        if self.objects['PlotGroundPlane'] is None:  # Create the plot
            self.objects['PlotGroundPlane'] = self.objects['Axes'].plot(
                    x, y, c=[0.3, 0.3, 0.3],
                    linewidth=1)[0]
        else:  # Update the plot
            self.objects['PlotGroundPlane'].set_data(x, y)

        # Create or update the markers plot
        x, y = get_perspective(markers_data[:, 0],
                               markers_data[:, 1],
                               markers_data[:, 2])
        if self.objects['PlotMarkers'] is None:  # Create the plot
            self.objects['PlotMarkers'] = self.objects['Axes'].plot(
                    x, y, '.', c='w', markersize=3, picker=5)[0]

        else:  # Update the plot with new values
            self.objects['PlotMarkers'].set_data(x, y)

        # Create or update the rigid bodies plot
        xx, yx = get_perspective(rbx_data[:, 0],
                                 rbx_data[:, 1],
                                 rbx_data[:, 2])
        xy, yy = get_perspective(rby_data[:, 0],
                                 rby_data[:, 1],
                                 rby_data[:, 2])
        xz, yz = get_perspective(rbz_data[:, 0],
                                 rbz_data[:, 1],
                                 rbz_data[:, 2])
        if self.objects['PlotRigidBodiesX'] is None:  # Create the plot
            self.objects['PlotRigidBodiesX'] = self.objects['Axes'].plot(
                    xx, yx, c='r', linewidth=2)[0]
            self.objects['PlotRigidBodiesY'] = self.objects['Axes'].plot(
                    xy, yy, c='g', linewidth=2)[0]
            self.objects['PlotRigidBodiesZ'] = self.objects['Axes'].plot(
                    xz, yz, c='b', linewidth=2)[0]
        else:  # Update the plot
            self.objects['PlotRigidBodiesX'].set_data(xx, yx)
            self.objects['PlotRigidBodiesY'].set_data(xy, yy)
            self.objects['PlotRigidBodiesZ'].set_data(xz, yz)

        # Update the window title
        self.objects['Figure'].canvas.set_window_title(
                f'Frame {self.current_frame}, ' +
                '%2.2f s.' % self.markers.time[self.current_frame])

        # Refresh Matplotlib
        plt.pause(1E-6)

    def _timer_event(self, frame=None):
        if self.running is True:
            self._set_frame_to_time(time.time() - self.last_update)

    def _set_frame_to_time(self, time):
        index = np.argmin(np.abs(self.markers.time - time))
        self.current_frame = index
        self._update_plots()

    def _next_frame(self, frame=None):
        self.current_frame += 1
        self._update_plots()

    def _previous_frame(self, frame=None):
        self.current_frame -= 1
        self._update_plots()

    def _on_pick(self, event):
        index = event.ind
        selected_marker = list(self.markers.data.keys())[index[0]]
        plt.title(selected_marker)
        plt.pause(1E-6)

    def _on_key(self, event):
        if event.key == ' ':
            if self.running is True:
                self.running = False
            else:
                self.last_update = time.time()
                self.running = True
            plt.pause(1E-6)

        elif event.key == 'left':
            self._previous_frame()

        elif event.key == 'right':
            self._next_frame()

        elif event.key == 'shift':
            self.state['ShiftPressed'] = True

    def _on_release(self, event):
        if event.key == 'shift':
            self.state['ShiftPressed'] = False

    def _on_scroll(self, event):
        if event.button == 'up':
            self.zoom *= 1.05
        elif event.button == 'down':
            self.zoom /= 1.05
        self._update_plots()

    def _on_mouse_press(self, event):
        self.state['TargetOnMousePress'] = self.target
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

    def _on_mouse_release(self, event):
        if event.button == 1:
            self.state['MouseLeftPressed'] = False
        elif event.button == 2:
            self.state['MouseMiddlePressed'] = False
        elif event.button == 3:
            self.state['MouseRightPressed'] = False

    def _on_mouse_motion(self, event):

        # Pan:
        if ((self.state['MouseLeftPressed'] and self.state['ShiftPressed']) or
                self.state['MouseMiddlePressed']):
            self.target = (
                    self.state['TargetOnMousePress'][0] +
                    (event.x - self.state['MousePositionOnPress'][0]) /
                    (100 * self.zoom),
                    self.state['TargetOnMousePress'][1] +
                    (event.y - self.state['MousePositionOnPress'][1]) /
                    (100 * self.zoom),
                    0)
            self._update_plots()

        # Rotation:
        elif self.state['MouseLeftPressed'] and not self.state['ShiftPressed']:
            self.azimuth = self.state['AzimutOnMousePress'] + \
                (event.x - self.state['MousePositionOnPress'][0]) / 250
            self.elevation = self.state['ElevationOnMousePress'] + \
                (event.y - self.state['MousePositionOnPress'][1]) / 250
            self._update_plots()

        # Zoom:
        elif self.state['MouseRightPressed']:
            self.zoom = self.state['ZoomOnMousePress'] + \
                (event.y - self.state['MousePositionOnPress'][1]) / 250
            self._update_plots()
