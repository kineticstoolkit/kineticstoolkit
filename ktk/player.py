#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 16:17:10 2019

@author: felix
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import numpy as np
import numpy.ma as ma
import time
from ktk._timeseries import TimeSeries


class Player:

    def __init__(self, markers=None, rigid_bodies=None, segments=None,
                 sample=0, marker_radius=0.008, rigid_body_size=0.2):

        # Set self.n_frames, and verify that we have at least markers or
        # rigid bodies.
        if markers is not None:
            self.n_frames = len(markers.time)
        elif rigid_bodies is not None:
            self.n_frames = len(rigid_bodies.time)
        else:
            raise(ValueError('Either markers or rigid_bodies must be set.'))

        self.markers = markers

        if rigid_bodies is not None:
            self.rigid_bodies = rigid_bodies.copy()
        else:
            self.rigid_bodies = TimeSeries(time=markers.time)

        # Add the origin to the rigid bodies
        self.rigid_bodies.data['Global'] = np.repeat(
                np.eye(4, 4)[np.newaxis, :, :], self.n_frames, axis=0)

        self.segments = segments
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
        # Create the 3d figure
        self.objects['Figure'], ax = plt.subplots(num=None,
                                               figsize=(9, 9),
                                               facecolor='k',
                                               edgecolor='w')

        # Remove the toolbar
        self.objects['Figure'].canvas.toolbar.setVisible(False)

        # Create the 3d axis
        self.objects['Axes'] = self.objects['Figure'].add_subplot(
                111, facecolor='k', projection='3d')
        self.objects['Axes'].elev = 0
        self.objects['Axes'].azim = 0
        self.objects['Axes'].disable_mouse_rotation()

        plt.tight_layout()

        # Delete the first (non-3d) axis to speed up.
        self.objects['Figure'].delaxes(ax)

        # Add the title
        title_obj = plt.title('Player')
        plt.getp(title_obj)
        plt.setp(title_obj, color=[0, 1, 0])  # Set a green title

        # Remove the background for faster plotting
        self.objects['Axes'].set_axis_off()

        # Draw the markers
        self._update_plots()

        coordinate = (0, 0, 0)
        lim = 4
        xlim = [coordinate[0] - lim/2, coordinate[0] + lim/2]
        ylim = [coordinate[1] - lim/2, coordinate[1] + lim/2]
        zlim = [coordinate[2] - lim/2, coordinate[2] + lim/2]
        self.objects['Axes'].set_xlim(xlim)
        self.objects['Axes'].set_ylim(ylim)
        self.objects['Axes'].set_zlim(zlim)

        # Start the animation timer
        self.anim = animation.FuncAnimation(self.objects['Figure'],
                                       self._timer_event,
                                       interval=33)  # 30 ips

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

        # Create the rotation matrix to convert the lab's coordinates
        # (x anterior, y up, z right) to mplot3d coordinates (x anterior,
        # y right, z up)
        R = (self.zoom *
             np.array([[0, 0, -1, 0],
                       [1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1]]) @
             np.array([[1, 0, 0, self.target[0]],
                       [0, 1, 0, self.target[1]],
                       [0, 0, 1, self.target[2]],
                       [0, 0, 0, 1]]) @
             np.array([[1, 0, 0, 0],
                       [0, -np.sin(self.elevation), np.cos(self.elevation), 0],
                       [0, np.cos(self.elevation), np.sin(self.elevation), 0],
                       [0, 0, 0, 1]]) @
             np.array([[np.cos(self.azimuth), 0, np.sin(self.azimuth), 0],
                       [0, 1, 0, 0],
                       [-np.sin(self.azimuth), 0, np.cos(self.azimuth), 0],
                       [0, 0, 0, 1]]))

        # Get a matrix of every marker at a given index
        markers = self.markers
        n_markers = len(markers.data)
        markers_data = np.empty([n_markers, 3])

        for i_marker, marker in enumerate(markers.data):
            markers_data[i_marker] = (R @ markers.data[marker][self.current_frame])[0:3]

        # Create or update the markers plot
        if self.objects['PlotMarkers'] is None:  # Create the plot
            self.objects['PlotMarkers'] = self.objects['Axes'].plot(
                    markers_data[:, 0],
                    markers_data[:, 1],
                    markers_data[:, 2],
                    '.', c='w', picker=5)[0]
#            self.objects['PlotMarkers'] = ax.scatter(x, y, z, s=self.marker_radius*1000, c='b', picker=5)

        else:  # Update the plot with new values
            self.objects['PlotMarkers'].set_data(
                    markers_data[:, 0],
                    markers_data[:, 1])
            self.objects['PlotMarkers'].set_3d_properties(
                    markers_data[:, 2])


        # Get a matrix of every rigid body at a given index
        rigid_bodies = self.rigid_bodies
        n_rigid_bodies = len(rigid_bodies.data)
        rbx_data = np.empty([n_rigid_bodies * 3, 4])
        rby_data = np.empty([n_rigid_bodies * 3, 4])
        rbz_data = np.empty([n_rigid_bodies * 3, 4])

        for i_rigid_body, rigid_body in enumerate(rigid_bodies.data):

            # Origin
            rbx_data[i_rigid_body * 3] = R @ (
                    rigid_bodies.data[rigid_body][self.current_frame, :, 3])
            rby_data[i_rigid_body * 3] = R @ (
                    rigid_bodies.data[rigid_body][self.current_frame, :, 3])
            rbz_data[i_rigid_body * 3] = R @ (
                    rigid_bodies.data[rigid_body][self.current_frame, :, 3])

            # Direction
            rbx_data[i_rigid_body * 3 + 1] = R @ (
                    rigid_bodies.data[rigid_body][self.current_frame] @
                    np.array([self.rigid_body_size, 0, 0, 1]))
            rby_data[i_rigid_body * 3 + 1] = R @ (
                    rigid_bodies.data[rigid_body][self.current_frame] @
                    np.array([0, self.rigid_body_size, 0, 1]))
            rbz_data[i_rigid_body * 3 + 1] = R @ (
                    rigid_bodies.data[rigid_body][self.current_frame] @
                    np.array([0, 0, self.rigid_body_size, 1]))

            # NaN
            rbx_data[i_rigid_body * 3 + 2] = np.repeat(np.nan, 4)
            rby_data[i_rigid_body * 3 + 2] = np.repeat(np.nan, 4)
            rbz_data[i_rigid_body * 3 + 2] = np.repeat(np.nan, 4)

        # Create or update the rigid bodies plot
        if self.objects['PlotRigidBodiesX'] is None:  # Create the plot
            self.objects['PlotRigidBodiesX'] = self.objects['Axes'].plot(
                    rbx_data[:, 0], rbx_data[:, 1], rbx_data[:, 2], c='r')[0]
            self.objects['PlotRigidBodiesY'] = self.objects['Axes'].plot(
                    rby_data[:, 0], rby_data[:, 1], rby_data[:, 2], c='g')[0]
            self.objects['PlotRigidBodiesZ'] = self.objects['Axes'].plot(
                    rbz_data[:, 0], rbz_data[:, 1], rbz_data[:, 2], c='b')[0]
        else:  # Update the plot
            self.objects['PlotRigidBodiesX'].set_data(
                    rbx_data[:, 0],
                    rbx_data[:, 1])
            self.objects['PlotRigidBodiesX'].set_3d_properties(
                    rbx_data[:, 2])
            self.objects['PlotRigidBodiesY'].set_data(
                    rby_data[:, 0],
                    rby_data[:, 1])
            self.objects['PlotRigidBodiesY'].set_3d_properties(
                    rby_data[:, 2])
            self.objects['PlotRigidBodiesZ'].set_data(
                    rbz_data[:, 0],
                    rbz_data[:, 1])
            self.objects['PlotRigidBodiesZ'].set_3d_properties(
                    rbz_data[:, 2])



        # Update the window title
        self.objects['Figure'].canvas.set_window_title(f'Frame {self.current_frame}, ' +
                  '%2.2f s.' % self.markers.time[self.current_frame])

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
        mouse_button= event.mouseevent.button.value
#            line = event.artist
        index = event.ind
        selected_marker = list(self.markers.data.keys())[index[0]]

        if mouse_button == 2:  # Center on selected marker

            coordinate = self.markers.data[selected_marker][
                    self.current_frame]

            xlim = self.objects['Axes'].get_xlim()
            ylim = self.objects['Axes'].get_ylim()
            zlim = self.objects['Axes'].get_zlim()
            lim = np.max([xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]])
            lim = 4
            xlim = [coordinate[0] - lim/2, coordinate[0] + lim/2]
            ylim = [coordinate[1] - lim/2, coordinate[1] + lim/2]
            zlim = [coordinate[2] - lim/2, coordinate[2] + lim/2]
            self.objects['Axes'].set_xlim(xlim)
            self.objects['Axes'].set_ylim(ylim)
            self.objects['Axes'].set_zlim(zlim)

        else:

            plt.title(selected_marker)

        plt.pause(1E-6)
#        print('on pick line:', np.array([xdata[ind], ydata[ind]]).T)

    def _on_key(self, event):
        print('you pressed', event.key, event.xdata, event.ydata)

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
        if self.state['MouseLeftPressed'] is True:

            if self.state['ShiftPressed'] is True:  # Pan
                self.target = (
                        self.state['TargetOnMousePress'][0]
                        + (event.x - self.state['MousePositionOnPress'][0]) / 100,
                        self.state['TargetOnMousePress'][1]
                        + (event.y - self.state['MousePositionOnPress'][1]) / 100,
                        0)
                self._update_plots()

        elif self.state['MouseRightPressed'] is True:
            self.azimuth = self.state['AzimutOnMousePress'] + \
                (event.x - self.state['MousePositionOnPress'][0]) / 500
            self.elevation = self.state['ElevationOnMousePress'] + \
                (event.y - self.state['MousePositionOnPress'][1]) / 500
            self._update_plots()
