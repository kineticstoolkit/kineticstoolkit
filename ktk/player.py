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


class Player:

    def __init__(self, markers=None, rigid_bodies=None, segments=None,
           sample=0, marker_radius=0.008, rigid_body_size=0.1):

        self.markers = markers
        self.rigid_bodies = rigid_bodies
        self.segments = segments
        self.current_frame = sample
        self.marker_radius = marker_radius
        self.rigid_body_size = rigid_body_size
        self.running = False
        self.scatter = None
        self.last_update = time.time()

        self._create_figure()

    def _update_scatter(self, frame=None):

        # Plot every marker at a given index
        markers = self.markers
        x = []
        y = []
        z = []
        min_coordinate = 99999999.
        max_coordinate = -99999999.

        for data in markers.data:
            temp_data = markers.data[data]
            x.append(temp_data[self.current_frame, 0])
            y.append(temp_data[self.current_frame, 2])
            z.append(temp_data[self.current_frame, 1])
#            min_coordinate = np.min([min_coordinate, np.nanmin(temp_data[:, 0])])
#            min_coordinate = np.min([min_coordinate, np.nanmin(temp_data[:, 1])])
#            min_coordinate = np.min([min_coordinate, np.nanmin(temp_data[:, 2])])
#            max_coordinate = np.max([max_coordinate, np.nanmax(temp_data[:, 0])])
#            max_coordinate = np.max([max_coordinate, np.nanmax(temp_data[:, 1])])
#            max_coordinate = np.max([max_coordinate, np.nanmax(temp_data[:, 2])])

        ax = plt.gca()
#        ax.cla()
        if self.scatter is None:
            self.scatter = ax.plot(x, y, z, '.', c='w', picker=5)[0]
#            self.scatter = ax.scatter(x, y, z, s=self.marker_radius*1000, c='b', picker=5)

        self.scatter.set_data(x, y)
        self.scatter.set_3d_properties(np.array(z))

        self.figure.canvas.set_window_title(f'Frame {self.current_frame}, ' +
                  '%2.2f s.' % self.markers.time[self.current_frame])

#        self.scatter._offsets3d = (ma.masked_array(np.nan_to_num(x), False),
#                                   ma.masked_array(np.nan_to_num(y), False),
#                                   ma.masked_array(np.nan_to_num(z), False))
      #  self.scatter._offsets3d = (np.array(x), np.array(y), np.array(z))

        plt.pause(1E-6)

    def _timer_event(self, frame=None):
        if self.running is True:
            self._set_frame_to_time(time.time() - self.last_update)

    def _set_frame_to_time(self, time):
        index = np.argmin(np.abs(self.markers.time - time))
        self.current_frame = index
        self._update_scatter()

    def _next_frame(self, frame=None):
        self.current_frame += 1
        self._update_scatter()

    def _previous_frame(self, frame=None):
        self.current_frame -= 1
        self._update_scatter()

    def _create_figure(self):
        # Create the 3d figure
        fig, ax = plt.subplots(num=None,
                               figsize=(16*0.75, 12*0.75),
                               facecolor='k', edgecolor='w')
        self.figure = fig

        # Remove the toolbar
        self.figure.canvas.toolbar.setVisible(False)

        # Create the 3d axis
        ax = self.figure.add_subplot(111, facecolor='k',
                                     projection='3d')
        plt.tight_layout()

        # Add the title
        title_obj = plt.title('Player')
        plt.getp(title_obj)
        plt.setp(title_obj, color=[0, 1, 0])  # Set a green title

        # Remove the background for faster plotting
        ax.set_axis_off()
        self._update_scatter()

        # Start the animation timer
        self.anim = animation.FuncAnimation(self.figure,
                                       self._timer_event,
                                       interval=33)  # 30 ips


        def on_pick(event):
            mouse_button= event.mouseevent.button.value
#            line = event.artist
            index = event.ind
            selected_marker = list(self.markers.data.keys())[index[0]]

            if mouse_button == 2:  # Center on selected marker

                coordinate = self.markers.data[selected_marker][
                        self.current_frame]

                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                zlim = ax.get_zlim()
                lim = np.max([xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]])
                lim = 4
                xlim = [coordinate[0] - lim/2, coordinate[0] + lim/2]
                ylim = [coordinate[1] - lim/2, coordinate[1] + lim/2]
                zlim = [coordinate[2] - lim/2, coordinate[2] + lim/2]
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_zlim(zlim)

            else:

                plt.title(selected_marker)

            plt.pause(1E-6)
    #        print('on pick line:', np.array([xdata[ind], ydata[ind]]).T)

        def on_key(event):
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

        self.figure.canvas.mpl_connect('pick_event', on_pick)
        self.figure.canvas.mpl_connect('key_press_event', on_key)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()
        lim = np.max([xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]])
        xlim = [np.mean(xlim) - lim/2, np.mean(xlim) + lim/2]
        ylim = [np.mean(ylim) - lim/2, np.mean(ylim) + lim/2]
        zlim = [np.mean(zlim) - lim/2, np.mean(zlim) + lim/2]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)

        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
