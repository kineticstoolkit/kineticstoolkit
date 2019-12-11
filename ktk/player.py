#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 16:17:10 2019

@author: felix
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class Player:

    def __init__(self, markers=None, rigid_bodies=None, segments=None,
           sample=0, marker_radius=0.008, rigid_body_size=0.1):

        self.markers = markers
        self.rigid_bodies = rigid_bodies
        self.segments = segments
        self.current_frame = sample
        self.marker_radius = marker_radius
        self.rigid_body_size = rigid_body_size

        self._create_figure()

    def _update_scatter(self, ax):
        # Plot every marker at a given index
        markers = self.markers.copy()  # Since we add stuff to it.
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
            min_coordinate = np.min([min_coordinate, np.nanmin(temp_data[:, 0])])
            min_coordinate = np.min([min_coordinate, np.nanmin(temp_data[:, 1])])
            min_coordinate = np.min([min_coordinate, np.nanmin(temp_data[:, 2])])
            max_coordinate = np.max([max_coordinate, np.nanmax(temp_data[:, 0])])
            max_coordinate = np.max([max_coordinate, np.nanmax(temp_data[:, 1])])
            max_coordinate = np.max([max_coordinate, np.nanmax(temp_data[:, 2])])

        ax.cla()
        ax.scatter(x, y, z, s=self.marker_radius*1000, c='b', picker=5)
        plt.pause(1E-6)

    def _create_figure(self):
        # Create the 3d figure
        self.figure = plt.figure()
        self.figure.canvas.toolbar.setVisible(False)

        ax = self.figure.add_subplot(111, projection='3d')
        self._update_scatter(ax)

        def on_pick(event):
            line = event.artist
            print(dir(line))
    #        xdata, ydata = line.get_data()
            index = event.ind
            plt.title(list(self.markers.data.keys())[index[0]])
            plt.pause(1E-6)
    #        print('on pick line:', np.array([xdata[ind], ydata[ind]]).T)

        def on_key(event):
            print('you pressed', event.key, event.xdata, event.ydata)
            self.current_frame += 1
            self._update_scatter(ax)

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
