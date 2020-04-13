"""
Implementation of the ktk.Player class.

Author: Felix Chenier
"""

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import time
from ktk._timeseries import TimeSeries


class Player:
    """
    A class that allows visualizing markers and rigid bodies in 3D.

    player = ktk.Player(parameters) creates and launches a Player
    instance.

    Parameters
    ----------
    markers : TimeSeries (optional)
        Contains the markers to visualize, where each data key is a marker
        position expressed as Nx4 array (N=time). Default is None.
        markers and rigid_bodies cannot be both set to None.
    rigid_bodies : TimeSeries (optional)
        Contains the rigid bodies to visualize, where each data key is
        a rigid body pose expressed as a Nx4x4 array (N=time, the other
        dimensions are transformation matrices). Default is None.
        markers and rigid_bodies cannot be both set to None.
    segments : dict
        Used to draw lines between markers. Each key corresponds to a
        segment, where a segment is another dict with the following keys:
            - Links: list of list of 2 str, where each str is a marker
                     name. For example, to link Marker1 to Marker2 and
                     Marker1 to Marker3, Links would be:
                         [['Marker1', 'Marker2'], ['Marker1', 'Marker3']]
            - Color: character or tuple that represents the color of the
                     link. Color must be a valid value for matplotlib's
                     plots.
    current_frame : int (optional)
        Sets the inital frame number to show. Default is 0.
    marker_radius : float (optional)
        Sets the marker radius as defined by matplotlib. Default is 0.008.
    rigid_body_size : float (optional)
        Sets the rigid body size in meters. Default is 0.1.
    zoom : float (optional)
        Sets the initial camera zoom. Default is 1.0.
    azimuth : float (optional)
        Sets the initial camera azimuth in radians. Default is 0.0.
    elevation : float (optional)
        Sets the initial camera elevation in radians. Default is 0.0.
    translation : tuple of floats (optional)
        Sets the initial camera translation (panning). Default is (0.0, 0.0)
    target : tuple of floats or 'centroid' (optional)
        Sets the camera target in meters. Default is (0.0, 0.8, 0.0)
        If set to 'centroid', then the target is continuously updated to
        the centroid of the markers, which allows following moving
        objects more easily.

    Returns
    -------
    Player
    """

    def __init__(self, markers=None, rigid_bodies=None, segments=None,
                 current_frame=0, marker_radius=0.008, rigid_body_size=0.1,
                 zoom=1.0, azimuth=0.0, elevation=0.0, translation=(0.0, 0.0),
                 target=(0.0, 0.8, 0.0)):

        # ---------------------------------------------------------------
        # Set self.n_frames and self.time, and verify that we have at least
        # markers or rigid bodies.
        if markers is not None:
            self.time = markers.time
            self.n_frames = len(markers.time)
        elif rigid_bodies is not None:
            self.time = rigid_bodies.time
            self.n_frames = len(rigid_bodies.time)
        else:
            raise(ValueError('Either markers or rigid_bodies must be set.'))

        # ---------------------------------------------------------------
        # Assign the markers
        self.markers = markers
        self._select_none()

        # ---------------------------------------------------------------
        # Assign the rigid bodies
        if rigid_bodies is not None:
            self.rigid_bodies = rigid_bodies.copy()
        else:
            self.rigid_bodies = TimeSeries(time=markers.time)

        # Add the origin to the rigid bodies
        self.rigid_bodies.data['Global'] = np.repeat(
                np.eye(4, 4)[np.newaxis, :, :], self.n_frames, axis=0)

# TODO:  Continue the development to get a reference frame at the bottom left
#        of the screen.
#        # Add the origin without translation to the rigid bodies (for the
#        # fixed rigid body in the screen bottom-left
#        self.rigid_bodies.data['GlobalBottomLeft'] = np.repeat(
#                np.array([[1, 0, 0, 0],
#                          [0, 1, 0, 0],
#                          [0, 0, 1, 0],
#                          [0, 0, 0, 0]])[np.newaxis, :, :],
#                self.n_frames, axis=0)

        # ---------------------------------------------------------------
        # Assign the segments
        self.segments = segments

        # ---------------------------------------------------------------
        # Other initalizations
        self.current_frame = current_frame
        self.marker_radius = marker_radius
        self.rigid_body_size = rigid_body_size
        self.zoom = zoom
        self.azimuth = azimuth
        self.elevation = elevation
        self.target = target
        self.translation = translation
        self.playback_speed = 1.0
        self.anim = None

        self.objects = dict()
        self._colors = ['r', 'g', 'b', 'y', 'c', 'm', 'w']
        self.objects['PlotMarkers'] = dict()
        for color in self._colors:
            self.objects['PlotMarkers'][color] = None  # Not selected
            self.objects['PlotMarkers'][color + 's'] = None  # Selected
        self.objects['PlotRigidBodiesX'] = None
        self.objects['PlotRigidBodiesY'] = None
        self.objects['PlotRigidBodiesZ'] = None
        self.objects['PlotGroundPlane'] = None
        self.objects['PlotSegments'] = dict()

        self.objects['Figure'] = None
        self.objects['Axes'] = None
        self.objects['Help'] = None

        self.state = dict()
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

        self._create_figure()

    def _create_figure(self):
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

        # Init the markers plots
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
                    c=colors[color], markersize=4, picker=5)[0]
        for color in self._colors:
            self.objects['PlotMarkers'][color + 's'] = \
                    self.objects['Axes'].plot(
                            np.nan, np.nan, '.',
                            c=colors[color], markersize=12)[0]

        # Init the segments plots
        if self.segments is not None:
            for segment in self.segments:
                self.objects['PlotSegments'][segment] = \
                        self.objects['Axes'].plot(np.nan, np.nan, '-',
                                    c=self.segments[segment]['Color'])[0]

        # Draw the markers
        self._update_plots()

        plt.axis([-1.5, 1.5, -1, 1])

        # # Start the animation timer
        # self.anim = animation.FuncAnimation(self.objects['Figure'],
        #                                     self._on_timer,
        #                                     interval=33)  # 30 ips

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
            denom = z/5+5
            x = x / denom
            y = y / denom
            with np.errstate(invalid='ignore'):
                to_remove = (denom < 1E-12)
            x[to_remove] = np.nan
            y[to_remove] = np.nan
            return x, y

        # Get a Nx4 matrices of every marker at the current frame
        markers = self.markers
        if markers is not None:
            n_markers = len(markers.data)
        else:
            n_markers = 0

        markers_data = dict()  # Used to draw the markers with different colors
        segment_markers = dict()  # Used to draw the segments

        centroid = np.empty([n_markers, 4])
        for color in self._colors:
            markers_data[color] = np.empty([n_markers, 4])
            markers_data[color][:] = np.nan

            markers_data[color + 's'] = np.empty([n_markers, 4])
            markers_data[color + 's'][:] = np.nan

        if n_markers > 0:
            for i_marker, marker in enumerate(markers.data):

                # Get this marker's color
                try:
                    color = markers.data_info[marker]['Color']
                except KeyError:
                    color = 'w'

                these_coordinates = markers.data[marker][self.current_frame]
                markers_data[color][i_marker] = these_coordinates
                segment_markers[marker] = these_coordinates
                centroid[i_marker] = these_coordinates

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

        # ------------------------------------------------------------
        # Create the rotation matrix to convert the lab's coordinates
        # (x anterior, y up, z right) to the camera coordinates (x right,
        # y up, z deep)
        if self.target == 'centroid':
            centroid = np.nanmean(centroid, axis=0)
            if np.all(np.isnan(centroid)):
                centroid = np.nanmean(rbx_data, axis=0)
        else:
            centroid = self.target

        R = (np.array([[2*self.zoom, 0, 0, 0],
                       [0, 2*self.zoom, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]]) @
             np.array([[1, 0, 0, self.translation[0]],  # Pan
                       [0, 1, 0, self.translation[1]],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]]) @
             np.array([[1, 0, 0, 0],
                       [0, np.cos(-self.elevation), np.sin(self.elevation), 0],
                       [0, np.sin(-self.elevation), np.cos(-self.elevation), 0],
                       [0, 0, 0, 1]]) @
             np.array([[np.cos(-self.azimuth), 0, np.sin(self.azimuth), 0],
                       [0, 1, 0, 0],
                       [np.sin(-self.azimuth), 0, np.cos(-self.azimuth), 0],
                       [0, 0, 0, 1]]) @
             np.array([[1, 0, 0, -centroid[0]],  # Rotate around centroid
                       [0, 1, 0, -centroid[1]],
                       [0, 0, -1, -centroid[2]],
                       [0, 0, 0, 1]]))

        for color in self._colors:
            markers_data[color] = (R @ markers_data[color].T).T
            markers_data[color + 's'] = (R @ markers_data[color + 's'].T).T
        for marker in segment_markers:
            segment_markers[marker] = (R @ segment_markers[marker])
        rbx_data = (R @ rbx_data.T).T
        rby_data = (R @ rby_data.T).T
        rbz_data = (R @ rbz_data.T).T

        # ------------------------------------------------------------
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

        # ----------------------------------------
        # Create or update the plots

        # Create or update the ground plane plot
        x, y = get_perspective(gp[0, :], gp[1, :], gp[2, :])
        if self.objects['PlotGroundPlane'] is None:  # Create the plot
            self.objects['PlotGroundPlane'] = self.objects['Axes'].plot(
                    x, y, c=[0.3, 0.3, 0.3],
                    linewidth=1)[0]
        else:  # Update the plot
            self.objects['PlotGroundPlane'].set_data(x, y)

        # Create or update the markers plot
        for color in self._colors:
            x, y = get_perspective(markers_data[color][:, 0],
                                   markers_data[color][:, 1],
                                   markers_data[color][:, 2])
            self.objects['PlotMarkers'][color].set_data(x, y)

            x, y = get_perspective(markers_data[color + 's'][:, 0],
                                   markers_data[color + 's'][:, 1],
                                   markers_data[color + 's'][:, 2])
            self.objects['PlotMarkers'][color + 's'].set_data(x, y)


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

        # Update the segments plot
        if self.segments is not None:
            for segment in self.segments:
                n_links = len(self.segments[segment]['Links'])
                coordinates = np.empty((2*n_links, 4))
                for i_link in range(n_links):
                    marker1 = self.segments[segment]['Links'][i_link][0]
                    marker2 = self.segments[segment]['Links'][i_link][1]
                    if marker1 in segment_markers:
                        coordinates[2*i_link] = segment_markers[marker1]
                    else:
                        coordinates[2*i_link] = np.nan

                    if marker2 in segment_markers:
                        coordinates[2*i_link+1] = segment_markers[marker2]
                    else:
                        coordinates[2*i_link+1] = np.nan

                x, y = get_perspective(coordinates[:, 0],
                                       coordinates[:, 1],
                                       coordinates[:, 2])

                # Separate each segment by nans
                new_x = np.empty(int(3*x.shape[0]/2))
                new_y = np.empty(int(3*y.shape[0]/2))
                new_x[0::3] = x[0::2]
                new_x[1::3] = x[1::2]
                new_x[2::3] = np.nan
                new_y[0::3] = y[0::2]
                new_y[1::3] = y[1::2]
                new_y[2::3] = np.nan

                self.objects['PlotSegments'][segment].set_data(new_x, new_y)

        # Update the window title
        self.objects['Figure'].canvas.set_window_title(
                f'Frame {self.current_frame}, ' +
                '%2.2f s.' % self.time[self.current_frame])

        self.objects['Figure'].canvas.draw()

    # ------------------------------------
    # Helper functions
    def _set_frame(self, frame):
        """Set current frame to a given frame and update plots."""
        if frame >= self.n_frames:
            self.current_frame = self.n_frames - 1
        elif frame < 0:
            self.current_frame = 0
        else:
            self.current_frame = frame
        self._update_plots()

    def _set_time(self, time):
        """Set current frame to a given time and update plots."""
        index = np.argmin(np.abs(self.markers.time - time))
        self._set_frame(index)

    def _select_none(self):
        """Deselect every markers."""
        if self.markers is not None:
            for marker in self.markers.data:
                try:
                    # Keep 1st character, remove the possible 's'
                    self.markers.data_info[marker]['Color'] = \
                            self.markers.data_info[marker]['Color'][0]
                except KeyError:
                    self.markers.add_data_info(marker, 'Color', 'w')

    # ------------------------------------
    # Callbacks
    def _on_timer(self, _):
        if self.anim is not None:
            # We check self.anim because the garbage collector may take time
            # before deleting the animation timer, and unreferencing the
            # animation timer is the recommended way to deactivate a timer.
            """Callback for the animation timer object."""
            current_frame = self.current_frame
            self._set_time(self.time[self.current_frame] +
                           self.playback_speed * (
                           time.time() -
                           self.state['SystemTimeOnLastUpdate']))
            if current_frame == self.current_frame:
                # The time wasn't enough to advance a frame. Articifically
                # advance a frame.
                self._set_frame(self._current_frame + 1)
            self.state['SystemTimeOnLastUpdate'] = time.time()
            self._update_plots()

    def _on_pick(self, event):
        """Callback for marker selection."""
        if event.mouseevent.button == 1:
            index = event.ind
            selected_marker = list(self.markers.data.keys())[index[0]]
            self.objects['Axes'].set_title(selected_marker)

            # Mark selected
            self._select_none()
            self.markers.data_info[selected_marker]['Color'] = \
                    self.markers.data_info[selected_marker]['Color'][0] + 's'
            self._update_plots()

    def _on_key(self, event):
        """Callback for keyboard key pressed."""
        if event.key == ' ':
            if self.anim is None:
                # Start the animation timer
                self.state['SystemTimeOnLastUpdate'] = time.time()
                self.state['SelfTimeOnPlay'] = self.time[self.current_frame]
                self.running = True

                # Start the animation timer
                self.anim = animation.FuncAnimation(self.objects['Figure'],
                                                    self._on_timer,
                                                    interval=33)  # 30 ips
                self._update_plots()

            else:
                self.anim = None


        elif event.key == 'left':
            self._set_frame(self.current_frame - 1)

        elif event.key == 'shift+left':
            self._set_time(self.time[self.current_frame] - 1)

        elif event.key == 'right':
            self._set_frame(self.current_frame + 1)

        elif event.key == 'shift+right':
            self._set_time(self.time[self.current_frame] + 1)

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
                self.objects['Help'] = self.objects['Axes'].text(-1.5, -1, '''
                                  ktk.Player help
                ----------------------------------------------------
                KEYBOARD COMMANDS
                show/hide this help : h
                previous frame      : left
                next frame          : right
                previous second     : shift+left
                next second         : shift+right
                play/pause          : space
                2x playback speed   : +
                0.5x playback speed : -
                ----------------------------------------------------
                MOUSE COMMANDS
                select a marker     : left-click
                3d rotate           : left-drag
                pan                 : middle-drag or shift+left-drag
                zoom                : right-drag or wheel
                ''', color=[0,1,0], fontfamily='monospace')
            else:
                self.objects['Help'].remove()
                self.objects['Help'] = None

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
