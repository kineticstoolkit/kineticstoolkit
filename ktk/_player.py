"""
Implementation of the ktk.Player class.

Author: Felix Chenier
"""

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from numpy import sin, cos
import time
import copy
import ktk

# To intercept interactive navigation key presses:
from matplotlib.backend_bases import NavigationToolbar2

# To fit the new viewpoint on selecting a new marker
import scipy.optimize as optim


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
    rigid_body_length : float (optional)
        Sets the rigid body size in meters. Default is 0.1.
    zoom : float (optional)
        Sets the initial camera zoom. Default is 0.2.
    azimuth : float (optional)
        Sets the initial camera azimuth in radians. Default is 0.0.
    elevation : float (optional)
        Sets the initial camera elevation in radians. Default is 1.0.
    translation : tuple of floats (optional)
        Sets the initial camera translation (panning). Default is (0.0, 0.0)
    target : tuple of floats or 'centroid' (optional)
        Sets the camera target in meters. Default is (0.0, 0.0, 0.0)
        If set to 'centroid', then the target is continuously updated to
        the centroid of the markers, which allows following moving
        objects more easily.
    track : bool (optional)
        Set to True to track the last selected marker when changing frame,
        or False to keep the scene static. Default is False.
    perspective : bool (optional)
        Sets if the scene must be drawn using perspective (True) or
        orthogonal (False). Default is True.

    Returns
    -------
    Player
    """

    def __init__(self, markers=None, rigid_bodies=None, segments=None,
                 current_frame=0, marker_radius=0.008, rigid_body_length=0.1,
                 rigid_body_width=3, segment_width=1.5,
                 zoom=1.0, azimuth=0.0, elevation=0.2,
                 translation=(0.0, 0.0), target=(0.0, 0.0, 0.0),
                 track=False, perspective=True):

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
        self.last_selected_marker = None

        # ---------------------------------------------------------------
        # Assign the rigid bodies
        if rigid_bodies is not None:
            self.rigid_bodies = rigid_bodies.copy()
        else:
            self.rigid_bodies = ktk.TimeSeries(time=markers.time)

        # Add the origin to the rigid bodies
        self.rigid_bodies.data['Global'] = np.repeat(
            np.eye(4, 4)[np.newaxis, :, :], self.n_frames, axis=0)

        # ---------------------------------------------------------------
        # Assign the segments
        self.segments = segments

        # ---------------------------------------------------------------
        # Other initalizations
        self.current_frame = current_frame
        self.marker_radius = marker_radius
        self.rigid_body_length = rigid_body_length
        self.rigid_body_width = rigid_body_width
        self.segment_width = segment_width
        self.zoom = zoom
        self.azimuth = azimuth
        self.elevation = elevation
        self.target = target
        self.track = track
        self.translation = translation
        self.perspective = perspective
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

        self._help_text = '''
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
        self._create_segments()
        self._create_markers()
        self._create_ground_plane()
        self._first_refresh()

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

        # # Redirect standard toolbar callbacks
        # def do_nothing(self, *args, **kwargs):
        #     pass

        # NavigationToolbar2.home = do_nothing
        # NavigationToolbar2.pan = do_nothing
        # NavigationToolbar2.forward = do_nothing
        # NavigationToolbar2.back = do_nothing
        # NavigationToolbar2.zoom = do_nothing

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
        self.objects['Figure'].canvas.mpl_connect(
            'close_event', self._on_close)



    def _create_segments(self):
        """Create the segments plots in the player's figure."""
        if self.segments is not None:
            for segment in self.segments:
                self.objects['PlotSegments'][segment] = \
                    self.objects['Axes'].plot(
                        np.nan, np.nan, '-',
                        c=self.segments[segment]['Color'],
                        linewidth=self.segment_width)[0]

    def _create_markers(self):
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
                c=colors[color], markersize=4, picker=5)[0]
        for color in self._colors:
            self.objects['PlotMarkers'][color + 's'] = \
                self.objects['Axes'].plot(
                    np.nan, np.nan, '.',
                    c=colors[color], markersize=12)[0]

    def _create_ground_plane(self):
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

    def _first_refresh(self):
        """Draw the stuff and set the axis size."""
        self._update_plots()
        plt.axis([-1.5, 1.5, -1, 1])

    def _get_projection(self, points_3d):
        """
        Get a 3d --> 2d projection of a list of points.

        The method uses the class's camera variables to project a list of
        3d points onto a 2d canvas.

        Parameters
        ----------
        points_3d : array
            Nx4 array, where the first dimension is the number of points
            and  the second dimension is (x, y, z, 1).

        Returns
        -------
        points_2d : array
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

    def _update_markers_and_segments(self):
        # Get a Nx4 matrices of every marker at the current frame
        markers = self.markers
        if markers is None:
            return
        else:
            n_markers = len(markers.data)

        markers_data = dict()  # Used to draw the markers with different colors
        segment_markers = dict()  # Used to draw the segments

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

                these_coordinates = markers.data[marker][self.current_frame]
                markers_data[color][i_marker] = these_coordinates
                segment_markers[marker] = these_coordinates

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

        # Draw the segments
        if self.segments is not None:
            for segment in self.segments:
                n_links = len(self.segments[segment]['Links'])
                coordinates = np.empty((3 * n_links, 4))
                coordinates[:] = np.nan
                for i_link in range(n_links):
                    marker1 = self.segments[segment]['Links'][i_link][0]
                    marker2 = self.segments[segment]['Links'][i_link][1]
                    if marker1 in segment_markers:
                        coordinates[3 * i_link] = segment_markers[marker1]
                    else:
                        coordinates[3 * i_link] = np.nan

                    if marker2 in segment_markers:
                        coordinates[3 * i_link + 1] = segment_markers[marker2]
                    else:
                        coordinates[3 * i_link + 1] = np.nan

                coordinates = self._get_projection(coordinates)

                self.objects['PlotSegments'][segment].set_data(
                    coordinates[:, 0], coordinates[:, 1])

    def _update_plots(self):
        """Update the plots, or draw it if not plot has been drawn before."""
        self._update_markers_and_segments()

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
                np.array([self.rigid_body_length, 0, 0, 1]))
            rby_data[i_rigid_body * 3 + 1] = (
                rigid_bodies.data[rigid_body][self.current_frame] @
                np.array([0, self.rigid_body_length, 0, 1]))
            rbz_data[i_rigid_body * 3 + 1] = (
                rigid_bodies.data[rigid_body][self.current_frame] @
                np.array([0, 0, self.rigid_body_length, 1]))
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
                linewidth=self.rigid_body_width)[0]
            self.objects['PlotRigidBodiesY'] = self.objects['Axes'].plot(
                rby_data[:, 0], rby_data[:, 1], c='g',
                linewidth=self.rigid_body_width)[0]
            self.objects['PlotRigidBodiesZ'] = self.objects['Axes'].plot(
                rbz_data[:, 0], rbz_data[:, 1], c='b',
                linewidth=self.rigid_body_width)[0]
        else:  # Update the plot
            self.objects['PlotRigidBodiesX'].set_data(
                rbx_data[:, 0], rbx_data[:, 1])
            self.objects['PlotRigidBodiesY'].set_data(
                rby_data[:, 0], rby_data[:, 1])
            self.objects['PlotRigidBodiesZ'].set_data(
                rbz_data[:, 0], rbz_data[:, 1])


        # Update the window title
        self.objects['Figure'].canvas.set_window_title(
            f'Frame {self.current_frame}, ' +
            '%2.2f s.' % self.time[self.current_frame])

        self.objects['Axes'].axis([-1.5, 1.5, -1, 1])
        self.objects['Figure'].canvas.draw()

    def _set_new_target(self, target):
        """Set new target and adapts translation and zoom consequently."""
        if np.sum(np.isnan(target)) > 0:
            return
        initial_translation = copy.deepcopy(self.translation)
        initial_zoom = copy.deepcopy(self.zoom)
        initial_target = copy.deepcopy(self.target)

        n_markers = len(self.markers.data)
        markers = np.empty((n_markers, 4))
        for i_marker, marker in enumerate(self.markers.data):
            markers[i_marker] = self.markers.data[marker][self.current_frame]

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
    def _set_frame(self, frame):
        """Set current frame to a given frame and update plots."""
        if frame >= self.n_frames:
            self.current_frame = self.n_frames - 1
        elif frame < 0:
            self.current_frame = 0
        else:
            self.current_frame = frame

        if (self.track is True and
                self.markers is not None):
            new_target = self.markers.data[
                self.last_selected_marker][self.current_frame]
            if not np.isnan(np.sum(new_target)):
                self.target = new_target

    def _set_time(self, time):
        """Set current frame to a given time and update plots."""
        index = np.argmin(np.abs(self.time - time))
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

    def close(self):
        """Close the Player and its associated window."""
        plt.close(self.objects['Figure'])
        self.objects = None

    # ------------------------------------
    # Callbacks
    def _on_close(self, _):
        # Release all references to objects
        self.close()

    def _on_timer(self, _):
        if self.running is True:
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
                self._set_frame(self.current_frame + 1)
            self.state['SystemTimeOnLastUpdate'] = time.time()

            self._update_plots()
        else:
            self.anim.event_source.stop()

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

            # Set as new target
            self.last_selected_marker = selected_marker
            self._set_new_target(
                self.markers.data[selected_marker][self.current_frame])
            marker_position = self.markers.data[selected_marker][
            self.current_frame]

            self._update_plots()

    def _on_key(self, event):
        """Callback for keyboard key pressed."""
        if event.key == ' ':
            if self.running is False:
                self.state['SystemTimeOnLastUpdate'] = time.time()
                self.state['SelfTimeOnPlay'] = self.time[self.current_frame]
                self.running = True
                self.anim.event_source.start()
            else:
                self.running = False
                self.anim.event_source.stop()

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

        if self.last_selected_marker is not None:
            self._set_new_target(
                self.markers.data[
                    self.last_selected_marker][self.current_frame])

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

