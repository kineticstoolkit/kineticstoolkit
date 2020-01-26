"""
kinematics module for KTK.

Work in progress
"""

# import os.path
import numpy as np
import ktk
import external.pyc3d.c3d as c3d
import warnings
import subprocess
from time import sleep
from datetime import datetime
from ezc3d import c3d as ezc3d
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import struct  # To unpack data from N3D files


def read_c3d_file(filename):
    """
    Read a C3D file.

    Parameters
    ----------
    filename : str
        Path of the C3D file

    Returns
    -------
    A TimeSeries where each point in the C3D correspond to a data entry in
    the TimeSeries.
    """
    # Create the reader
    reader = ezc3d(filename)

    # Create the output timeseries
    output = ktk.TimeSeries()

    # Get the marker label names and create a timeseries data entry for each
    # Get the labels
    labels = reader['parameters']['POINT']['LABELS']['value']
    n_frames = reader['parameters']['POINT']['FRAMES']['value'][0]
    point_rate = reader['parameters']['POINT']['RATE']['value'][0]
    point_unit = reader['parameters']['POINT']['UNITS']['value'][0]

    if point_unit == 'mm':
        point_factor = 0.001
    elif point_unit == 'm':
        point_factor = 1
    else:
        raise(ValueError("Point unit must be 'm' or 'mm'."))

    n_labels = len(labels)
    for i_label in range(n_labels):
        # Strip leading and ending spaces
        labels[i_label] = labels[i_label].strip()

        label_name = labels[i_label]

        output.data[label_name] = (point_factor *
                                   reader['data']['points'][:, i_label, :].T)

        output.add_data_info(label_name, 'Unit', 'm')

    # Create the timeseries time vector
    output.time = np.arange(n_frames) / point_rate

    return output


def write_c3d_file(filename, ts):
    """Write a C3D file based on a timeseries of markers."""
    # Convert the timeseries data to a large 3d array:
    # First dimension = marker
    # Second dimension = time
    # Third dimension = x y z -visible -visible
    labels = sorted(list(ts.data.keys()))
    n_labels = len(labels)
    n_frames = len(ts.time)
    point_rate = 1 / (ts.time[1] - ts.time[0])

    all_points = np.zeros([n_labels, n_frames, 5])

    for i_label in range(len(labels)):
        the_label = labels[i_label]

        points = ts.data[the_label].copy()

        # Multiply by -1000 to get mm with correct scaling.
        points = np.block([-1000 * points, np.ones([n_frames, 1])])
        # Set -1 to fourth and fifth dimensions on missing samples
        nan_indices = ts.isnan(the_label)
        points[nan_indices, 3:5] = -1
        points[~nan_indices, 3:5] = 0
        # Remove nans
        points[nan_indices, 0:3] = 0

        all_points[i_label, :, :] = points

    # Create the c3d file
    writer = c3d.Writer(point_rate=point_rate)
    for i_frame in range(n_frames):
        writer.add_frames([(all_points[:, i_frame, :], np.array([[]]))])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        with open(filename, 'wb') as fid:
            writer.write(fid, labels)


def read_n3d_file(filename, labels=[]):
    """
    Read an Optitrak N3D file.

    Parameters
    ----------
    filename : str
        Path of the N3D file.
    labels : list of str (optional)
        Marker names

    Returns
    -------
    A TimeSeries where each point in the N3D correspond to a data entry in
    the TimeSeries.
    """
    with open(filename, 'rb') as fid:
        _ = fid.read(1)  # 32
#        n_markers = int.from_bytes(fid.read(1), 'big')
        n_markers = struct.unpack('h', fid.read(2))[0]
        n_data_per_marker = struct.unpack('h', fid.read(2))[0]
        n_columns = n_markers * n_data_per_marker

        n_frames = struct.unpack('i', fid.read(4))[0]

        collection_frame_frequency = struct.unpack('f', fid.read(4))[0]
        user_comments = struct.unpack('60s', fid.read(60))[0]
        system_comments = struct.unpack('60s', fid.read(60))[0]
        file_description = struct.unpack('30s', fid.read(30))[0]
        cutoff_filter_frequency = struct.unpack('h', fid.read(2))[0]
        time_of_collection = struct.unpack('8s', fid.read(8))[0]
        _ = fid.read(2)
        date_of_collection = struct.unpack('8s', fid.read(8))[0]
        extended_header = struct.unpack('73s', fid.read(73))[0]

        # Read the rest and put it in an array
        ndi_array = np.ones((n_frames, n_columns)) * np.NaN

        for i_frame in range(n_frames):
            for i_column in range(n_columns):
                data = struct.unpack('f', fid.read(4))[0]
                if (data < -1E25):  # technically, it is -3.697314e+28
                    data = np.NaN
                ndi_array[i_frame, i_column] = data

        # Conversion from mm to meters
        ndi_array /= 1000

        # Transformation to a TimeSeries
        ts = ktk.TimeSeries(
                time=np.linspace(0, n_frames / collection_frame_frequency,
                                 n_frames))

        for i_marker in range(n_markers):
            if labels != []:
                label = labels[i_marker]
            else:
                label = f'Marker{i_marker}'

            ts.data[label] = np.block([[
                    ndi_array[:, 3*i_marker:3*i_marker+3],
                    np.ones((n_frames, 1))]])

    return ts


def plot3d(markers=None, rigid_bodies=None, segments=None,
           sample=0, marker_radius=0.008, rigid_body_size=0.1):
    """Plot a TimeSeries of markers and/or rigid bodies on a 3D MPL axis."""
    markers = markers.copy()  # Since we add stuff to it.

    # Plot every marker at a given index
    x = []
    y = []
    z = []
    min_coordinate = 99999999.
    max_coordinate = -99999999.

    for data in markers.data:
        temp_data = markers.data[data]
        x.append(temp_data[sample, 0])
        y.append(temp_data[sample, 2])
        z.append(temp_data[sample, 1])
        min_coordinate = np.min([min_coordinate, np.nanmin(temp_data[:, 0])])
        min_coordinate = np.min([min_coordinate, np.nanmin(temp_data[:, 1])])
        min_coordinate = np.min([min_coordinate, np.nanmin(temp_data[:, 2])])
        max_coordinate = np.max([max_coordinate, np.nanmax(temp_data[:, 0])])
        max_coordinate = np.max([max_coordinate, np.nanmax(temp_data[:, 1])])
        max_coordinate = np.max([max_coordinate, np.nanmax(temp_data[:, 2])])

    # Create the 3d figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, s=marker_radius*1000, c='b')

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


    # ax.scatter(
    #     [min_coordinate, min_coordinate, min_coordinate, min_coordinate,
    #       max_coordinate, max_coordinate, max_coordinate, max_coordinate],
    #     [min_coordinate, min_coordinate, max_coordinate, max_coordinate,
    #       min_coordinate, min_coordinate, max_coordinate, max_coordinate],
    #     [min_coordinate, max_coordinate, min_coordinate, max_coordinate,
    #       min_coordinate, max_coordinate, min_coordinate, max_coordinate],
    #     s=1E-6)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')


def open_in_mokka(markers=None, rigid_bodies=None, segments=None,
                  marker_radius=0.008, rigid_body_size=0.1):
    """Open a timeseries of markers for 3D visualization in Mokka."""
    if markers is None:
        markers = ktk.TimeSeries(time=rigid_bodies.time)
    else:
        markers = markers.copy()  # Since we add stuff to it.

    # Set the filenames
    base_filename = str(datetime.now().strftime('%H:%M:%S'))
    c3d_filename = base_filename + '.c3d'
    mvc_filename = base_filename + '.mvc'

    # Open the mvc filename
    fid = open(mvc_filename, 'w')

    # Add the markers information to the mcv file
    print('<?xml version="1.0" encoding="UTF-8"?>', file=fid)
    print('<MokkaModelVisualConfiguration name="KTK" '
          'version="1.0">',
          file=fid)

    # Add the real markers list to the mvc file
    print('<MarkersList>', file=fid)
    for marker_name in markers.data.keys():
        print(f'<Marker label="{marker_name}" radius="{marker_radius*1000}" '
              f'colorR="1" colorG="1" colorB="1" visible="1" '
              f'trajectory="0"/>', file=fid)

    # Add the rigid bodies 'markers' to the markers TimeSeries and to the mvc
    # file
    if rigid_bodies is not None:
        for rb_name in rigid_bodies.data.keys():
            marker_name = '_' + rb_name + '_o'
            markers.data[marker_name] = rigid_bodies.data[rb_name][:, :, 3]
            print(f'<Marker label="{marker_name}" '
                  f'radius="{marker_radius*1000}" '
                  f'colorR="1" colorG="1" colorB="1" visible="1" '
                  f'trajectory="0"/>', file=fid)

            marker_name = '_' + rb_name + '_x'
            markers.data[marker_name] = (
                    rigid_bodies.data[rb_name] @
                    np.array([rigid_body_size, 0, 0, 1]))
            print(f'<Marker label="{marker_name}" '
                  f'radius="{marker_radius*1000}" '
                  f'colorR="1" colorG="0" colorB="0" visible="0" '
                  f'trajectory="0"/>', file=fid)

            marker_name = '_' + rb_name + '_y'
            markers.data[marker_name] = (
                    rigid_bodies.data[rb_name] @
                    np.array([0, rigid_body_size, 0, 1]))
            print(f'<Marker label="{marker_name}" '
                  f'radius="{marker_radius*1000}" '
                  f'colorR="0" colorG="1" colorB="0" visible="0" '
                  f'trajectory="0"/>', file=fid)

            marker_name = '_' + rb_name + '_z'
            markers.data[marker_name] = (
                    rigid_bodies.data[rb_name] @
                    np.array([0, 0, rigid_body_size, 1]))
            print(f'<Marker label="{marker_name}" '
                  f'radius="{marker_radius*1000}" '
                  f'colorR="0" colorG="0" colorB="1" visible="0" '
                  f'trajectory="0"/>', file=fid)

    print('</MarkersList>', file=fid)

    # Add the rigid body 'segments' to the mvc file
    print('<SegmentsList>', file=fid)
    if rigid_bodies is not None:
        for rb_name in rigid_bodies.data.keys():

            # Segment x
            print(f'<Segment label="{"_" + rb_name + "_x"}" '
                  f'colorR="1" colorG="0" colorB="0" visible="1" surface="1" '
                  f'description="">', file=fid)
            print(f'<Point label="{"_" + rb_name + "_o"}"/>', file=fid)
            print(f'<Point label="{"_" + rb_name + "_x"}"/>', file=fid)
            print(f'<Link pt1="{"_" + rb_name + "_o"}" '
                  f'pt2="{"_" + rb_name + "_x"}"/>', file=fid)
            print(f'</Segment>', file=fid)

            # Segment y
            print(f'<Segment label="{"_" + rb_name + "_y"}" '
                  f'colorR="0" colorG="1" colorB="0" visible="1" surface="1" '
                  f'description="">', file=fid)
            print(f'<Point label="{"_" + rb_name + "_o"}"/>', file=fid)
            print(f'<Point label="{"_" + rb_name + "_y"}"/>', file=fid)
            print(f'<Link pt1="{"_" + rb_name + "_o"}" '
                  f'pt2="{"_" + rb_name + "_y"}"/>', file=fid)
            print(f'</Segment>', file=fid)

            # Segment z
            print(f'<Segment label="{"_" + rb_name + "_z"}" '
                  f'colorR="0" colorG="0" colorB="1" visible="1" surface="1" '
                  f'description="">', file=fid)
            print(f'<Point label="{"_" + rb_name + "_o"}"/>', file=fid)
            print(f'<Point label="{"_" + rb_name + "_z"}"/>', file=fid)
            print(f'<Link pt1="{"_" + rb_name + "_o"}" '
                  f'pt2="{"_" + rb_name + "_z"}"/>', file=fid)
            print(f'</Segment>', file=fid)

    print('</SegmentsList>', file=fid)

    # Close the mvc file
    print('</MokkaModelVisualConfiguration>', file=fid)
    fid.close()

    # Create the c3d file
    c3d_filename = str(datetime.now().strftime('%H:%M:%S')) + '.c3d'
    write_c3d_file(c3d_filename, markers)

    # Open Mokka in background, then delete these temporary files when Mokka
    # closes.
    print('Opening Mokka...')
    if ktk.config['IsMac'] is True:
        subprocess.call(('bash -c '
                         f'"{ktk.config["RootFolder"]}/external/mokka/'
                         f'macos/Mokka.app/Contents/'
                         f'MacOS/Mokka {c3d_filename} ; '
                         f'rm {c3d_filename}; rm {mvc_filename}; " &'),
                        shell=True)
        # Try to activate it.
        sleep(1)
        while subprocess.call(['osascript', '-e',
                               'tell application "Mokka" to activate'],
                              stderr=subprocess.DEVNULL) == 1:
            sleep(0.5)

    else:
        raise NotImplementedError('Only implemented on Mac for now.')


# def read_xml_file(file_name):
#
#    # Define a helper function that reads the file line by line until it finds
#    # one of the strings in a given list.
#    def read_until(list_strings):
#        while True:
#            one_line = fid.readline()
#            if len(one_line) == 0:
#                return(one_line)
#
#            for one_string in list_strings:
#                if one_line.find(one_string) >= 0:
#                    return(one_line)
#
#
#    # Open the file
#    if os.path.isfile(file_name) == False:
#        raise(FileNotFoundError)
#
#    fid = open(file_name, 'r')
#
#    # Reading loop
#    the_timeseries = []
#
#    while True:
#
#        # Wait for next label
#        one_string = read_until(['>Label :</Data>'])
#
#        if len(one_string) > 0:
#            # A new label was found
#
#            # Isolate the label name
#            one_string = read_until(['<Data'])
#            label_name = one_string[
#                    (one_string.find('"String">')+9):one_string.find('</Data>')]
#            print(label_name)
#
#            # Isolate the data format
#            one_string = read_until(['>Coords'])
#            one_string = one_string[
#                    (one_string.find('<Data>')+6):one_string.find('</Data>')]
#            data_unit = one_string[
#                    (one_string.find('x,y:')+4):one_string.find('; ')]
#
#            # Ignore the next data lines (header)
#            one_string = read_until(['<Data'])
#            one_string = read_until(['<Data'])
#            one_string = read_until(['<Data'])
#
#            # Find all data for this marker
#            time = np.array([1.0]) # Dummy init
#            data = np.zeros((1,2)) # Dummy init
#            sample_index = 0
#
#            while(True):
#
#                # Find the next x data
#                one_string = read_until(['<Data'])
#                one_string = one_string[
#                        (one_string.find('"Number">')+9):one_string.find('</Data>')]
#
#                try:
#                    # If it's a float, then add it.
#                    # Add a new row to time and data
#                    if sample_index > 0:
#                        time = np.block([time, np.array(1)])
#                        data = np.block([[data], [np.zeros((1,2))]])
#
#                    data[sample_index, 0] = float(one_string)
#
#                except:
#                    the_timeseries.append(ts.TimeSeries(name=label_name, data=data, time=time, dataunit=data_unit, timeunit='s'))
#                    break #No data
#
#                # Find the next y data
#                one_string = read_until(['<Data'])
#                one_string = one_string[
#                        (one_string.find('"Number">')+9):one_string.find('</Data>')]
#                data[sample_index, 1] = float(one_string)
#
#                # Find the next t data
#                one_string = read_until(['<Data'])
#                one_string = one_string[
#                        (one_string.find('">')+2):one_string.find('</Data>')]
#
#                if one_string.find(':') < 0:
#                    time[sample_index] = float(one_string) # milliseconds or #frame
#                else:
#                    index = one_string.find(':')
#                    hours = one_string[0:index]
#                    one_string = one_string[index+1:]
#
#                    index = one_string.find(':')
#                    minutes = one_string[0:index]
#                    one_string = one_string[index+1:]
#
#                    index = one_string.find(':')
#                    seconds = one_string[0:index]
#                    milliseconds = one_string[index+1:]
#
#                    time[sample_index] = (3600. * float(hours) +
#                        60. * float(minutes) + float(seconds) +
#                        (int(milliseconds) % 1000) / 1000)
#
#
#                sample_index = sample_index + 1
#
#        else:
#            # EOF
#            return(the_timeseries)

def create_rigid_body_config(markers, marker_names):
    """
    Create a rigid body configuration based on a static acquisition.

    Parameters
    ----------
    markers : TimeSeries
        Markers trajectories during the static acquisition.
    marker_names : list of str
        The markers that define the rigid body.

    Returns
    -------
    dict with the following keys:
        - MarkerNames : the same as marker_names
        - LocalPoints : a 1x4xM array that indicates the local position of
                        each M marker in the created rigid body config.
    """
    n_samples = len(markers.time)
    n_markers = len(marker_names)

    # Construct the global points array
    global_points = np.empty((n_samples, 4, n_markers))

    for i_marker, marker in enumerate(marker_names):
        global_points[:, :, i_marker] = \
                markers.data[marker][:, :]

    # Remove samples where at least one marker is invisible
    global_points = global_points[~ktk.geometry.isnan(global_points)]

    rigid_bodies = ktk.geometry.create_reference_frames(global_points)
    local_points = ktk.geometry.get_local_coordinates(
            global_points, rigid_bodies)

    # Take the average
    local_points = np.mean(local_points, axis=0)
    local_points = local_points[np.newaxis]

    return {
            'LocalPoints' : local_points,
            'MarkerNames' : marker_names
            }


def register_markers(markers, rigid_body_configs, verbose=True):
    """
    Compute the rigid body trajectories using ktk.geometry.register_points.

    Parameters
    ----------
    markers : TimeSeries
        Markers trajectories to calculate the rigid body trajectories on.
    rigid_body_configs : dict of dict
        A dict where each key is a rigid body configuration, and where
        each rigid body configuration is a dict with the following
        keys: 'MarkerNames' and 'LocalPoints'.
    verbose : bool (optional)
        True to print the rigid body being computed. Default is True.

    Returns
    -------
    TimeSeries where each data key is a Nx4x4 series of rigid transformations.
    """
    rigid_bodies = ktk.TimeSeries(time=markers.time,
                                  time_info=markers.time_info,
                                  events=markers.events)

    for rigid_body_name in rigid_body_configs:
        if verbose is True:
            print(f'Computing trajectory of rigid body {rigid_body_name}...')

            # Set local and global points
            local_points = rigid_body_configs[rigid_body_name]['LocalPoints']

            global_points = np.empty(
                    (len(markers.time), 4, local_points.shape[2]))
            for i_marker in range(global_points.shape[2]):
                marker_name = rigid_body_configs[
                        rigid_body_name]['MarkerNames'][i_marker]
                global_points[:, :, i_marker] = markers.data[marker_name]

            (local_points, global_points) = ktk.geometry.match_size(
                    local_points, global_points)

            # Compute the rigid body trajectory
            rigid_bodies.data[rigid_body_name] = ktk.geometry.register_points(
                    global_points, local_points)

    return rigid_bodies


def create_virtual_marker_config(markers, rigid_body_name, rigid_body_config,
                                 probe_tip_label):
    """
    Create a virtual marker configuration based on a probing acquisition.

    Parameters
    ----------
    markers : TimeSeries
        Markers trajectories during the probing acquisition.
    rigid_body_name : str
        Name of the virtual marker's rigid body.
    rigid_body_config : dict
        Configuration of the rigid body. This dict must contain the keys
        'MarkerNames' and 'LocalPoints'.
    probe_tip_label : str
        Name of the marker that corresponds to the probe tip.

    Returns
    -------
    dict with the following keys:
        RigidBodyName : Name of the virtual marker's rigid body
        LocalPoint : Local position of this marker in the reference frame
                     defined by the rigid body RigidBodyName. LocalPoint is
                     expressed as a 1x4 array.
    """
    pass
    """
    % Créer une structure simplifiée de définition de corps rigides, pour
    % accélérer le registering.
    simpleDefRigidBodies.(rigidBodyName) = defRigidBodies.(rigidBodyName);
    simpleDefRigidBodies.(probeName) = defRigidBodies.(probeName);

    % Déterminer la liste de marqueurs dont on a besoin
    defReferenceRigidBody = defRigidBodies.(rigidBodyName);
    defProbe = defRigidBodies.(probeName);
    simpleMarkerNames = [defReferenceRigidBody.MarkerNames defProbe.MarkerNames];

    % Faire un subset de markers
    nMarkers = length(simpleMarkerNames);
    for iMarker = 1:nMarkers
        theMarker = simpleMarkerNames{iMarker};
        simpleMarkers.(theMarker) = markers.(theMarker);
    end

    % Remplir les trous jusqu'à un quart de seconde
    time = simpleMarkers.(theMarker).Time;
    fech = 1/(time(2) - time(1));
    simpleMarkers = ktkTimeSeries.fillmissingsamples(simpleMarkers, 0.25 * fech);

    % Statifier l'essai (ne conserver que la moyenne des samples où tous ces
    % marqueurs sont visibles
    doubleMarkers = ktkKinematics.meanstaticmarkers(simpleMarkers);

    if isempty(doubleMarkers)
        % Les marqueurs nécessaires ne sont jamais visibles en même temps. Tenter
        % une autre approche : rigidifier tout l'essai et reconstruire les
        % marqueurs nécessaires. Çe peut aider dans le cas où un corps rigide ou la
        % probe comporte plus de 3 marqueurs.

        rigidBodies = ktkKinematics.registermarkers(markers, simpleDefRigidBodies);

        % Reconstruire les marqueurs du corps rigide
        globalMarkers = ktkGeometry.getglobalcoordinates(...
            simpleDefRigidBodies.(rigidBodyName).LocalPoints, rigidBodies.(rigidBodyName));

        theMarkerNames = defReferenceRigidBody.MarkerNames;
        for iMarker = 1:length(theMarkerNames)
            theMarkerName = theMarkerNames{iMarker};
            simpleMarkers.(theMarkerName).Data = globalMarkers.Data(:,iMarker,:);
        end

        % Reconstruire les marqueurs de la probe
        globalMarkers = ktkGeometry.getglobalcoordinates(...
            simpleDefRigidBodies.(probeName).LocalPoints, rigidBodies.(probeName));

        theMarkerNames = defProbe.MarkerNames;
        for iMarker = 1:length(theMarkerNames)
            theMarkerName = theMarkerNames{iMarker};
            simpleMarkers.(theMarkerName).Data = globalMarkers.Data(:,iMarker,:);
        end


        % Statifier l'essai (ne conserver que la moyenne des samples où tous ces
        % marqueurs sont visibles
        doubleMarkers = ktkKinematics.meanstaticmarkers(simpleMarkers);

    end

    if isempty(doubleMarkers) % Est-ce que la reconstruction des marqueurs manquants a réussi ?

        warning('La tentative de reconstruction des marqueurs manquant a échoué. Impossible de définir ce marqueur virtuel.');
        % Créer la structure de sortie
        defVirtualMarker.RigidBodyName = rigidBodyName;
        defVirtualMarker.LocalPoint = [NaN; NaN; NaN; 1];

    else

        % Register
        rigidBodies = ktkKinematics.registermarkers(doubleMarkers, simpleDefRigidBodies);

        % Trouver la position relative de la pointe de la probe
        REF = rigidBodies.(rigidBodyName);
        Probe = rigidBodies.(probeName);
        position = ktkGeometry.getlocalcoordinates(Probe, REF);

        % Créer la structure de sortie
        defVirtualMarker.RigidBodyName = rigidBodyName;
        defVirtualMarker.LocalPoint = position(:,4);

    end

end
    """
    return None
