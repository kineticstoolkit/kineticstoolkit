"""
kinematics module for KTK

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


def read_c3d_file(filename):
    """
    Read a C3D file

    Parameters
    ----------
    filename : str
        Path of the C3D file

    Returns
    -------
    ktk.TimeSeries where each point in the C3D correspond to a data entry in
    the timeseries.
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
    for i in range(n_labels):
        labels[i] = labels[i].strip()  # Strip leading and ending spaces
        # Begin with an increasing list, then convert to an array at the end.
        output.data[labels[i]] = (point_factor *
                                  reader['data']['points'][:, i, :].T)
        output.add_data_info(labels[i], 'Unit', 'm')

#    # Convert the timeseries data to 2-dimension arrays
#    # with nans as missing samples
#    for label in labels:
#        output.data[label] = np.array(output.data[label])
#
#        # Find missing samples
#        nan_index = np.nonzero(output.data[label][:, 3])
#        # Keep only x,y,z
#        output.data[label] = output.data[label][:, 0:3]
#        # Add ones to 4th element
#        output.data[label] = np.block([output.data[label],
#                                      np.ones([np.shape(
#                                              output.data[label])[0], 1])])
#        # Fill missing samples with nans
#        output.data[label][nan_index, :] = np.nan

    # Creating the timeseries time vector
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
