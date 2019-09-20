"""
kinematics module for KTK

Work in progress
"""

# import os.path
import numpy as np
import ktk
import external.pyc3d.c3d as c3d
import warnings


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
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        fid = open(filename, 'rb')
        reader = c3d.Reader(fid)

    # Create the output timeseries
    output = ktk.TimeSeries()

    # Get the marker label names and create a timeseries data entry for each
    labels = reader.point_labels.copy()
    n_labels = len(labels)
    for i in range(n_labels):
        labels[i] = labels[i].strip()  # Strip leading and ending spaces
        # Begin with an increasing list, then convert to an array at the end.
        output.data[labels[i]] = []
        output.add_data_info(labels[i], 'Unit', 'm')

    # Read each frame and append to the timeseries
    n_frames = 0
    for one_frame in reader.read_frames():
        n_frames += 1
        points = one_frame[1]
        for i_label in range(n_labels):
            output.data[labels[i_label]].append(points[i_label])

    # Convert the timeseries data to 2-dimension arrays
    # with nans as missing samples
    for label in labels:
        output.data[label] = np.array(output.data[label])

        # Find missing samples
        nan_index = np.nonzero(output.data[label][:, 3])
        # Keep only x,y,z
        output.data[label] = output.data[label][:, 0:3]
        # Add ones to 4th element
        output.data[label] = np.block([output.data[label],
                                      np.ones([np.shape(
                                              output.data[label])[0], 1])])
        # Fill missing samples with nans
        output.data[label][nan_index, :] = np.nan

    # Creating the timeseries time vector
    output.time = np.arange(n_frames) / reader.point_rate

    # Close and return
    fid.close()
    return output


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
