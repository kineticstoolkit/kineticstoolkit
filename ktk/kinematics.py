#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:29:38 2019

@author: felix
"""

import os.path
import numpy as np
import timeseries as ts

def read_xml_file(file_name):
    
    # Define a helper function that reads the file line by line until it finds
    # one of the strings in a given list.
    def read_until(list_strings):
        while True:
            one_line = fid.readline()
            if len(one_line) == 0:
                return(one_line)
            
            for one_string in list_strings:
                if one_line.find(one_string) >= 0:
                    return(one_line)
        
    
    # Open the file
    if os.path.isfile(file_name) == False:
        raise(FileNotFoundError)

    fid = open(file_name, 'r')
    
    # Reading loop
    the_timeseries = []
    
    while True:
        
        # Wait for next label
        one_string = read_until(['>Label :</Data>'])
        
        if len(one_string) > 0:
            # A new label was found
            
            # Isolate the label name
            one_string = read_until(['<Data'])
            label_name = one_string[
                    (one_string.find('"String">')+9):one_string.find('</Data>')]
            print(label_name)
            
            # Isolate the data format
            one_string = read_until(['>Coords'])
            one_string = one_string[
                    (one_string.find('<Data>')+6):one_string.find('</Data>')]
            data_unit = one_string[
                    (one_string.find('x,y:')+4):one_string.find('; ')]
            
            # Ignore the next data lines (header)
            one_string = read_until(['<Data'])
            one_string = read_until(['<Data'])
            one_string = read_until(['<Data'])
            
            # Find all data for this marker
            time = np.array([1.0]) # Dummy init
            data = np.zeros((1,2)) # Dummy init
            sample_index = 0
            
            while(True):
                
                # Find the next x data
                one_string = read_until(['<Data'])
                one_string = one_string[
                        (one_string.find('"Number">')+9):one_string.find('</Data>')]
                
                try:
                    # If it's a float, then add it.                    
                    # Add a new row to time and data
                    if sample_index > 0:                        
                        time = np.block([time, np.array(1)])                        
                        data = np.block([[data], [np.zeros((1,2))]])
                        
                    data[sample_index, 0] = float(one_string)

                except:
                    the_timeseries.append(ts.TimeSeries(name=label_name, data=data, time=time, dataunit=data_unit, timeunit='s'))
                    break #No data             
                
                # Find the next y data
                one_string = read_until(['<Data'])
                one_string = one_string[
                        (one_string.find('"Number">')+9):one_string.find('</Data>')]
                data[sample_index, 1] = float(one_string)

                # Find the next t data
                one_string = read_until(['<Data'])
                one_string = one_string[
                        (one_string.find('">')+2):one_string.find('</Data>')]
                
                if one_string.find(':') < 0:
                    time[sample_index] = float(one_string) # milliseconds or #frame
                else:
                    index = one_string.find(':')
                    hours = one_string[0:index]
                    one_string = one_string[index+1:]
                    
                    index = one_string.find(':')
                    minutes = one_string[0:index]
                    one_string = one_string[index+1:]
                    
                    index = one_string.find(':')
                    seconds = one_string[0:index]
                    milliseconds = one_string[index+1:]                        
                    
                    time[sample_index] = (3600. * float(hours) + 
                        60. * float(minutes) + float(seconds) + 
                        (int(milliseconds) % 1000) / 1000)
                    

                sample_index = sample_index + 1                    
            
        else:
            # EOF
            return(the_timeseries)
    