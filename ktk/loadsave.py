#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 13:29:24 2019

@author: felix
"""

import scipy.io as spio
from ktk.timeseries import TimeSeries

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    data = _check_keys(data)
    return convert_to_timeseries(data)



def convert_to_timeseries(the_input):
    
    if isinstance(the_input, dict):
#c        print("This is a dict. Checking if it's a timeseries.")
        
        is_a_timeseries = False
        
        for the_key in the_input.keys():
            
            if isinstance(the_input[the_key], dict):
                if 'type' in the_input[the_key].keys():
                    if the_input[the_key]['type'] == 'timeseries':
                        is_a_timeseries = True
#                    else:
#                        is_a_timeseries = False
#                else:
#                    is_a_timeseries = False
#            else:
#                is_a_timeseries = False
        #end for the_key
                

        if is_a_timeseries == True:            
            #After checking if each key is a timeseries, and it is, we get here.
            the_output = TimeSeries()
            for the_key in the_input.keys():
                try:
                    the_output['time'] = the_input[the_key]['Time']
                    the_output[the_key] = the_input[the_key]['Data']
                    the_data = the_output[the_key]
                    the_shape = the_data.shape
                    if len(the_shape) == 2:
                        the_output[the_key] = the_data.transpose((1,0))
                    elif len(the_shape) == 3:
                        the_output[the_key] = the_data.transpose((2,0,1))
                    
                except:
                    pass
                
            return the_output
        else:
            print('This was not a timeseries.')
            
            for the_key in the_input.keys():
                print('  Now processing key %s' % the_key)
                the_input[the_key] = convert_to_timeseries(the_input[the_key])
            return the_input
        
    else:
        return the_input




def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict
