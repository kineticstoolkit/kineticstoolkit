"""
Module that manages the TimeSeries class.

timeseries
==========

This is a tentative of an implementation of Matlab's timeseries to help me see
if I'll rebuild KTK under Python.

Created on Thu Jun  6 11:07:32 2019

@author: Felix Chenier
"""

import matplotlib.pyplot as plt
import numpy as np
import collections

from copy import deepcopy as _deepcopy
import ktk.gui as _gui


class TimeSeriesEvent(list):
    """
    Define an event in a timeseries.

    Attributes
    ----------
    time : float
        The time at which the event happened.
    name : str
        The name of the event.
    """

    def __init__(self, time=0., name='event'):
        list.__init__(self)
        self.append(float(time))
        self.append(str(name))

    @property
    def time(self):
        return self[0]
    
    @time.setter
    def time(self, time):
        self[0] = float(time)
        
    @property
    def name(self):
        return self[1]
    
    @name.setter
    def name(self, name):
        self[0] = str(name)
        

#    def __str__(self):
#        """Return the string representation of the TimeSeriesEvent."""
#        return 'time: ' + str(self.time) + ', name: ' + str(self.name)
#
#    def __repr__(self):
#        """Return the string representation of the TimeSeriesEvent."""
#        return '<' + str(self.time) + ' ' + str(self.name) + '>'


class TimeSeries(dict):
    """
    A class that implements TimeSeries.

    This class implements a Timeseries in a way that resembles the timeseries
    and tscollection found in Matlab.
    
    The TimeSeries class is simply a dict with added methods and
    specifications:
        - It always has a 'time' key, which is a 1-dimension np.array
          containing the time entries.
        - It always has an 'info' key, which is a dict containing info on
          'time' and any other data entry.
          unit.
        - It always has an 'events' key, which is a list of TimeSeriesEvent.
          
    Example of creation:
    
        >>> ts = TimeSeries({time: np.array(range(0,100))})
    """

    def __init__(self, dict_entry={}):
        dict.__init__(self)
        self['time'] = np.array([])
        self['info'] = {'time': {'unit': 's'}}
        self['events'] = []
        for the_key in dict_entry.keys():
            self[the_key] = _deepcopy(dict_entry[the_key])


    def add_info(self, signal_name, info_name, value):
        """
        Add information on a signal of the TimeSeries.
        
        Examples of use:
            >>> the_timeseries.add_info('time', 'unit', 's')
            >>> the_timeseries.add_info('forces', 'unit', 'N')
            >>> the_timeseries.add_info('marker1', 'color', [43, 2, 255])
        
        This creates a corresponding entry in the 'info' dict.
        """
        try:
            self['info'][signal_name]  # Test if it exists
            self['info'][signal_name][info_name] = value  # Assign the value
            
        except:  # No info yet for this signal name
            self['info'][signal_name] = {info_name: value}  # Assign the value

    
    def remove_info(self, signal_name, info_name):
        """TODO"""
        raise NotImplementedError('This feature is not implemented yet')


    def add_event(self, time, name='event'):
        """
        Add an event to the TimeSeries.

        Parameters
        ----------
        time : float
            The time at which the event happened.
        name : str
            The name of the event.

        Returns
        -------
        self.

        This is a convenience function, the same can be reached by simply
        appending a TimeSeriesEvent to the TimeSeries' event list:

        >>> the_time_series['events'].append(TimeSeriesEvent(time, name))
        """
        self['events'].append(TimeSeriesEvent(time, name))
        return self


    def ui_add_events(self, name='event'):
        """
        Add one or many events interactively to the TimeSeries.

        Parameters
        ----------
        name : str
            The name of the event.

        Returns
        -------
        self.
        """
        self.plot()
        _gui.buttondialog(title='uiaddevents',
                          message=('Please zoom on the figure, then click '
                                   + 'Continue, or End to terminate.'),
                          choices=['Continue', 'End'])
        plt.suptitle(('Left-click to add events,\n'
                      + 'Right-click to remove last added events,\n'
                      + 'Enter to terminate.'))
        points = plt.ginput(1000)

        for the_point in points:
            self.addevent(time=the_point[0], name=name)

        # TODO Continue


    def save(self, file_name): #TODO, still not what I want.
        np.save(file_name, self, allow_pickle=True)


    def load(file_name): #TODO, still not what I want.
        temp = np.load(file_name, allow_pickle=True)
        temp = temp.tolist()
        return(temp)


    def plot(self):
        """Plot the TimeSeries using matplotlib."""

        plt.cla()
        the_keys = self.keys()
        for the_key in the_keys:
            if the_key != 'time' and isinstance(self[the_key], np.ndarray):
                plt.plot(self['time'], self[the_key])
                
        plt.xlabel('Time (' + self['time_unit'] + ')')
        
        return
    
        #How many plots
        thekeys = self.data.keys()
        nplots = len(thekeys)

        #Now plot
        iplot = 1
        for thekey in thekeys:

            if iplot == 1:
                ax = plt.subplot(nplots,1,iplot)
            else:
                plt.subplot(nplots,1,iplot, sharex=ax)

            plt.plot(self.time, self.data[thekey])

            if thekey in self.data_unit:
                plt.ylabel(thekey + ' (' + self.data_unit[thekey] + ')')
            else:
                plt.ylabel(thekey)

            iplot = iplot+1

        plt.xlabel('Time (' + self.time_unit + ')')
        plt.tight_layout()
        plt.show()

