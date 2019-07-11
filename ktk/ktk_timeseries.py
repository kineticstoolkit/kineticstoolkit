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
from copy import deepcopy


class TimeSeries():
    """
    A class that implements timeseries.

    This class implements a timeseries in a way that resembles the timeseries
    and tscollection found in Matlab.

    Attributes
    ----------
    time : 1d np.array
        A time vector common to all data of the timeseries.
    data : dict
        Dictionary of np.arrays containing the timeseries data.
    time_unit : str
        The time unit. The default is 's'.
    data_unit : dict
        Dictionary of strings containing the data units.
    events : list of ktk.Event
        A list of events.
    """

    def __init__(
            self, time=np.array([]), data=dict(), time_unit='s',
            data_unit=dict(), events=[]):
        self.time = np.array(time)
        self.data = data
        self.time_unit = time_unit
        self.data_unit = data_unit
        self.events = events

    def __str__(self):
        """
        Return the string representation of the timeseries.

        Returns
        -------
        str: The string representation of the timeseries.

        """
        str_out = 'TimeSeries'
        str_out += '\n  time: array of shape ' + str(np.shape(self.time))
        str_out += "\n  time_unit: '" + str(self.time_unit) + "'"
        str_out += '\n  data:'
        for key in self.data.keys():
            str_out += ('\n    ' + str(key) + ': array of shape '
                        + str(np.shape(self.data[key])))

        str_out += '\n  data_unit: '
        for key in self.data_unit.keys():
            str_out += ('\n    ' + str(key) + ": '"
                        + str(self.data_unit[key]) + "'")

        str_out += '\n  events: ' + str(len(self.events))

        return(str_out)

    def __repr__(self):
        """
        Return the string representation of the timeseries.

        This method simply calls __str__(self).

        Returns
        -------
        str: The string representation of the timeseries.

        """
        return(str(self))

    def copy(self):
        """
        Return a deep copy of the timeseries.

        Usage
        -----
        >>> ts1 = ktk.TimeSeries(...)
        >>> ts2 = ts1.copy()

        Returns
        -------
        TimeSeries: An identical, deep copy of the timeseries where each
        original element has been copied so that no reference exists between
        the original and returned timeseries.
        """
        # TODO Unit test
        return deepcopy(self)


    def save(self, file_name): #TODO, still not what I want.
        np.save(file_name, self, allow_pickle=True)

    def load(file_name): #TODO, still not what I want.
        temp = np.load(file_name, allow_pickle=True)
        temp = temp.tolist()
        return(temp)

    def plot(self):

        #How many plots
        thekeys = self.keys()
        nplots = 0
        for thekey in thekeys:
            if thekey != 'Info' and thekey != 'Time' and thekey[0] != '_':
                nplots = nplots + 1

        #Now plot
        iplot = 1
        for thekey in thekeys:

            if thekey != 'Info' and thekey != 'Time' and thekey[0] != '_':

                if iplot == 1:
                    ax = plt.subplot(nplots,1,iplot)
                else:
                    plt.subplot(nplots,1,iplot, sharex=ax)

                plt.plot(self.Time, self[thekey], linewidth=1)

                if (thekey + 'Unit') in self.Info:
                    plt.ylabel(thekey + ' (' + self.Info[thekey + 'Unit'] + ')')
                else:
                    plt.ylabel(thekey)

                iplot = iplot+1

        plt.xlabel('Time (' + self.Info['TimeUnit'] + ')')
        plt.tight_layout()
        plt.show()

