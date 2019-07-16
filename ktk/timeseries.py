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

from copy import deepcopy as _deepcopy
import ktk.gui as _gui


class TimeSeriesEvent():
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
        self.time = time
        self.name = name

    def __str__(self):
        """Return the string representation of the TimeSeriesEvent."""
        return 'time: ' + str(self.time) + ', name: ' + str(self.name)

    def __repr__(self):
        """Return the string representation of the TimeSeriesEvent."""
        return self.__str__()


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
    events : list of ktk.TimeSeriesEvent
        A list of events.
    """

    def __init__(self, time=np.array([]), data=dict(), time_unit='s',
                 data_unit=dict(), events=[]):
        self.time = np.array(time)
        self.data = data
        self.time_unit = time_unit
        self.data_unit = data_unit
        self.events = events

    def __str__(self):
        """
        Return the string representation of the TimeSeries.

        Returns
        -------
        str: The string representation of the TimeSeries.

        """
        str_out = 'TimeSeries'
        str_out += '\n  time:  array of shape' + str(np.shape(self.time))
        str_out += "\n  time_unit: '" + str(self.time_unit) + "'"
        str_out += '\n  data: (array of shape:)'
        for key in self.data.keys():
            str_out += ('\n    ' + str(key) + ': \t'
                        + str(np.shape(self.data[key])))

        str_out += '\n  data_unit: '
        for key in self.data_unit.keys():
            str_out += ('\n    ' + str(key) + ": '"
                        + str(self.data_unit[key]) + "'")

        str_out += '\n  events: list of length ' + str(len(self.events))

        return(str_out)

    def __repr__(self):
        """
        Return the string representation of the TimeSeries.

        This method simply calls __str__(self).

        Returns
        -------
        str: The string representation of the TimeSeries.

        """
        return(str(self))
        
    def _repr_pprint(self, p, cycle):
        if cycle:
            p.pretty("...")
        else:
            p.text("TimeSeries with time=")

    def copy(self):
        """
        Return a deep copy of the TimeSeries.

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
        return _deepcopy(self)

    def addevent(self, time=0.0, name='event'):
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

        >>> the_time_series.events.append(TimeSeriesEvent(time, name))
        """
        self.events.append(TimeSeriesEvent(time, name))
        return self

    def uiaddevents(self, name='event'):
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

