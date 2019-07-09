"""
TimeSeries
==========

This is a tentative of an implementation of Matlab's timeseries to help me see
if I'll rebuild KTK under Python.

Created on Thu Jun  6 11:07:32 2019

@author: Felix Chenier
"""

import matplotlib.pyplot as plt
import numpy as np
import ktkcore

class TimeSeries(ktkcore.ObjDict):
    """
    This class implements a timeseries in a close way to the Matlab's
    timeseries object.

    Parameters
    ----------
    name : String, optional
        Name of the timeseries. The default is ''.
    data : TYPE, optional
        DESCRIPTION. The default is [].
    time : TYPE, optional
        DESCRIPTION. The default is [].
    dataunit : TYPE, optional
        DESCRIPTION. The default is ''.
    timeunit : TYPE, optional
        DESCRIPTION. The default is 's'.
    
    Returns
    -------
    None.
    
    """

    def __init__(
            self, name='', data=[], time=[], dataunit='', timeunit='s'):
        if len(name) > 0: self[name] = data
        self.Time = time
        self.Info = ktkcore.ObjDict()
        self.Info.TimeUnit = timeunit
    
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

