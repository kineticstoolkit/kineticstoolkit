Kinetics Toolkit
================

.. note::
   This is the website for the development version. Switch to the stable_
   version.

.. _stable: http://felixchenier.uqam.ca/kineticstoolkit

Kinetics Toolkit (ktk) is an open-source, pure-python biomechanical library
developed by Professor Félix Chénier at Université du Québec à Montréal,
Canada. It is a package of integrated classes and functions that aims to
facilitate research in biomechanics using python. It does not attempt to
provide a complete workflow from raw files to final analysis (although it may
in a far future), or a main graphical user interface, or magical blackboxes
that process everything automatically.

Kinetics Toolkit is mainly addressed to researchers and students in
biomechanics with a little background in programming, who may or may not
already have a working workflow and who want to understand and control their
data. This is why special attention is made to API documentation and tutorials,
and to ensure the interoperability of ktk with other environments (using pandas
Dataframes as intermediate data containers).


Example
-------
>>>    markers = ktk.kinematics.read_c3d_file('my_file.c3d')
>>>    ktk.Player(markers)

.. image:: https://felixchenier.uqam.ca/wp-content/uploads/2020/05/Sample_ktk.Player_Wheelchair.gif

This library is a work in progress and is still very incomplete.
The stable version API is however mostly settled and generally well tested
using a comprehensive set of unit tests and doc tests, and currently includes:

- `timeseries.TimeSeries` : a generic class to represent time-varying
  n-dimensional data and events, with many methods to extract, merge and subset
  TimeSeries data.

- `kinematics` : a module that loads c3d and n3d files as TimeSeries of
  3d marker positions.

- `player.Player` : a class that allows visualizing 3d markers using a simple
  graphical user interface.
  
- `filters` : a module that wraps some filters from scipy to be applied directly on TimeSeries.

- `save` and `load` : two functions to save and load results to/from JSON-based `ktk.zip` files.

- and other helper functions.

Please be warned that this is still mostly experimental software. If you are
using ktk or are planning to be, you are warmly invited to contact me, first to
say Hello :-), and so that I can warn you before doing major, possibly breaking
changes. Also remind that I develop ktk mainly for my lab and I have limited
resources for troubleshooting. You can however ask your questions_ and if I
can answer, I'll do.


Development version
-------------------

While the development version is also open source, it is developed in parallel
with my research projects following the needs of the moment, and is therefore
well less settled, tested and stable.

.. _questions: mailto:chenier.felix@uqam.ca


Credits
-------

Some external code has been directly included into ktk's source code. Here are
the credits for these nice people.

- Clay Flannigan : icp_ - Python implementation of m-dimensional Iterative
  Closest Point method

I also want to credit the people involved in ktk's dependencies:

- Benjamin Michaud : ezc3d_ - Easy to use C3D reader/writer for C++, Python
  and Matlab

- The dedicated people behind major software and packages used by ktk such as
  python, numpy, matplotlib, pandas, jupyter, pytest, sphinx, etc.

.. _icp: https://github.com/ClayFlannigan/icp
.. _ezc3d: https://github.com/pyomeca/ezc3d


Site map
---------

.. toctree::
    :maxdepth: 3

    install
    customize
    timeseries
    loadsave
    geometry
    filters
    kinematics
    pushrimkinetics
    inversedynamics
    cycles
    dbinterface
    release_notes
    Stable version <https://felixchenier.uqam.ca/kineticstoolkit>
    api_reference

