Kinetics Toolkit
================

Kinetics Toolkit is an open-source, pure-python package of integrated
classes and functions that aims to facilitate research in biomechanics using
python.

Kinetics Toolkit is mainly addressed to researchers and students in
biomechanics with a little background in programming, who may or may not
already have a working workflow and who want to understand and control their
data. This is why special attention is made to build rich API documentation and
tutorials, and to ensure the interoperability of ktk with other environments
(using pandas Dataframes and JSON files as intermediate data containers).

Kinetics Toolkit is developed at the `Mobility and Adaptive Sports Research Lab`_ in
Montreal.

.. _`Mobility and Adaptive Sports Research Lab`: https://felixchenier.uqam.ca


Example
-------
>>>    markers = ktk.kinematics.read_c3d_file('my_file.c3d')
>>>    ktk.Player(markers)

.. image:: https://felixchenier.uqam.ca/wp-content/uploads/2020/05/Sample_ktk.Player_Wheelchair.gif


Stable version
--------------

The stable version API is mostly settled and well tested using a comprehensive set of unit tests and
doc tests, and currently includes:

Lower level modules, classes and functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `TimeSeries` : a generic class to represent time-varying
  n-dimensional data and events, with many methods to extract, merge and subset
  TimeSeries data.

- `filters` : a module that wraps some filters from scipy to be applied directly on TimeSeries.

- `cycles` : a module that detects, time-normalizes and stacks cycles (e.g., gait cycles,
  wheelchair propulsion cycles, etc.)
  
- `save` and `load` : two functions to save and load results to/from JSON-based `ktk.zip` files.

- other helper functions.

Higher level modules and classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `kinematics` : a module that loads c3d and n3d files as TimeSeries of
  3d marker positions.

- `player.Player` : a class that allows visualizing 3d markers using a simple
  graphical user interface.

- `pushrimkinetics` : a module that reads files from instrumented wheelchair wheels, reconstructs
  the pushrim kinetics, removes dynamic offsets in kinetic signals, and perform speed and power
  calculation for analysing spatiotemporal and kinetic parameters of wheelchair propulsion.


You can ask your questions and submit bugs or feature requests on `ktk's github issue tracker`_.
However, please keep in mind that I develop Kinetics Toolkit primarily for my lab and I have limited
resources for troubleshooting. But if I can answer, I'll do.

.. _`ktk's github issue tracker`: https://github.com/felixchenier/kineticstoolkit/issues


Development version
-------------------

The development_ version is developed in parallel with my research projects following the needs
of the moment, and is therefore well less settled, tested and stable.

.. _development: https://felixchenier.uqam.ca/ktk_develop/


Credits
-------
Kinetics Toolkit is developed by Professor `Félix Chénier`_ at Université du
Québec à Montréal, Canada.

Some external code has been directly included into ktk's source code. Here are
the credits for these nice people.

- Clay Flannigan : icp_ - Python implementation of m-dimensional Iterative
  Closest Point method

I also want to credit the people involved in ktk's dependencies:

- Benjamin Michaud : ezc3d_ - Easy to use C3D reader/writer for C++, Python
  and Matlab

- The dedicated people behind major software and packages used by ktk such as
  python, numpy, matplotlib, pandas, jupyter, pytest, sphinx, etc.

.. _`Félix Chénier`: https://felixchenier.uqam.ca
.. _icp: https://github.com/ClayFlannigan/icp
.. _ezc3d: https://github.com/pyomeca/ezc3d


Site map
---------

.. toctree::
    :maxdepth: 3

    install
    labmode
    timeseries
    loadsave
    filters
    kinematics
    pushrimkinetics
    cycles
    release_notes
    Development version <https://felixchenier.uqam.ca/ktk_develop>
    api_reference

