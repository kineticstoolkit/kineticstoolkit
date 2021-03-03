Kinetics Toolkit
================

Kinetics Toolkit (ktk) is an open-source, pure-python package of integrated classes and functions that aims to facilitate research in biomechanics using python. It is mainly addressed to researchers and students in biomechanics with a little background in programming, who want to understand and control their data. This is why special attention is made on documentation and interoperability with other environments (using pandas Dataframes and JSON files as intermediate data containers).

Kinetics Toolkit is developed at the `Mobility and Adaptive Sports Research Lab`_ in Montreal.

.. _`Mobility and Adaptive Sports Research Lab`: https://felixchenier.uqam.ca


Example
-------
>>>    markers = ktk.kinematics.read_c3d_file('my_file.c3d')
>>>    ktk.Player(markers)

.. image:: https://felixchenier.uqam.ca/wp-content/uploads/2020/05/Sample_ktk.Player_Wheelchair.gif


Package contents
----------------

Kinetics Toolkit currently includes:

Lower level modules, classes and functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `TimeSeries` : a generic class to represent time-varying n-dimensional data and events, with many methods to extract, merge and subset TimeSeries data.

- `filters` : a module to filter TimeSeries.

- `cycles` : a module that detects and processes cycles and phases.

- `save`/`load` : functions to save/load data in a zipped JSON format.

- other helper functions.

Higher level modules and classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `kinematics` : a module to read and process marker data.

- `player.Player` : a class to visualize markers in 3D using a simple interactive interface.

- `pushrimkinetics` : a module to read and process kinetics data from instrumented wheelchair wheels.


Other modules are also being developed.

You can ask questions and submit bugs or feature requests on `ktk's github issue tracker`_. However, please keep in mind that I develop Kinetics Toolkit primarily for my lab and I have limited resources for troubleshooting. But if I can answer, I'll be glad to help.

.. _`ktk's github issue tracker`: https://github.com/felixchenier/kineticstoolkit/issues


Credits
-------
Kinetics Toolkit is developed at the Mobility and Adaptive Sports Research Lab by Professor `Félix Chénier`_ at Université du Québec à Montréal, Canada.

Thanks to Clay Flannigan for his icp_ method, to Benjamin Michaud for his ezc3d_ module, and to the dedicated people behind major software and packages used by Kinetics Toolkit, such as python, numpy, matplotlib, pandas, jupyter, pytest, sphinx, etc.

.. _`Félix Chénier`: https://felixchenier.uqam.ca
.. _icp: https://github.com/ClayFlannigan/icp
.. _ezc3d: https://github.com/pyomeca/ezc3d


Site map
---------

.. toctree::
    :caption: THE BASICS
    :maxdepth: 2

    install
    timeseries
    loadsave

.. toctree::
    :caption: SIMPLE OPERATIONS
    :maxdepth: 2

    filters/filters
    cycles
    geometry/geometry

.. toctree::
    :caption: MORE COMPLEX ANALYSES
    :maxdepth: 2

    kinematics/kinematics
    pushrimkinetics

.. toctree::
    :caption: IN DEVELOPMENT

    inversedynamics
    dbinterface

.. toctree::
    :caption: IN DEPTH

    labmode
    release_notes
    api_reference
