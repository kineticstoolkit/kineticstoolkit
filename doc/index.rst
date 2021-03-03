Kinetics Toolkit
================

Kinetics Toolkit (ktk) is an open-source, pure-python package of integrated classes and functions that aims to facilitate research in biomechanics using python. It is mainly addressed to researchers and students in biomechanics with a little background in programming, who want to understand and control their data. This is why special attention is made on documentation and interoperability with other environments (using pandas Dataframes and JSON files as intermediate data containers).


.. image:: https://felixchenier.uqam.ca/wp-content/uploads/2020/05/Sample_ktk.Player_Wheelchair.gif


You can ask questions and submit bugs or feature requests on `ktk's github issue tracker`_. Please keep in mind that I develop Kinetics Toolkit primarily for my lab and I have limited resources for troubleshooting. However, if I can, it will be my great pleasure to help.

.. _`ktk's github issue tracker`: https://github.com/felixchenier/kineticstoolkit/issues


Credits
-------

Kinetics Toolkit is developed at the `Mobility and Adaptive Sports Research Lab`_ by Professor `Félix Chénier`_ at `Université du Québec à Montréal`_, Canada.

Thanks to Clay Flannigan for his icp_ method, to Benjamin Michaud for his ezc3d_ module, and to the dedicated people behind major software and packages used by Kinetics Toolkit, such as python, numpy, matplotlib, pandas, jupyter, pytest, sphinx, etc.

.. _`Mobility and Adaptive Sports Research Lab`: https://felixchenier.uqam.ca
.. _`Université du Québec à Montréal`: https://uqam.ca
.. _icp: https://github.com/ClayFlannigan/icp
.. _ezc3d: https://github.com/pyomeca/ezc3d


Site map
---------

.. toctree::
    :caption: GETTING STARTED
    :maxdepth: 2

    install
    conventions
    timeseries


.. toctree::
    :caption: TUTORIALS
    :maxdepth: 2

    loadsave
    filters/filters
    cycles
    geometry/geometry
    kinematics/kinematics
    pushrimkinetics

.. toctree::
    :caption: FOR DEVELOPERS
    :maxdepth: 2

    dev/install
    dev/website
    dev/conventions
    dev/tutorials

.. toctree::
    :caption: API REFERENCE

    lab_mode
    release_notes
    api_reference
