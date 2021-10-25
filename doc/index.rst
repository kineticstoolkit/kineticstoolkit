Kinetics Toolkit
================

*An Open-Source Python Package to Facilitate Research in Biomechanics*

.. image:: https://joss.theoj.org/papers/10.21105/joss.03714/status.svg
   :target: https://doi.org/10.21105/joss.03714

.. image:: https://anaconda.org/conda-forge/kineticstoolkit/badges/version.svg
   :target: https://anaconda.org/conda-forge/kineticstoolkit
   
.. image:: https://anaconda.org/conda-forge/kineticstoolkit/badges/latest_release_date.svg
   :target: https://anaconda.org/conda-forge/kineticstoolkit
   
----------------

.. ifconfig:: release in ['master']

    .. warning::
        You are currently on Kinetics Toolkit's development website. This
        website may refer to unreleased and unstable material. Please click
        here_ to consult the standard website that refers to the current
        version.

    .. _here: https://felixchenier.uqam.ca/kineticstoolkit


`Kinetics Toolkit (ktk)`_ is an open-source, pure-python package of integrated
classes and functions that aims to facilitate research in biomechanics.
It is mainly addressed to researchers and students in biomechanics with
minimal experience in programming, who want to learn processing and controlling
their research data using python. A special attention is made on documentation
and interoperability with other environments (using pandas Dataframes and JSON
files as intermediate data containers).

This is a long term project that is focused not only on the tool itself, but
also a lot on learning. Please consult the "What is Kinetics Toolkit" section
for more information on the project.

.. image:: https://felixchenier.uqam.ca/wp-content/uploads/2020/05/Sample_ktk.Player_Wheelchair.gif


Please ask questions and submit bugs or feature requests on the
`git-hub issue tracker`_. While I develop Kinetics Toolkit primarily for my lab
and have limited resources for troubleshooting, it will be my great pleasure to
help if I can.

.. _`git-hub issue tracker`: https://github.com/felixchenier/kineticstoolkit/issues
.. _`Kinetics Toolkit (ktk)`: https://felixchenier.uqam.ca/kineticstoolkit

Credits
-------

Kinetics Toolkit is developed at the
`Mobility and Adaptive Sports Research Lab`_ by Professor Félix Chénier at
`Université du Québec à Montréal`_, Canada.

Thanks to Clay Flannigan for his icp_ method, to Benjamin Michaud for his
ezc3d_ module, and to the dedicated people behind major software and packages
used by Kinetics Toolkit, such as python, numpy, matplotlib, pandas, jupyter,
pytest, sphinx, etc.

.. _`Mobility and Adaptive Sports Research Lab`: https://felixchenier.uqam.ca
.. _`Université du Québec à Montréal`: https://uqam.ca
.. _icp: https://github.com/ClayFlannigan/icp
.. _ezc3d: https://github.com/pyomeca/ezc3d


Site map
---------

.. toctree::
    :caption: GETTING STARTED
    :maxdepth: 2

    what_is_kinetics_toolkit
    install
    conventions

.. toctree::
    :caption: TUTORIALS
    :maxdepth: 2

    timeseries/timeseries
    loadsave
    filters/filters
    cycles
    geometry/geometry
    kinematics/kinematics
    pushrimkinetics

.. toctree::
    :caption: IN DEPTH

    lab_mode
    release_notes
    api_reference
    Development website <https://felixchenier.uqam.ca/ktk_develop>
    GitHub <https://github.com/felixchenier/kineticstoolkit>

.. toctree::
    :caption: FOR DEVELOPERS
    :maxdepth: 2

    dev/install
    dev/conventions
    dev/tutorials
