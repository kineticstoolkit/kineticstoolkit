
Kinetics Toolkit
================

Kinetics Toolkit (ktk) is a pure-python biomechanical library developed by
Professor Félix Chénier at Université du Québec à Montréal, Canada. It does not
attempt to provide user-friendly graphical user interfaces (apart from the
Player class to visualize 3d kinematics) or magical blackboxes that process
everything automatically. It is rather a framework that aims to integrate
flexible classes and functions to facilitate research in biomechanics.

Although most ktk modules express data using the custom ktk.TimeSeries class,
it is still easy to integrate ktk with other environments using pandas
Dataframes as intermediate containers, using the TimeSeries' from_dataframe
and to_dataframe methods.

Kinetics Toolkit is addressed mainly to researchers and students in
biomechanics with a little background in programming, who want to understand
and control their data. This is why special attention is made to API
documentation and tutorials.

[Laboratory website](https://felixchenier.uqam.ca)

[Kinetics Toolkit (ktk) website](https://felixchenier.uqam.ca/kineticstoolkit)

Public version
--------------

The public open-source version API is mostly stable and currently includes:

- `timeseries.TimeSeries` : a generic class to represent time-varying
  n-dimensional data and events, with many methods to extract, merge and subset
  TimeSeries data.

- `kinematics` : a module that loads c3d and n3d files as TimeSeries of
  3d marker positions.

- `player.Player` : a class that allows visualizing 3d markers using a simple
  graphical user interface.

- and some helper functions.

Please be warned that this is mostly experimental software. If you are using
ktk or are planning to be, you are warmly invited to contact me, first to say
Hello :-), and so that I can warn you before doing major, possibly breaking
changes. Also remind that I develop ktk mainly for my lab and I have limited
resources for troubleshooting. You can however
[ask your questions](mailto:chenier.felix@uqam.ca)
and if I can answer, I'll do.

[Tutorials](https://felixchenier.uqam.ca/ktk_dist/tutorials)

[API documentation](https://felixchenier.uqam.ca/ktk_dist/api)


Private development version
---------------------------

The development version is exclusively used in my lab and is developed in
parallel with my research projects, following the needs of the moment. I
usually wait several months before releasing new code to the public, mostly to
ensure the modules are stable, well tested, documented, and the API is mature and global enough to be shared.

[Tutorials](https://felixchenier.uqam.ca/ktk_lab/tutorials)

[API documentation](https://felixchenier.uqam.ca/ktk_lab/api)


Credits
-------

Some external code has been directly included into ktk's source code. Here are
the credits for these nice people.

- Clay Flannigan : [icp](https://github.com/ClayFlannigan/icp) -
  Python implementation of m-dimensional Iterative Closest Point method

I also want to credit the people involved in ktk's dependencies:

- Pariterre and contributors : [ezc3d](https://github.com/pyomeca/ezc3d) -
  Easy to use C3D reader/writer for C++, Python and Matlab

- The dedicated people behind major software and packages used by ktk such as
  python, numpy, matplotlib, pandas, jupyter, pytest, pdoc3, etc.

