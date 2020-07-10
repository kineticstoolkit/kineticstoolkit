Kinetics Toolkit (ktk)
======================

Kinetics Toolkit (ktk) is a pure-python biomechanical library developed by
Professor Félix Chénier at Université du Québec à Montréal, Canada. It is a
package of integrated classes and functions that aims to facilitate research in
biomechanics using python. It does not attempt to provide a complete workflow
from raw files to final analysis (although it may in a far future), or a main
graphical user interface, or magical blackboxes that process everything
automatically.

Kinetics Toolkit is mainly addressed to researchers and students in
biomechanics with a little background in programming, who may or may not
already have a working workflow and who want to understand and control their
data. This is why special attention is made to API documentation and tutorials,
and to maximize the interoperability of ktk with other environments (using
Pandas Dataframes as intermediate data containers).

!!! example
    markers = ktk.kinematics.read_c3d_file('my_file.c3d')

	ktk.Player(markers)


The public open-source version API is mostly stable and currently includes:

- `TimeSeries` : a generic class to represent time-varying
  n-dimensional data and events, with many methods to extract, merge and subset
  TimeSeries data.

- `kinematics` : a module that loads c3d and n3d files as TimeSeries of
  3d marker positions.

- `Player` : a class that allows visualizing 3d markers using a simple
  graphical user interface.

- and some helper functions.

Please be warned that this is mostly experimental software. If you are using
ktk or are planning to be, you are warmly invited to contact me, first to say
Hello :-), and so that I can warn you before doing major, possibly breaking
changes. Also remind that I develop ktk mainly for my lab and I have limited
resources for troubleshooting. You can however
[ask your questions](mailto:chenier.felix@uqam.ca)
and if I can answer, I'll do.

![Player sample](https://felixchenier.uqam.ca/wp-content/uploads/2020/05/Sample_ktk.Player_Wheelchair.gif)



User's requirements
--------------------

The ktk library is a Python API that aims to ease the processing of biomechanical data. Since this is an API and not a GUI, some experience in programming is required. Here is a probably non-exhaustive list of prerequesites :

- Basic python skills :
    - Variable types : bool, int, float, str, dict, list, tuple
    - Function definitions : def
- Numpy :
    - array, indexing an array, operations on arrays, etc.
- Matplotlib :
    - basic plotting functions (figure, plot)

Although there are some references to object-oriented programming (classes) and Pandas DataFrames, using ktk does not require mastering those concepts and tools.



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