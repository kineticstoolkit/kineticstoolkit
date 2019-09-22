py-c3d
======

This is a small library for reading and writing C3D binary files. C3D files are
a standard format for recording 3-dimensional time sequence data, especially
data recorded by a 3D motion tracking apparatus.

Installing
----------

Install with pip::

    pip install c3d

Or if you'd like to use the bleeding-edge version, just clone the github
repository and build and install using the normal Python setup process::

    git clone https://github.com/EmbodiedCognition/py-c3d
    cd py-c3d
    python setup.py install

Usage
-----

Tools
~~~~~

This package includes a script for converting C3D motion data to CSV format
(``c3d2csv``) and an OpenGL-based visualization tool for observing the motion
described by a C3D file (``c3d-viewer``).

Library
~~~~~~~

To use the C3D library, just import the package and create a ``Reader`` or
``Writer`` depending on your intended usage

.. code-block:: python

    import c3d

    with open('data.c3d', 'rb') as handle:
        reader = c3d.Reader(handle)
        for i, (points, analog) in enumerate(reader.read_frames()):
            print('Frame {}: {}'.format(i, points.round(2)))

You can also get and set metadata fields using the library; see the `package
documentation`_ for more details.

.. _package documentation: http://c3d.readthedocs.org

Caveats
-------

This library is minimally effective, in the sense that the only motion tracking
system I have access to (for testing) is a Phasespace system. If you try out the
library and find that it doesn't work with your motion tracking system, let me
know. Pull requests are also welcome!

Also, if you're looking for more functionality than just reading and writing C3D
files, there are a lot of better toolkits out there that support a lot more file
formats and provide more functionality, perhaps at the cost of increased
complexity. The `biomechanical toolkit`_ is a good package for analyzing motion
data.

.. _biomechanical toolkit: http://code.google.com/p/b-tk/
