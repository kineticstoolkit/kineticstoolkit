
Kinetics Toolkit
================

Kinetics Toolkit (ktk) is an in-house biomechanical library developed
exclusively in Python, by Professor Félix Chénier at Université du Québec
à Montréal.

[Laboratory website](https://felixchenier.uqam.ca)

[Kinetics Toolkit (ktk) website](https://felixchenier.uqam.ca/kineticstoolkit)

Public version
--------------

The public open-source version API is mostly stable (although currently almost
empty). I do not expect to remove or rename much stuff. However please keep
in mind that this is experimental software. If you are using ktk or are
planning to be, you are warmly invited to contact me, first to say Hello :-),
and so that I can warn you before doing major, possibly breaking changes.

[Tutorials](https://felixchenier.uqam.ca/ktk_dist/tutorials)

[API documentation](https://felixchenier.uqam.ca/ktk_dist/api)


Private unstable version
------------------------

This version is exclusively used in my lab and is developed in parallel with
my research projects, following the needs of the moment. I usually wait several
months before releasing code to the public, mostly to ensure the modules are
stable and the API is mature and global enough to be shared. If you are
interested in collaborating either in research or software development, please
contact me at chenier.felix@uqam.ca

[Tutorials](https://felixchenier.uqam.ca/ktk_lab/tutorials)

[API documentation](https://felixchenier.uqam.ca/ktk_lab/api)


Credits
-------

Some external code has been included into ktk's source code. Here are the
credits for these nice people.

Clay Flannigan : [icp](https://github.com/ClayFlannigan/icp) -
Python implementation of m-dimensional Iterative Closest Point method


Customization
-------------

By default, importing ktk changes some defaults in IPython and matplotlib to
get a more 'research' and less 'programming' experience. Please note that this
does not affect anything besides visual representations.

### Modification to repr of dictionaries ###
In ktk, data are often stored as dictionaries, which can lead to very large
printouts when we simply want to see the dictionary's contents. Importing ktk
changes the repr of dictionaries in IPython so that a summary of the dict's
content is shown, more like the representation of a Matlab struct.

    import numpy as np
    data = dict()
    data['data1'] = np.arange(30)
    data['data2'] = np.arange(30) ** 2
    data['data3'] = np.arange(30) ** 3

    data
    {'data1': array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]),
     'data2': array([  0,   1,   4,   9,  16,  25,  36,  49,  64,  81, 100, 121, 144,
            169, 196, 225, 256, 289, 324, 361, 400, 441, 484, 529, 576, 625,
            676, 729, 784, 841]),
     'data3': array([    0,     1,     8,    27,    64,   125,   216,   343,   512,
              729,  1000,  1331,  1728,  2197,  2744,  3375,  4096,  4913,
             5832,  6859,  8000,  9261, 10648, 12167, 13824, 15625, 17576,
            19683, 21952, 24389])}

    import ktk
    data
    {
        'data1': <array of shape (30,)>,
        'data2': <array of shape (30,)>,
        'data3': <array of shape (30,)>
    }

### Modification of repr of numpy's floats ###
Numpy is set to display floats with floating point precision.

### Alternative defaults to matplotlib ###
We assume that most work with figure is interactive, on screen. In that view,
the following modifications are made to default matplotlib figures:

- The standard dpi is changed to 75, which allows for more work space by
  reducing the font size on screen.
- The standard figure size is changed to [10, 5], which is a little bigger
  than the default and is thus more practical for interactive navigation.
- The default color order is changed to (rgbcmyk). The first colors, red,
  green and blue, are consistent the colours assigned to x, y and z in most
  3D visualization softwares, and the next colours are consistent with
  Matlab's legacy color order.

### Changing defaults ###
In some case we would not want a module to change the behaviour of the
console. To change ktk's defaults, open ktk's `__init__`.py and change the
corresponding entries in the config dictionary. Most often this file can be
opened using:

        import ktk
        edit ktk

