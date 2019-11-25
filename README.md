# KTK - Kinetics Toolkit

Copyright, Felix Chenier, 2019.
This is not to be redistributed.

This is the Python development version of KTK. It is highly unstable, unfinished and is
a work completely in progress.

To install KTK:

## Installing Anaconda Python

Download and install Anaconda Python 3.7.

## Fixing Spyder ipykernel on macOS

In date of September 18, 2019, there is a bug in the ipykernel package that causes the
IPython console to hand in Spyder. Do this to revert ipykernel to a working version.
This is a workaround and will change with time.

>> echo "ipykernel 4.10.0" >> ~/anaconda3/conda-meta/pinned
>> conda update anaconda

See this post for information and follow-up:
https://stackoverflow.com/questions/53381373/ipython-console-in-spyder-extremely-slow-in-anaconda/57618660#57618660

## Adding KTK folder to Python path

In Spyder, there is a Python Path editor. Add the KTK base folder (the one that contains
the ktk folder) to the python path, so that ''import ktk'' finds the KTK package.

## Adding dependencies

>> conda install -c conda-forge ezc3d
>> conda install pytest