Installing the stable version
=============================

Although Kinetics Toolkit is distributed via `pip`, I strongly recommend using a conda environment, if only to install `ezc3d` which is required for opening c3d files and which is distributed via `conda-forge`.

1. Download and install Anaconda or Miniconda with python 3.8 or newer.

2. Create a new environment (recommended): open an Anaconda Prompt (on Windows) or a terminal (on macOS and Linux) and type these commands one by one to create a `ktk` virtual environment and install the dependencies in this environment.

        conda create -n ktk

        conda activate ktk

        conda install -c conda-forge python=3.8 matplotlib scipy pandas scikit-learn pyqt ezc3d
   
3. Interactive functions can be tricky in Python. Ktk's interactive functions make use of IPython's integration of Matplotlib/Qt5's event loop, which is completely supported in the Spyder IDE. Therefore I recommend using Spyder, but any IDE that uses IPython should also work well.

        conda install -c conda-forge spyder

4. Install Kinetics Toolkit

        pip install kineticstoolkit

5. Choose the Qt5 backend for IPython: if you are using the Spyder IDE, go to the Spyder's preferences, to the **IPython console** item, then to the **Graphics** pane. In the **Graphics backend** box, select **Qt5**. Restart Spyder.

Verify that you are able to import Kinetics Toolkit in the interactive IPython console. This should work without error:

    >>> import kineticstoolkit
    
    or
    
    >>> import kineticstoolkit.lab as ktk
