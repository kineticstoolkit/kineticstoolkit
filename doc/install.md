Installing
==========

Kinetics Toolkit is distributed via both conda-forge and PyPi. The source code is hosted on [git-hub](https://github.com/felixchenier/kineticstoolkit).

Since reading c3d files relies on `ezc3d` which binaries are distributed only via conda-forge, the recommended installation method is therefore using conda-forge.

1. Download and install Anaconda or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) with python 3.8 or newer.

2. Create a new environment (recommended): open an Anaconda Prompt (on Windows) or a terminal (on macOS and Linux) and type these commands one by one to create a `ktk` virtual environment and install Kinetics Toolkit in this environment.

        conda create -n ktk

        conda activate ktk
        
3. Install Kinetics Toolkit

    Stable version (recommended):
        
    a) Install:

            conda install -c conda-forge kineticstoolkit ezc3d
            
    b) To keep up to date:
        
            conda update -c conda-forge kineticstoolkit
        
    Development version (for programmers):
    
    a) Install the dependencies:
        
            conda install -c conda-forge python=3.8 matplotlib scipy pandas scikit-learn pyqt ezc3d limitedinteraction git pytest mypy coverage jupyterlab spyder sphinx sphinx-material recommonmark sphinx-autodoc-typehints autodocsumm nbsphinx twine
            
    b) Clone the git repository:
        
            git clone https://github.com/felixchenier/kineticstoolkit.git ktk_develop
            
    c) Add ktk_develop to your PYTHON_PATH
        
    d) To keep up to date:
        
            cd ktk_develop; git pull origin master
   
3. Interactive functions can be tricky in Python. Kinetics Toolkit's interactive functions make use of IPython's integration of Matplotlib/Qt5's event loop, which is completely supported in Spyder IDE. Therefore I recommend using [Spyder](https://www.spyder-ide.org) since it is so oriented towards science, but any IDE that uses IPython should also work well. Be sure to [configure Spyder](https://docs.spyder-ide.org/current/faq.html#using-existing-environment) to use the ktk environment you just created.

4. Choose the Qt5 backend for IPython.

    - If you are using the Spyder IDE, go to the Spyder's preferences, to the **IPython console** item, then to the **Graphics** pane. In the **Graphics backend** box, select **Qt5**. Restart Spyder.
    - Otherwise, you can just type `%matplotlib qt5` in IPython or Jupyter.

Verify that you are able to import Kinetics Toolkit in the interactive IPython console. This should work without error:

    >>> import kineticstoolkit
    
    or
    
    >>> import kineticstoolkit.lab as ktk
