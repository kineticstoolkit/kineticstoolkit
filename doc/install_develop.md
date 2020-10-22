Installing the development version
==================================

Requirements
------------

### Download and install Anaconda or Miniconda ###

I suggest Miniconda as it is way smaller than the whole Anaconda suite.

[Miniconda website](https://docs.conda.io/en/latest/miniconda.html) - Select the most recent version that corresponds to your system.

### Create a virtual environment and install the external dependencies ###

Open an Anaconda Prompt (on Windows) or a terminal (on macOS and Linux) and type these commands one by one to create a `ktk` virtual environment and install the dependencies in this environment.

    conda create -n ktk

    conda activate ktk

    conda install -c conda-forge python=3.8 matplotlib scipy pandas scikit-learn pyqt ezc3d git pytest mypy jupyterlab spyder sphinx sphinx-material recommonmark sphinx-autodoc-typehints autodocsumm nbsphinx twine


Installing Kinetics Toolkit
---------------------------

Clone Kinetics Toolkit from github: On Windows, open Git Bash (ktk) from the Anaconda3 menu. On macOS or Linux, open a terminal. In that terminal, run this command. This will create a folder named 'kineticstoolkit' in the current folder. You may wish to facultatively change the current folder before running theses commands.

    git clone https://github.com/felixchenier/kineticstoolkit.git


Configuring Spyder
------------------

On Windows, open Spyder by selecting 'Spyder (ktk)' from the Anaconda3 menu. On macOS and Linux, open a terminal and write:

    conda activate ktk
    spyder

### Qt5 backend ###

Kinetics Toolkit uses Matplotlib for user interaction. To set it permanently in Spyder, go to the Spyder's preferences, to the **IPython console** item, then to the
**Graphics** pane. In the **Graphics backend** box, select **Qt5**.

### Python path ###

In Spyder, look for the **PYTHONPATH manager**. Open this manager and add the `kineticstoolkit` folder that you just cloned to the python path (the outer one, not the inner one).

Restart Spyder. Writing `import kineticstoolkit` should now find and import Kinetics Toolkit
without error.


Keeping up to date
------------------

To update Kinetics Toolkit to the laster master version, open Git bash (on Windows) or a terminal (on macOS and Linux), navigate to the kineticstoolkit folder and type:

    git pull origin master
