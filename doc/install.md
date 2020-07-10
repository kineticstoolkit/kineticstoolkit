Installing
==========

Requirements
------------

Although ktk is distributed via `pip`, I strongly recommend using a conda environment, if only to install `ezc3d` which is required for opening c3d files and which is distributed via `conda-forge`.

### Download and install Anaconda or Miniconda ###

I suggest Miniconda as it is way smaller than the whole Anaconda suite.

[Miniconda website](https://docs.conda.io/en/latest/miniconda.html) - Select the Python 3.7 version that corresponds to your system.

### Create a virtual environment and install the external dependencies ###

Open an Anaconda Prompt (on Windows) or a terminal (on macOS and Linux) and type these commands one by one to create a `ktk` virtual environment and install the dependencies in this environment.

    conda create -n ktk

    conda activate ktk

    conda install -c conda-forge python=3.7 matplotlib scipy pandas scikit-learn ezc3d

Additional dependencies for the private development version:

    conda install -c conda-forge git pytest jupyter spyder pdoc3 mkdocs mkdocs-material mkdocs-material-extensions
    

Installing ktk
--------------

### Public version ###

The public, open-source version of ktk is distributed va `pip`. Open an Anaconda Prompt (on Windows) or a terminal (on macOS and Linux), and type:

    pip install ktk

### Development version ###

1. Obtain access: Please [email me](mailto:chenier.felix@uqam.ca) if you are interested to collaborate on research and/or ktk development.

2. Clone ktk from bitbucket: On Windows, open Git Bash (ktk) from the Anaconda3 menu. On macOS or Linux, open a terminal. In that terminal, run these lines one by one, replacing USERNAME with your bitbucket username. This will create a folder named 'kineticstoolkit' in the current folder. You may wish to facultatively change the current folder before running theses commands.

        git clone https://USERNAME@bitbucket.org/felixchenier/kineticstoolkit.git')
        cd kineticstoolkit
        git pull origin


Configuring Spyder
------------------

On Windows, open Spyder by selecting 'Spyder (ktk)' from the Anaconda3 menu. On macOS and Linux, open a terminal and write:

	conda activate ktk
	spyder

### Qt5 backend ###

KTK uses Matplotlib for user interaction. To set it permanently in Spyder, go to the Spyder's preferences, to the **IPython console** item, then to the
**Graphics** pane. In the **Graphics backend** box, select **Qt5**.

### Python path (development version only) ###

In Spyder, look for the **PYTHONPATH manager**. Open this manager and add the
`kineticstoolkit` folder that you just cloned to the python path.

Restart Spyder. Writing `import ktk` should now find and import ktk without error.


Keeping ktk up to date
----------------------

For the public version, open an Anaconda Prompt (on Windows) or a terminal (on macOS and Linux), and type:

    pip upgrade ktk
    
For the development version, in an IPython console, type:

    import ktk
    ktk.update()    
