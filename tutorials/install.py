# %%
"""
Installing Kinetics Toolkit (ktk)
=================================

Please follow these steps to install ktk.

Obtain access to ktk
--------------------
At the current time, ktk can only be used in the context of collaborations with
me (Felix Chenier). I do plan to open source it someday, but it needs to be
polished before it is. Please [email me](mailto:chenier.felix@uqam.ca) if you
are interested to try out ktk.

Download and install git
------------------------
Ktk is versionned using git, and includes functions to update itself. The first
step to obtain ktk is to install git on your computer.

[git website](https://git-scm.com)

Just use the default checkboxes in every dialog, everything should be fine.

Download and install Python and the required packages
-----------------------------------------------------
There are many ways to do this. I recommend to install Anaconda or Miniconda, and use
conda to install the required packages.

### Windows ###
The easiest way is to install Anaconda, then add some additional packages.

#### Download and install Anaconda ####
[Anaconda individual website](https://www.anaconda.com/products/individual)

#### Add external dependencies ####
Open the Anaconda console and type these commands to install ktk's additional
dependencies:

    conda install -c conda-forge ezc3d
    conda install pytest
    
From now on, you can open Spyder by clicking on its icon in the Windows start menu.

### macOS ###

#### Download and install Anaconda or Miniconda ####
[Anaconda individual website](https://www.anaconda.com/products/individual)

[Miniconda website](https://docs.conda.io/en/latest/miniconda.html)

#### Create a virtual environment and install the external dependencies ####
Open a terminal and type these commands to create a ``ktk`` virtual environment and
install the dependencies in this environment. I wrote specific versions for some packages
to help resolving some bugs with qt and matplotlib on macOS Catalina. Please note that
an unresolved major bug exists in macOS Mojave that makes the whole session crash when
using tkinter. Either upgrade to Catalina or revert to python 3.3.0.

    conda create -n ktk
    conda activate ktk
    conda install -c conda-forge python=3.7 pyqt=5.12 matplotlib=3.2 ezc3d spyder
    conda install -c conda-forge scipy pandas scikit-learn pytest jupyter

From now on, you can open Spyder by opening a terminal and writing:

	conda activate ktk
	spyder

Clone ktk from bitbucket
------------------------
Open Spyder, navigate to the directory where you want to install ktk (for example, F:\),
and run these lines in the IPython console, replacing USERNAME with your
bitbucket username.

    import os
    print('Cloning repository...')
    os.system('git clone https://USERNAME@bitbucket.org/felixchenier/kineticstoolkit.git')
    print('Pulling origin/master...')
    %cd kineticstoolkit
    os.system('git pull origin master')

Configure Spyder
-------------------------------------------------

### Select the correct matplotlib framework ###

In Spyder's preference, go to the **IPython console** item, then to the
**Graphics** pane. In the **Graphics backend** box, select **Qt5**.

### Add the kineticstoolkit folder to the Python path ###

In Spyder, look for the **PYTHONPATH manager**. Open this manager and add the
`kineticstoolkit` folder that you just cloned to the python path.

Restart Spyder. Writing `import ktk` should find and import ktk.

---------------------------------------------------

Congratulation, ktk is now installed. From now on, you can always update to
the most recent version using:

    import ktk
    ktk.update()

"""
