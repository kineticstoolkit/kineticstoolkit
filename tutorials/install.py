# %%
"""
Installing Kinetics Toolkit (ktk)
=================================

Please follow these steps to install ktk.

Obtain access to ktk
--------------------
At the current time, ktk can only be used in the context of collaborations with
me (Felix Chenier). Please [email me](mailto:chenier.felix@uqam.ca) if you are
interested to try ktk.

Download and install git
------------------------
KTK is versionned using git, and includes functions to update itself. The first
step to obtain KTK is to install git on your computer.

[git website](https://git-scm.com)

Just use the default checkboxes in every dialog, everything should be fine.

Download and install Anaconda Python 3.7.
----------------------------------------
[Anaconda distribution website](https://www.anaconda.com/distribution/)

Only on macOS: Fixing Spyder
----------------------------
In date of September 18, 2019, there is a bug in the ipykernel package that
causes the IPython console to hand in Spyder. Do this to revert ipykernel to
a working version. This is a workaround and it will hopefully become useless
with time.

    echo "ipykernel 4.10.0" >> ~/anaconda3/conda-meta/pinned
    conda update anaconda

[See this post for information and follow-up](https://stackoverflow.com/questions/53381373/ipython-console-in-spyder-extremely-slow-in-anaconda/57618660#57618660)

Add external dependencies
-------------------------
Open the Anaconda console on Windows, or a terminal in macOS, and type these
commands to install ktk's additional dependencies:

    conda install -c conda-forge ezc3d
    conda install pytest

Clone KTK from bitbucket
------------------------
Open Spyder from your Anaconda distribution, navigate to the directory where
you want to install KTK (for example, F:\), and run these lines in the
IPython console:

    import os
    print('Cloning repository...')
    os.system('git clone https://labofelixchenier@bitbucket.org/felixchenier/kineticstoolkit.git')
    print('Pulling origin/master...')
    %cd kineticstoolkit
    os.system('git pull origin master')

Add kineticstoolkit to the Python path
--------------------------
In Spyder, there is a PYTHONPATH manager. Add the kineticstoolkit folder that
you just cloned to the python path, then restart Spyder. From now on,
`import ktk` should find and import ktk.

Congratulation, ktk is now installed. From now on, you can always update to
the most recent version using:

    import ktk
    ktk.config.update()

"""
