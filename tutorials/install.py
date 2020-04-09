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

Download and install Anaconda Python 3.7.
----------------------------------------
[Anaconda distribution website](https://www.anaconda.com/distribution/)

Add external dependencies
-------------------------
Open the Anaconda console on Windows, or a terminal in macOS, and type these
commands to install ktk's additional dependencies:

    conda install -c conda-forge ezc3d
    conda install pytest

Clone ktk from bitbucket
------------------------
Open Spyder from your Anaconda distribution, navigate to the directory where
you want to install ktk (for example, F:\), and run these lines in the
IPython console, replacing YOUR_BITBUCKET_USERNAME with your bitbucket
username.

    import os
    print('Cloning repository...')
    os.system('git clone https://YOUR_BITBUCKET_USERNAME@bitbucket.org/felixchenier/kineticstoolkit.git')
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

Restart Spyder. From now on, `import ktk` should find and import ktk.

---------------------------------------------------

Congratulation, ktk is now installed. From now on, you can always update to
the most recent version using:

    import ktk
    ktk.update()

"""
