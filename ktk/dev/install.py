"""
Installation script for KineticsToolkit (KTK)

Requirements
------------
This installer requires python3 and Git.

- To install python3, the easiest way is to install Anaconda from
  http://anaconda.com/distribution

- If git is not already installed on your computer (typical case on Windows)
  then install it from https://git-scm.com/downloads
  You can select every default installation choice.

Installation
------------
Please copy this file in the folder where you want to install KTK. Then, in
python, switch to this folder and run this file. It will create a folder named
kineticstoolkit, then download KTK into this folder.

Once the installation has finished, you will be prompted to add the
kineticstoolkit to your PYTHONPATH. In Spyder, this can be done using the
PYTHONPATH editor.

Author: Felix Chenier <chenier.felix@uqam.ca>
Date: October 2019
"""
from os import chdir, getcwd, system
from time import sleep

user = input('Please enter your Bitbucket user name: ')
password = input('Please enter your Bitbucket password '
                 '(WARNING: THIS WILL BE PRINTED IN FULL LETTERS): ')
print('Cloning repository...')
system(f'git clone https://{user}:{password}@bitbucket.org/'
       'felixchenier/kineticstoolkit.git')
print('Pulling origin/master...')
chdir('kineticstoolkit')
system('git pull')
sleep(1)
print('====================================================')
print('       DOWNLOAD AND INSTALLATION COMPLETE.')
print('----------------------------------------------------')
print(f'Now please add "{getcwd()}" to your PYTHONPATH.')
print('On Spyder for Windows, look in the Tools menu.')
print('On Spyder for macOS, look in the Python menu.')
print('Then, restart Spyder. You can now use KineticsToolkit:')
print('    import ktk')
print('====================================================')
chdir('..')
