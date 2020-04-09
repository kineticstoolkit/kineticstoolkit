# %%
"""
DBInterface
========================
The ktk.DBInterface class interfaces with the BIOMEC database
(https://felixchenier.uqam.ca/biomec) to fetch all non-personal information
about a specified project.

Please note that the user/password combination used in this tutorial is not
valid, and that you should have propel access to BIOMEC to use ktk.dbinterface.
"""
import ktk

"""
Connecting to a project in BIOMEC
---------------------------------------
The class constructor connects to the project and asks the user's credentials
and the folder where the data files are stored.

``project = ktk.DBInterface(project_label)``

For example:

``project = ktk.DBInterface.fetch_project('FC_XX18A')``

The constructor can also be run non-interactively:
"""
project_label = 'dummyProject'
username = 'dummyUser'
password = 'dummyPassword'
root_folder = 'data/dbinterface/FC_XX18A'
url = ''
url = 'http://localhost/biomec'  # This line is only for this tutorial,
                                         # please don't execute it.

project = ktk.DBInterface(project_label, user=username, password=password,
                          root_folder=root_folder, url=url)

# %%
"""
Navigating in the project
-------------------------
Just typing ``project`` gives an overview of the project's content.
"""
project

# %%
"""
The method ``get`` is used to extract the project's contents. It always returns
a dict with the fields corresponding to the request. For example:
"""
project.get()

# %%
project.get('P1')

# %%
project.get('P1')['Sessions']

# %%
project.get('P1', 'GymnaseN1')

# %%
project.get('P1', 'GymnaseN1')['Trials']

# %%
project.get('P1', 'GymnaseN1', 'Run1')

# %%
project.get('P1', 'GymnaseN1', 'Run1')['Files']

# %%
project.get('P1', 'GymnaseN1', 'Run1', 'Kinematics')

# %%
project.get('P1', 'GymnaseN1', 'Run1', 'Kinematics')['FileName']

# %%
"""
Saving data and link to BIOMEC
------------------------------
The ktk library provides the function ktk.save to save a variable to a
.ktk.zip file. The ktk.save function is helpful to save temporary results.

Sometimes we need to save results to BIOMEC so that these results become new
inputs for subsequent work. In this case, we use the dbinterface's save
method.

For example, let's say we just synchronized the kinematics for Run1 of
participant 1:
"""
synced_kinematics = 'For the demo, this will only be a string.'

"""
We can save these kinematics as a file that is referenced in BIOMEC, using:
"""
project.save('P1', 'GymnaseN1', 'Run1', 'SyncedKinematics', synced_kinematics)

"""
This creates the file entry in BIOMEC if needed, then save the file with
a relevant name into the project folder.
"""
project.get('P1', 'GymnaseN1', 'Run1', 'SyncedKinematics')['FileName']
