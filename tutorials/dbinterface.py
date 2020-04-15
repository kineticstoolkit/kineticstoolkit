"""
DBInterface
========================
The ktk.DBInterface class interfaces with the BIOMEC database
(https://felixchenier.uqam.ca/biomec) to fetch all non-personal information
about a specified project.

Please note that the user/password combination used in this tutorial is not
valid, and that you should have propel access to BIOMEC to use ktk.dbinterface.
"""

# %% exclude

# NOTE ON TEST DATABASE AND DUMMY USER.

# ktkDBInterfaceTest and ktkDBInterfaceTutorial run on a local database
# that has the same format as the one on biomec.uqam.ca. However, for the
# tests and tutorials to pass, a special user (dummyUser) must exist and
# have rights on a specially crafter project (dummyProject).

# To add this user, enter these SQL commands in mysql:

# SELECT ProjectID FROM Projects WHERE ProjectLabel = "dummyProject" INTO @ProjectID;
# INSERT INTO Users (username, password, role) VALUES ("dummyUser", "0726aee645e102b5607e7ed5ad4a029a", "STUDENT");
# SELECT UserID FROM Users WHERE username = "dummyUser" INTO @UserID;
# INSERT INTO ProjectsUsers (ProjectID, UserID) VALUES (@ProjectID, @UserID);

# ---------
# IMPORTANT
# ---------
# For security reasons, the dummyUser must not exist on the real database
# %iomec.uqam.ca. To remove this user, run these SQL commands in mysql:

# SELECT ProjectID FROM Projects WHERE ProjectLabel = "dummyProject" INTO @ProjectID;
# SELECT UserID FROM Users WHERE username = "dummyUser" INTO @UserID;
# SELECT ProjectUserID FROM ProjectsUsers WHERE ProjectID = @ProjectID AND UserID = @UserID INTO @ProjectUserID;
# DELETE FROM ProjectsUsers WHERE ProjectUserID = @ProjectUserID;
# DELETE FROM Users WHERE UserID = @UserID;


# %%
import ktk
import os
import shutil

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
Saving data to a BIOMEC referenced file
---------------------------------------
The ktk library provides the function ktk.save to save a variable to a
.ktk.zip file. The ktk.save function is helpful to save temporary results.

However, sometimes we need to save results to BIOMEC so that these results
become new inputs for subsequent work. In this case, we use the
ktk.DBInterface's ``save`` method.

For example, let's say we just synchronized the kinematics for Run1 of
participant 1:
"""
synced_kinematics = {'dummy_data':
                     'Normally we would save something more useful'}

"""
We can save these kinematics as a file that is referenced in BIOMEC, using:
"""
project.save('P1', 'GymnaseN1', 'Run1', 'SyncedKinematics', synced_kinematics)

"""
This creates the file entry in BIOMEC if needed, then save the file with
a relevant name into the project folder.
"""
project.get('P1', 'GymnaseN1', 'Run1', 'SyncedKinematics')['FileName']

"""
Loading data from a BIOMEC referenced file
------------------------------------------
To load back data saved to BIOMEC, we use the ktk.DBInterface's ``load``
method.
"""
test = project.load('P1', 'GymnaseN1', 'Run1', 'SyncedKinematics')

test

# %%
"""
We do a little clean up before going on.
"""
shutil.rmtree(root_folder + '/SyncedKinematics')


# %%
"""
Dealing with external software
------------------------------
The DBInterface's ``save`` and ``load`` methods work very well for data that
were processed in Python using ktk. However, things may get complicated when
using external software to process data.

In this example, we will synchronize the kinematics using an external
synchronizing tool, then enter the resulting files into BIOMEC. We will work
with these files:
"""
file_list = []
for trial in ['Walk1', 'Walk2', 'Run1', 'Run2']:
    file_list.append(project.get(
        'P1', 'GymnaseN1', trial, 'Kinematics')['FileName'])

file_list

"""
Let say we synchronized these files using an external software, and then we
exported the synchronized files into a separate folder.

(Here we will simply copy those files into a separate folder).
"""
os.mkdir(root_folder + '/synchronized_files')

for file in file_list:
    dest_file = file.replace(root_folder, root_folder + '/synchronized_files')
    shutil.copyfile(file, dest_file)

os.listdir(root_folder + '/synchronized_files')

"""
All is good, but the dbfids in the new ``synchronized_files`` folder refer to
the original ``Kinematics`` file type, not to the ``SyncedKinematics`` file
type. Moreover, there are now duplicate dbfids in the project:
"""
project.refresh()
# %%
project.duplicates

# %%
"""
Therefore we need to assign new dbfids to the files we just synchronized, so
that they refer to ``SyncedKinematics`` entries in BIOMEC. The method
``batch_fix_file_type`` will help.
"""
project.batch_fix_file_type(root_folder + '/synchronized_files',
                            'SyncedKinematics',
                            create_file_entries=True,
                            dry_run=False)

"""
Now let see what happened in the ``synchronized_files`` folder:
"""
os.listdir(root_folder + '/synchronized_files')

"""
The files' dbfid have been updated so they now refer to ``SyncedKinematics``
and not to ``Kinematics`` anymore. Moreover, the project does not have
duplicate dbfids anymore:
"""
project.refresh()

# %%
"""
We do a little clean up before going on.
"""
shutil.rmtree(root_folder + '/synchronized_files')
project.refresh()
# %%

"""
Including information in file names
-----------------------------------
It can be difficult to deal with a bunch of numbered files without knowing
their signification without looking in BIOMEC. The ``tag_files`` method
allows adding the trial name to the file names, so that their context is
a bit clearer and less error-prone.
"""

os.listdir(root_folder)

"""
Include the trial name in the file names:
"""
project.tag_files(include_trial_name=True, dry_run=False)
# %%
os.listdir(root_folder)

"""
Remove the trial name from the file names:
"""
project.tag_files(include_trial_name=False, dry_run=False)
# %%
os.listdir(root_folder)
