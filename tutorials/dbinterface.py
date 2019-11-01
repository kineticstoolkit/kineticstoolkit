# %% markdown
"""
ktk.dbinterface Tutorial
========================
ktk.dbinterface connects to the BIOMEC database
(https://felixchenier.uqam.ca/biomec) to fetch all non-personal information
about a specified project.

Please note that the user/password combination used in this tutorial is not
valid, and that you should have propel access to BIOMEC to use ktk.dbinterface.
"""

# %%
import ktk

# %% markdown
"""
Fetching a complete project from BIOMEC
---------------------------------------
The easiest way to fetch a project is to use the fetch_project function:

``project = ktk.dbinterface.fetch_project(projectLabel)``

which is an interactive function. For example:

``project = ktk.dbinterface.fetch_project('FC_XX18A')``

The fetchproject function can also be run non-interactively:
"""

# %%
project_label = 'dummyProject'
username = 'dummyUser'
password = 'dummyPassword'
root_folder = 'data/dbinterface/FC_XX18A'
url = ''
url = 'http://localhost/biomec.uqam.ca'  # This line is only for this tutorial, please don't execute it.

project = ktk.dbinterface.fetch_project(project_label, user=username, password=password, root_folder=root_folder,
                                       url=url)

# %% markdown
"""
Navigating in the project structure
-----------------------------------
The result of ktk.dbinterface.fetch_project is a nested dict with all the project information. Here are some examples to access all the different values contained in this tree.
"""

# %%
project

# %%
project['Participants']

# %%
project['Participants']['P1']

# %%
project['Participants']['P1']['Sessions']

# %%
project['Participants']['P1']['Sessions']['GymnaseN1']

# %%
project['Participants']['P1']['Sessions']['GymnaseN1']['Trials']

# %%
project['Participants']['P1']['Sessions']['GymnaseN1']['Trials']['Run1']

# %%
project['Participants']['P1']['Sessions']['GymnaseN1']['Trials']['Run1']['Files']

# %%
project['Participants']['P1']['Sessions']['GymnaseN1']['Trials']['Run1']['Files']['Kinematics']

# %%
project['Participants']['P1']['Sessions']['GymnaseN1']['Trials']['Run1']['Files']['Kinematics']['FileName']