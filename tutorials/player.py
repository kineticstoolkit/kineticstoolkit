# %% markdown
"""
ktk.Player Tutorial
===================
The Player module allow visualizing markers, rigid bodies and segments in
movement and in 3D using a keyboard and mouse-driven user interface.

Loading sample data
-------------------
In this tutorial, we will load propulsion data from a sprint in Wheelchair
Basketball, as recorded using an optoelectronic system (Optitrack). Data were
preprocessed in Matlab.
"""

# %%
import ktk

kinematics = ktk.loadmat('data/inversedynamics/basketball_kinematics.mat')
kinematics = kinematics['kinematics']

kinematics

# %% markdown
"""
The player can be instanciated to show markers:
"""

# %%
pl = ktk.Player(markers=kinematics['Markers'])

# %% markdown
"""
The player can be instanciated to show rigid bodies:
"""

# %%
pl = ktk.Player(rigid_bodies=kinematics['VirtualRigidBodies'])

# %% markdown
"""
Or the player can be instanciated to show both markers and rigid bodies:
"""

# %%
pl = ktk.Player(markers=kinematics['Markers'],
                rigid_bodies=kinematics['VirtualRigidBodies'])

# %% markdown
"""
To obtain help on the different keyboard and mouse commands, press 'h'. A
help overlay will appear on top of the current view.
"""
