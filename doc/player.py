# %%
"""
Player
======
The Player class allows visualizing markers, rigid bodies and segments in
movement and in 3D using a keyboard and mouse-driven user interface.

This tutorial is still incomplete. Please see the `kinematics` tutorial, which
make other uses of the Player class.
"""
import ktk

"""
Loading sample data
-------------------
In this tutorial, we will load propulsion data from a sprint in Wheelchair
Basketball, as recorded using an optoelectronic system (Optitrack). Data were
preprocessed in Matlab.
"""
kinematics = ktk.loadmat('data/inversedynamics/basketball_kinematics.mat')
kinematics = kinematics['kinematics']

kinematics

# %%
"""
The player can be instanciated to show markers:
"""
pl = ktk.Player(markers=kinematics['Markers'], target=[-5, 0, 0])

# %%
"""
It is possible to colorize markers, using one of these letters:
    - (r)ed
    - (g)reen
    - (b)lue
    - (y)ellow
    - (m)agenta
    - (c)yan
    - (w)hite
"""
kinematics['Markers'].add_data_info('Sync_Sync', 'Color', 'y')
kinematics['Markers'].add_data_info('SWR_Marker1', 'Color', 'r')
kinematics['Markers'].add_data_info('SWR_Marker2', 'Color', 'r')
kinematics['Markers'].add_data_info('SWR_Marker3', 'Color', 'r')
kinematics['Markers'].add_data_info('SWR_Marker4', 'Color', 'r')
pl = ktk.Player(markers=kinematics['Markers'], target=[-5, 0, 0])

# %%
"""
The player can be instanciated to show rigid bodies:
"""
pl = ktk.Player(rigid_bodies=kinematics['VirtualRigidBodies'],
                target=[-5, 0, 0])

# %%
"""
Or the player can be instanciated to show both markers and rigid bodies:
"""
pl = ktk.Player(markers=kinematics['Markers'],
                rigid_bodies=kinematics['VirtualRigidBodies'],
                target=[-5, 0, 0])

# %%
"""
To obtain help on the different keyboard and mouse commands, press 'h'. A
help overlay will appear on top of the current view.
"""
