# %%
"""
kinematics
==========
At the current time, the kinematics module only allows loading c3d files
containing 3d markers, and returning those markers as TimeSeries.
"""
import ktk
import numpy as np

"""
We will load a c3d file with wheelchair basketball sprinting data.
"""
ts = ktk.kinematics.read_c3d_file('data/kinematics/sprintbasket.c3d')

ts

# %% exclude
# Regression tests with a reference mat made on KTK for Matlab
ref = ktk.loadmat('data/kinematics/sprintbasket.mat')
# ref seems to miss a list sample. We will then samples compare 0:-1.
for label in ts.data.keys():
    reflabel = label.replace(':', '_')
    assert(np.nanmean(
            np.abs(ts.data[label][0:-1] - ref.data[reflabel])) < 1E-6)

# %%
"""
This results in a TimeSeries where each data corresponds to a marker's
trajectory.
"""
ts.data

# %%
"""
In this c3d file, the global reference frame is:
    - x anterior
    - y left
    - z up

In the ktk.Player convention, the global reference frame is:
    - x anterior
    - y up
    - z right

Thus we need to rotate each marker's coordinates 90 degrees clockwise around
the x axis.
"""
R = ktk.geometry.create_rotation_matrices('x', [-np.pi/2])
for key in ts.data:
    ts.data[key] = ktk.geometry.matmul(R, ts.data[key])

# %%
"""
We can now show these markers in a Player.
"""
player = ktk.Player(markers=ts)
