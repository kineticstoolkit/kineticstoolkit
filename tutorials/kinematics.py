"""
kinematics
==========
At the current time, the kinematics module only allows loading c3d files
containing 3d markers, and returning those markers as TimeSeries.
"""
import ktk

"""
We will load a c3d file with wheelchair basketball sprinting data.
"""
ts = ktk.kinematics.read_c3d_file('data/kinematics/sprintbasket.c3d')

# %% exclude
# Regression tests with a reference mat made on KTK for Matlab
import numpy as np
ref = ktk.loadmat('data/kinematics/sprintbasket.mat')
# ref seems to miss a list sample. We will then samples compare 0:-1.
for label in ts.data.keys():
    reflabel = label.replace(':', '_')
    assert(np.nanmean(
            np.abs(ts.data[label][0:-1] - ref.data[reflabel])) < 1E-6)

# %%
"""
We can now show these markers in a Player. TODO The orientation needs to be
corrected.
"""
player = ktk.Player(markers=ts)
