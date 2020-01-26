# %%
"""
kinematics
==========
At the current time, the kinematics module only allows loading c3d files
containing 3d markers, and returning those markers as TimeSeries.
"""
import ktk
import numpy as np

# %%
"""
Defining the files to process
-----------------------------
As there as many data files to process, we will begin by creating a dictionary
of all the file names, which will simplify the processing later.
"""
file_names = dict()

# Static trial where we see all markers without movement
file_names['Static'] = 'data/kinematics/sample_static.n3d'

# Some probing trials
file_names['ProbingAcromionR'] = \
        'data/kinematics/sample_probing_acromion_R.n3d'
file_names['ProbingMedialEpicondyleR'] = \
        'data/kinematics/sample_probing_medial_epicondyle_R.n3d'
file_names['ProbingOlecraneR'] = \
        'data/kinematics/sample_probing_olecrane_R.n3d'
file_names['ProbingRadialStyloidR'] = \
        'data/kinematics/sample_probing_radial_styloid_R.n3d'
file_names['ProbingUlnarStyloidR'] = \
        'data/kinematics/sample_probing_ulnar_styloid_R.n3d'

# A propulsion sample to analyze
file_names['Propulsion'] = 'data/kinematics/sample_propulsion.n3d'

# %%
"""
Defining marker names
---------------------
In these samples, the marker names are not included in the file name. We do
however know the markers order as recorded by Optotrak. If we had used
another system such as Optitrack or Vicon, the marker names would be
included in the c3d files.
"""

marker_names = ['Probe1', 'Probe2', 'Probe3', 'Probe4', 'Probe5', 'Probe6',
                'WheelchairRearR', 'WheelchairRearL', 'WheelchairFront',
                'ScapulaL1', 'ScapulaL2', 'ScapulaL3',
                'ScapulaR1', 'ScapulaR2', 'ScapulaR3',
                'Head1', 'Head2', 'Head3',
                'Sternum',
                'ArmL1', 'ArmL2', 'ArmL3',
                'LatEpicondyleL', 'ForearmL1', 'ForearmL2', 'ForearmL3', 'NAG',
                'GloveL1', 'GloveL2', 'GloveL3',
                'ArmR1', 'ArmR2', 'ArmR3',
                'LatEpicondyleR', 'ForearmR1', 'ForearmR2', 'ForearmR3', 'NAR',
                'GloveR1', 'GloveR2', 'GloveR3']

# %%
"""
Defining the rigid body configurations using the static trial
-------------------------------------------------------------
One of the aims of the static trial is to have a sample where every marker
is visible. We use this trial to define the rigid body configuration.
A rigid body configuration is a list of markers that form a rigid body,
along with their local position in the rigid body's reference frame.

For this example, we will create rigid bodies for the markers triads
'ArmR' and 'ForearmR'.
"""
config = dict()
config['RigidBodies'] = dict()

# Read the static trial
markers = ktk.kinematics.read_n3d_file(file_names['Static'],
                                       labels=marker_names)

# Create the rigid body configurations
config['RigidBodies']['ArmR'] = ktk.kinematics.create_rigid_body_config(
        markers, ['ArmR1', 'ArmR2', 'ArmR3'])

config['RigidBodies']['ForearmR'] = ktk.kinematics.create_rigid_body_config(
        markers, ['ForearmR1', 'ForearmR2', 'ForearmR3'])

"""
The rigid body configuration will be created manually as we already have
the probe configuration from its specifications. Each local point is expressed
relative to a reference frame that is centered at the probe's tip.
"""

config['RigidBodies']['Probe'] = {
        'MarkerNames': ['Probe1', 'Probe2', 'Probe3',
                        'Probe4', 'Probe5', 'Probe6'],
        'LocalPoints': np.array([[
                [2.1213,   2.1213,  2.0575,   2.1213,   1.7070,   1.7762],
                [-15.8328, 15.8508, 16.0096,  16.1204,  -15.5780, -15.6057],
                [86.4285,  86.4285, 130.9445, 175.4395, 175.3805, 130.8888],
                [1000,     1000,    1000,     1000,     1000,     1000]]]
                ) / 1000
        }

# %%
"""
Add the probe tip on a probing acquisition.
"""

markers = ktk.kinematics.read_n3d_file(file_names['ProbingAcromionR'],
                                       labels=marker_names)
rigid_bodies = ktk.kinematics.register_markers(
        markers, config['RigidBodies'])
markers.data['ProbeTip'] = rigid_bodies.data['Probe'][:, :, 3]
markers.add_data_info('ProbeTip', 'Color', 'r')



# %%

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
