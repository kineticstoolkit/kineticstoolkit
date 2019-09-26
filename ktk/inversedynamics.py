import ktk
import numpy as np


def calculate_proximal_wrench(ts):
    """
    Calculate the proximal wrench based on a TimeSeries.

    This function is based on R. Dumas, R. Aissaoui, and J. A. De Guise,
    "A 3D generic inverse dynamic method using wrench notation and quaternion
    algebra,” Comput Meth Biomech Biomed Eng, vol. 7, no. 3, pp. 159–166, 2004.
    """
#%%
    ts_com = ts.get_subset('CenterOfMassPosition')
    ts_acc = ktk.filters.savgol(ts_com, window_length=21, poly_order=2,
                                deriv=2)
    ts.data['CenterOfMassAcceleration'] = ts_acc.data['CenterOfMassPosition']

    segment_mass = 2  # Mass of the segment

    m = segment_mass * np.eye(3)
    g = np.array([0, 9.81, 0])


    print(m)

#%%

kinetics = ktk.loadmat(
        ktk.config['RootFolder'] +
        '/tutorials/data/inversedynamics/basketball_kinetics.mat')
kinetics = kinetics['kineticsSWR']

kinematics = ktk.loadmat(
        ktk.config['RootFolder'] +
        '/tutorials/data/inversedynamics/basketball_kinematics.mat')
kinematics = kinematics['kinematics']

markers = kinematics['Markers']
rigid_bodies = kinematics['VirtualRigidBodies']

#%%

ts = ktk.TimeSeries(time = markers.time)

ts.data['ProximalJointPosition'] = markers.data['ElbowR']
ts.data['DistalJointPosition'] = markers.data['RadialStyloidR']
ts.data['CenterOfMassPosition'] = (0.682 *
        (ts.data['DistalJointPosition'] - ts.data['ProximalJointPosition']) +
        ts.data['ProximalJointPosition'])
ts.data['ForceApplicationPosition'] = markers.data['RearWheelCenterR']

calculate_proximal_wrench(ts)
