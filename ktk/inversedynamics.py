import ktk
import numpy as np


def calculate_proximal_wrench(ts, segment_mass, com_ratio, gyration_ratio):
    """
    Calculate the proximal wrench based on a TimeSeries.

    This function is based on R. Dumas, R. Aissaoui, and J. A. De Guise,
    "A 3D generic inverse dynamic method using wrench notation and quaternion
    algebra,” Comput Meth Biomech Biomed Eng, vol. 7, no. 3, pp. 159–166, 2004.

    Parameters
    ----------
        ts : TimeSeries
            A TimeSeries with the following data keys:
                - ProximalJointPosition (Nx4)
                - DistalJointPosition (Nx4)
                - ForceApplicationPosition (Nx4)
                - DistalForces (Nx4)
                - DistalMoments (Nx4)
        segment_mass : float
            The mass of the segment in kg.
        com_ratio : float
            The position of the segment's center of mass, as a proximal ratio
            of the segment length.
        gyration_ratio : float
            The radius of gyration around the segment's center of mass, as a
            ratio of the segment length.

    Returns
    -------
    The input timeseries, with these added data keys:
        - ProximalForces (Nx4)
        - ProximalMoments (Nx4)
    """
#%%
    ts = ts.copy()

    ts.data['CenterOfMassPosition'] = (com_ratio * (
            ts.data['DistalJointPosition'] -
            ts.data['ProximalJointPosition']) +
            ts.data['ProximalJointPosition'])

    ts.data['RadiusOfGyration'] = gyration_ratio * (
            ts.data['DistalJointPosition'] -
            ts.data['ProximalJointPosition'])

    ts_com = ts.get_subset('CenterOfMassPosition')
    ts_acc = ktk.filters.savgol(ts_com, window_length=21, poly_order=2,
                                deriv=2)
    ts.data['CenterOfMassAcceleration'] = ts_acc.data['CenterOfMassPosition']

    # Forces line of the wrench equation (16)
    m_i__E_3x3 = segment_mass * np.eye(3)
    a_i = ts.data['CenterOfMassAcceleration'][:,0:3]
    g = np.array([0, 9.81, 0])
    F_i_minus_1 = ts.data['DistalForces']

    # Moments line of the wrench equation (16)
    m_i__c_1 = np.diag(segment_mass * (
            ts.data['CenterOfMassPosition'][:, 0:3]
            - ts.data['ProximalJointPosition'][:, 0:3]))
    I_i = np.diag(segment_mass * ts.data['RadiusOfGyration'][:, 0:3] ** 2)

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

total_mass = 75


segment_mass = 2
ts.data['ProximalJointPosition'] = markers.data['ElbowR']
ts.data['DistalJointPosition'] = markers.data['RadialStyloidR']
ts.data['ForceApplicationPosition'] = markers.data['RearWheelCenterR']
ts.data['DistalForces'] = kinetics.data['Forces']

calculate_proximal_wrench(ts, segment_mass=segment_mass, com_ratio=0.682,
                          gyration_ratio=0.1)
