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

    n_frames = len(ts.time)

    ts.data['ProximalToDistalJointDistance'] = (
            ts.data['DistalJointPosition'] -
            ts.data['ProximalJointPosition'])

    ts.data['RadiusOfGyration'] = (
            gyration_ratio * ts.data['ProximalToDistalJointDistance'])

    # Center of mass position and acceleration
    ts.data['CenterOfMassPosition'] = (
            com_ratio * ts.data['ProximalToDistalJointDistance'] +
            ts.data['ProximalJointPosition'])

    ts_com = ts.get_subset('CenterOfMassPosition')
    ts_acc = ktk.filters.savgol(ts_com, window_length=21, poly_order=2,
                                deriv=2)
    ts.data['CenterOfMassAcceleration'] = ts_acc.data['CenterOfMassPosition']

    # Rotation angle, velocity and acceleration
    segment_angle_x = np.arctan2(
            ts.data['ProximalToDistalJointDistance'][:, 1],
            -ts.data['ProximalToDistalJointDistance'][:, 2])
    segment_angle_y = np.arctan2(
            -ts.data['ProximalToDistalJointDistance'][:, 2],
            ts.data['ProximalToDistalJointDistance'][:, 0])
    segment_angle_z = np.arctan2(
            ts.data['ProximalToDistalJointDistance'][:, 1],
            ts.data['ProximalToDistalJointDistance'][:, 0])
    ts.data['Angle'] = np.block([
            segment_angle_x[:, np.newaxis],
            segment_angle_y[:, np.newaxis],
            segment_angle_z[:, np.newaxis]])

    ts_angle = ts.get_subset('Angle')
    ts_angvel = ktk.filters.savgol(ts_angle, window_length=21, poly_order=2,
                                   deriv=1)
    ts_angacc = ktk.filters.savgol(ts_angle, window_length=21, poly_order=2,
                                   deriv=2)
    ts.data['AngularVelocity'] = ts_angvel.data['Angle']
    ts.data['AngularAcceleration'] = ts_angacc.data['Angle']





    # Forces line of the wrench equation (16)
    m_i__E_3x3 = segment_mass * np.eye(3)
    a_i = ts.data['CenterOfMassAcceleration'][:, 0:3]
    g = np.array([0, 9.81, 0])
    F_i_minus_1 = ts.data['DistalForces'][:, 0:3]

    # Moments line of the wrench equation (16)
    c_i = (ts.data['CenterOfMassPosition'][:, 0:3] -
           ts.data['ProximalJointPosition'][:, 0:3])

    I_i_temp = segment_mass * ts.data['RadiusOfGyration'][:, 0:3] ** 2
    # Diagonalize I_i:
    I_i = np.zeros((n_frames, 3, 3))
    I_i[:, 0, 0] = I_i_temp[:, 0]
    I_i[:, 1, 1] = I_i_temp[:, 1]
    I_i[:, 2, 2] = I_i_temp[:, 2]

    alpha_i = ts.data['AngularAcceleration']
    omega_i = ts.data['AngularVelocity']

    d_i = ts.data['ProximalToDistalJointDistance'][:, 0:3]

    M_i_minus_1 = ts.data['DistalMoments'][:, 0:3]

    # Calculation of the proximal wrench
    proximal_wrench = np.zeros((n_frames, 6, 1))
    for i_frame in range(n_frames):

        matrix_1 = np.block(
                [[segment_mass * np.eye(3), np.zeros((3,3))],
                 [segment_mass * np.diag(c_i[i_frame]), I_i[i_frame]]])

        matrix_2 = np.block([a_i[i_frame] - g, alpha_i[i_frame]])
        matrix_2 = matrix_2[:, np.newaxis]  # Convert 1d to column vector

        matrix_3 = np.block([
                np.zeros(3),
                np.cross(omega_i[i_frame], I_i[i_frame] @ omega_i[i_frame])])
        matrix_3 = matrix_3[:, np.newaxis]  # Convert 1d to column vector

        matrix_4 = np.block([
                [np.eye(3), np.zeros((3,3))],
                [np.diag(d_i[i_frame]), np.eye(3)]])

        matrix_5 = np.block([F_i_minus_1[i_frame], M_i_minus_1[i_frame]])
        matrix_5 = matrix_5[:, np.newaxis]  # Convert 1d to column vector

        proximal_wrench[i_frame] = (
                matrix_1 @ matrix_2 + matrix_3 + matrix_4 @ matrix_5)

    ts.data['ProximalForces'] = proximal_wrench[:, 0:3, 0]
    ts.data['ProximalMoments'] = proximal_wrench[:, 3:6, 0]

    return ts


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
markers.ui_sync('Sync_Sync')

rigid_bodies = kinematics['VirtualRigidBodies']

ts_all = markers.copy()
ts_all.merge(kinetics, interp_kind='linear', fill_value='extrapolate')

#%%

ts = ktk.TimeSeries(time = markers.time)

total_mass = 75


segment_mass = 2
ts.data['ProximalJointPosition'] = ts_all.data['ElbowR']
ts.data['DistalJointPosition'] = ts_all.data['RadialStyloidR']
ts.data['ForceApplicationPosition'] = ts_all.data['RearWheelCenterR']
ts.data['DistalForces'] = ts_all.data['Forces']
ts.data['DistalMoments'] = ts_all.data['Moments']

new_ts = calculate_proximal_wrench(ts, segment_mass=segment_mass,
                                   com_ratio=0.682,
                                   gyration_ratio=0.1)
