# %%
"""
pushrimkinetics
===============
The pushrimkinetics module allows processing kinetics data from instrumented
wheelchair wheels such as the SmartWheel.
"""
import ktk
import matplotlib.pyplot as plt

# %% exclude
# Additionnal imports and functions for unit tests
import numpy as np

def _assert_almost_equal(float1, float2):
    assert abs(float1 - float2) < 1E-6

# %%
"""
Read data from file
-------------------
The first step is to load data from a file. This is done using ``read_file``:
"""
filename = ('data/pushrimkinetics/'
            'sample_swl_overground_propulsion_withrubber.csv')
kinetics = ktk.pushrimkinetics.read_file(filename)

# %% exclude
# Non-regression test based on the Matlab's KTK tutorial.
_assert_almost_equal(np.mean(kinetics.data['Forces']),
                     -0.0044330903410570)
_assert_almost_equal(np.mean(kinetics.data['Moments']),
                     0.5374323092944534)
_assert_almost_equal(np.mean(kinetics.data['Angle']),
                     46.4698216459348359)
_assert_almost_equal(np.mean(kinetics.data['Channels']),
                     2059.6018397986695163)
_assert_almost_equal(np.mean(kinetics.data['Index']),
                     3841.5000000000000000)

# %%
"""
Now see what we just loaded.
"""
kinetics

# %%
kinetics.data

# %%
plt.figure()
kinetics.plot(['Forces', 'Moments'])

# %%
"""
Calculate forces and moments
----------------------------
If the source file is not a CSV file (from the SmartWheel software) but a TXT
file from the SmartWheel's SD Card, then the file contains only the raw
channels, index and angle. In this case, we must calculate the forces and
moments based on a calibration matrix. The function
``calculate_forces_and_moments`` does this calculation and already includes
calibration matrices based on SmartWheels' serial numbers. For example:
"""
new_kinetics = ktk.pushrimkinetics.calculate_forces_and_moments(
            kinetics, 'LIO-123')

# %% exclude
forces = np.nanmean(new_kinetics.data['Forces'], 0)
moments = np.nanmean(new_kinetics.data['Moments'], 0)
_assert_almost_equal(forces[0], -8.849994801918)
_assert_almost_equal(forces[1], -11.672364564453)
_assert_almost_equal(forces[2], -2.646989586045)
_assert_almost_equal(moments[0], -0.039625979603)
_assert_almost_equal(moments[1], -0.088833025939)
_assert_almost_equal(moments[2], 2.297597031073)

# %%
"""
Removing sinusoids in forces and moments
----------------------------------------
We observe in the last graphs that sinusoidal offsets are presents mostly in
the forces but also in the moments. We can auto-remove these offsets using
``remove_sinusoids``.

Let's apply this function on the data we just loaded.
"""
kinetics = ktk.pushrimkinetics.remove_sinusoids(kinetics)

plt.figure()
kinetics.plot(['Forces', 'Moments'])

# %% exclude
kinetics = ktk.pushrimkinetics.read_file(filename)  # reload from csv
kinetics = ktk.pushrimkinetics.remove_sinusoids(kinetics)
_assert_almost_equal(np.mean(kinetics.data['Forces']),
                     1.2971684579009064)
_assert_almost_equal(np.mean(kinetics.data['Moments']),
                     0.4972708141781993)

# %%
"""
This automatic method has only be validated for straight-line, level-ground
propulsion. For any other condition, a baseline trial is required. A baseline
trial is a trial where an operator pushes the wheelchair but no external
force appart from gravity is applied on the instrumented wheel. Let's see an
example.
"""
kinetics = ktk.pushrimkinetics.read_file(
        'data/pushrimkinetics/sample_swl_overground_propulsion_withrubber.csv')
baseline = ktk.pushrimkinetics.read_file(
        'data/pushrimkinetics/sample_swl_overground_baseline_withrubber.csv')
kinetics = ktk.pushrimkinetics.remove_sinusoids(kinetics, baseline)

plt.figure()
kinetics.plot(['Forces', 'Moments'])

# %% exclude
_assert_almost_equal(np.mean(kinetics.data['Forces']),
                     1.4048102831351081)

# %%
"""
Calculate velocity and power
----------------------------
Thee wheel velocity is calculated from the wheel angle with a derivative
Savitsky-Golay filter, using the ``calculate_velocity`` function. Once the
velocity has been calculated, the output power can also be calculated by
multiplying the velocity by the propulsion moment, using the
``calculate_power`` function.
"""
kinetics = ktk.pushrimkinetics.calculate_velocity(kinetics)
kinetics = ktk.pushrimkinetics.calculate_power(kinetics)

plt.figure()
kinetics.plot(['Velocity', 'Power'])

# %%
"""
Detecting pushes
----------------

The function ``detect_pushes`` allows detecting pushes and recoveries
automatically based on a double-threshold. Let's try it on our data.
"""
kinetics = ktk.pushrimkinetics.detect_pushes(kinetics)

kinetics

# %% exclude
float_event_times = []
event_times = np.array(kinetics.events)[:, 0]
for i in range(0, len(event_times)):
    float_event_times.append(float(event_times[i]))

assert len(kinetics.events) == 77
_assert_almost_equal(np.mean(float_event_times), 17.212175000000002)

# %%
"""
We see that the TimeSeries now has 77 items. Let's see these events on a plot.
"""
plt.figure()
kinetics.plot(['Forces', 'Moments'])

# %%
"""
Time-normalizing data
---------------------

Now, we may be intested in time-normalizing our pushes. For example, if we are
interested to find the average progression of the push force. Time-normalizing
is not specific to the ktk.pushrimkinetics module, it is in the ktk.cycles
module.

Let's say we want to time-normalize each push from the ``pushstart`` event to
the ``pushend`` event.
"""
kinetics = ktk.cycles.time_normalize(kinetics, 'pushstart', 'pushend')

plt.figure()
kinetics.plot(['Forces', 'Moments'])

# %%
"""
It is now possible to extract each push in a i_cycle x i_percent x i_component
form, using the ``ktk.TimeSeries.get_reshaped_time_normalized_data`` method.
"""
data = ktk.cycles.get_reshaped_time_normalized_data(kinetics)

data

# %%
plt.figure()

for i in range(data['Forces'].shape[0]):
    plt.plot(data['Forces'][i])
