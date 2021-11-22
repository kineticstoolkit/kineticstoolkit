---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Wheelchair kinetics

The [pushrimkinetics](../../api/kineticstoolkit.pushrimkinetics.rst) module allows processing kinetics data from instrumented wheelchair wheels such as the SmartWheel.

```{code-cell} ipython3
import kineticstoolkit.lab as ktk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
```

## Read data from file

The first step is to load data from a file, using the [pushrimkinetics.read_file()](../../api/kineticstoolkit.pushrimkinetics.read_file.rst) function.

```{code-cell} ipython3
filename = (
    ktk.config.root_folder +
    '/data/pushrimkinetics/sample_swl_overground_propulsion_withrubber.csv')

kinetics = ktk.pushrimkinetics.read_file(filename, file_format='smartwheel')
```

Let see what we loaded:

```{code-cell} ipython3
kinetics
```

```{code-cell} ipython3
kinetics.data
```

```{code-cell} ipython3
plt.subplot(2, 1, 1)
kinetics.plot('Forces')
plt.subplot(2, 1, 2)
kinetics.plot('Moments')
plt.tight_layout()
```

## Calculate forces and moments

If the loaded data doesn't include forces and moments but only raw data (for exemple, when loading data from a SmartWheel's SD card), we must calculate the forces and moments based on a calibration matrix. The function [pushrimkinetics.calculate_forces_and_moments()](../../api/kineticstoolkit.pushrimkinetics.calculate_forces_and_moments.rst) performs this calculation. In this example, we use a calibration matrix that is included in `ktk.pushrimkinetics.CALIBRATION_MATRICES`, and we express the calculated forces and moments in a reference frame that is orthogonal to the ground (levelled using the wheel's angular encoder).

```{code-cell} ipython3
calibration_matrices = ktk.pushrimkinetics.CALIBRATION_MATRICES['SmartWheel_123']
calibration_matrices
```

```{code-cell} ipython3
new_kinetics = ktk.pushrimkinetics.calculate_forces_and_moments(
    kinetics,
    gains=calibration_matrices['gains'],
    offsets=calibration_matrices['offsets'],
    transducer='smartwheel',
    reference_frame='hub')

plt.subplot(2, 1, 1)
new_kinetics.plot('Forces')
plt.subplot(2, 1, 2)
new_kinetics.plot('Moments')
plt.tight_layout()
```

We observe some sign differences here. In fact, the SmartWheel softwares inverts some signals based on the SmartWheel's sign convention and the side of the wheel. When the forces and moments are reconstructed from raw data, these sign changes do not happen, which explains these discrepancies.

## Removing dynamic offsets in forces and moments

We observe in the last graphs that dynamic (sinusoidal) offsets are presents mostly in the forces but also in the moments. We can auto-remove these offsets using [pushrimkinetics.remove_offsets()](../../api/kineticstoolkit.pushrimkinetics.remove_offsets.rst).

Let's apply this function on the data we just loaded.

```{code-cell} ipython3
plt.subplot(2, 1, 1)
kinetics.plot('Forces')
plt.title('Before removing offsets')
plt.tight_layout()
```

```{code-cell} ipython3
kinetics = ktk.pushrimkinetics.remove_offsets(kinetics)

plt.subplot(2, 1, 2)
kinetics.plot('Forces')
plt.title('After removing offsets')
plt.tight_layout()
```

This automatic method has only be validated for straight-line, level-ground propulsion. For any other condition, a baseline trial is required. A baseline trial is a trial where an operator pushes the wheelchair but no external force appart from gravity is applied on the instrumented wheel. Please consult the [pushrimkinetics.remove_offsets()](../../api/kineticstoolkit.pushrimkinetics.remove_offsets.rst) function help for more information.

## Calculate velocity and power

Thee wheel velocity is calculated from the wheel angle with a derivative Savitsky-Golay filter, using the [pushrimkinetics.calculate_velocity()](../../api/kineticstoolkit.pushrimkinetics.calculate_velocity.rst) function. Once the velocity has been calculated, the output power can also be calculated by
multiplying the velocity by the propulsion moment, using the [pushrimkinetics.calculate_power()](../../api/kineticstoolkit.pushrimkinetics.calculate_power.rst) function.

```{code-cell} ipython3
kinetics = ktk.pushrimkinetics.calculate_velocity(kinetics)
kinetics = ktk.pushrimkinetics.calculate_power(kinetics)

plt.subplot(2, 1, 1)
kinetics.plot('Velocity')
plt.subplot(2, 1, 2)
kinetics.plot('Power')
plt.tight_layout()
```

## Detecting the pushes

The [cycles](../24_cycles/00_cycles.rst) module provides powerful tools to detect and manage cycles. Here we use [cycles.detect_cycles()](../../api/kineticstoolkit.cycles.detect_cycles.rst) to detect the pushes using these specifications:
- a push starts when the total force crosses 5 N upward;
- a push ends when the total force crosses 2 N downward;
- for a push to be valid, it must last at least 100 ms;
- to be valid, the total force must reach 25 N.

```{code-cell} ipython3
# Create the total force data
kinetics.data['Ftot'] = np.sqrt(np.sum(kinetics.data['Forces'] ** 2, axis=1))

kinetics = ktk.cycles.detect_cycles(
    kinetics, 'Ftot',
    event_names=['push', 'recovery'],
    thresholds=[5.0, 2.0],
    min_durations=[0.1, 0.1],
    min_peak_heights=[25.0, -np.Inf]
)

kinetics.plot('Forces')
plt.tight_layout()
```

## Extracting spatiotemporal and kinetic parameters

As a conclusion to this tutorial, we will now extract some key spatiotemporal and kinetic parameters from these data, and express those parameters as a pandas DataFrame. Obviously, this is only an example and many other parameters can be calculated using a similar procedure.

```{code-cell} ipython3
n_cycles = 15  # Number of cycles to analyze

records = []  # Init a list that will contains the results of the analysis

for i_cycle in range(n_cycles):
    
    # Get a TimeSeries that spans only the push i_push
    ts_push = kinetics.get_ts_between_events('push', 'recovery', i_cycle, i_cycle)
    
    # Get a TimeSeries that spans the entire cycle i_push
    ts_cycle = kinetics.get_ts_between_events('push', '_', i_cycle, i_cycle)
    
    # Get some spatiotemporal parameters
    push_time = ts_push.time[-1] - ts_push.time[0]
    cycle_time = ts_cycle.time[-1] - ts_cycle.time[0]
    recovery_time = cycle_time - push_time

    push_angle = ts_push.data['Angle'][-1] - ts_push.data['Angle'][0]
    
    # Get some kinetic parameters
    propulsion_moment_mean = np.mean(ts_push.data['Moments'][:, 2])
    propulsion_moment_max = np.max(ts_push.data['Moments'][:, 2])
    
    total_force_mean = np.mean(ts_push.data['Ftot'])
    total_force_max = np.max(ts_push.data['Ftot'])
    
    # Record this information in the records list
    records.append({
        'Push time (s)': push_time,
        'Recovery time (s)': recovery_time,
        'Cycle time (s)': cycle_time,
        'Push angle (deg)': np.rad2deg(push_angle),
        'Mean propulsion moment (Nm)': propulsion_moment_mean,
        'Max propulsion moment (Nm)': propulsion_moment_max,
        'Mean total force (N)': total_force_mean,
        'Max total force (N)': total_force_max,
    })

# Create and show a DataFrame of this information
df = pd.DataFrame.from_dict(records)

# Copy the dataframe to the clipboard for pasting into Excel (facultative)
df.to_clipboard()

# Print the dataframe here
df
```

For more information, please check the [API Reference for the pushrimkinetics module](../../api/kineticstoolkit.pushrimkinetics.rst).
