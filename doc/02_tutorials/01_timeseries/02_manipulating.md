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

# Manipulating TimeSeries

The TimeSeries come with various method to copy, split, extract or combine data:

- [TimeSeries.copy()](../../api/kineticstoolkit.TimeSeries.copy.rst) to make a deep copy of a TimeSeries;
- [TimeSeries.get_subset()](../../api/kineticstoolkit.TimeSeries.get_subset.rst) to make a deep copy of a TimeSeries, but only with a selected subset of data;
- [TimeSeries.merge()](../../api/kineticstoolkit.TimeSeries.merge.rst) to merge two TimeSeries with a same time vector together;
- [TimeSeries.get_ts_before_index()](../../api/kineticstoolkit.TimeSeries.get_ts_before_index.rst),
  [TimeSeries.get_ts_after_index()](../../api/kineticstoolkit.TimeSeries.get_ts_after_index.rst),
  [TimeSeries.get_ts_between_indexes()](../../api/kineticstoolkit.TimeSeries.get_ts_between_indexes.rst),
  [TimeSeries.get_ts_before_time()](../../api/kineticstoolkit.TimeSeries.get_ts_before_time.rst),
  [TimeSeries.get_ts_after_time()](../../api/kineticstoolkit.TimeSeries.get_ts_after_time.rst),
  [TimeSeries.get_ts_between_times()](../../api/kineticstoolkit.TimeSeries.get_ts_between_times.rst),
  [TimeSeries.get_ts_before_event()](../../api/kineticstoolkit.TimeSeries.get_ts_before_event.rst),
  [TimeSeries.get_ts_after_event()](../../api/kineticstoolkit.TimeSeries.get_ts_after_event.rst),
  [TimeSeries.get_ts_between_events()](../../api/kineticstoolkit.TimeSeries.get_ts_between_events.rst) to split a TimeSeries in time following specific criteria.

In this tutorial, we will see how to use these methods to manage a TimeSeries of marker trajectories. We will start by loading a sample `c3d` file with some marker trajectories. This example has 26 markers with 3678 samples recorded at 120 Hz.

```{code-cell}
import kineticstoolkit.lab as ktk

markers = ktk.kinematics.read_c3d_file(
    ktk.config.root_folder
    + '/data/kinematics/sprintbasket.c3d'
)

markers
```

```{code-cell} ipython3
markers.data
```

## Copying a TimeSeries

As for most class instances in Python, a TimeSeries is a mutable type. This means that for a TimeSeries `ts1`, `ts2 = ts1` creates a second reference to the same TimeSeries. This means that modifying `ts2` will also modify `ts1`.

To create a completely independent copy of a TimeSeries, we use the [TimeSeries.copy()](../../api/kineticstoolkit.TimeSeries.copy.rst) method:

```{code-cell}
markers_copy = markers.copy()

markers_copy
```

Interestingly, [TimeSeries.copy()](../../api/kineticstoolkit.TimeSeries.copy.rst) has different arguments to select which attributes to copy. For instance, if we want to create an empty TimeSeries, but with the same time and events as the source, we could use:

```{code-cell}
markers_copy = markers.copy(copy_data=False, copy_data_info=False)

markers_copy
```

## Subsetting and merging TimeSeries

The [TimeSeries.get_subset()](../../api/kineticstoolkit.TimeSeries.get_subset.rst) method allows copying a TimeSeries with only a subset of the original TimeSeries. For example, in the markers TimeSeries, we may be interested only in the markers `BodyL:AcromionL` and `BodyL:LateralEpicondyleL`. To copy only these markers, we would use:

```{code-cell}
markers_subset = markers.get_subset(
    ['BodyL:AcromionL', 'BodyL:LateralEpicondyleL']
)

markers_subset.data
```

To merge two TimeSeries together, we use the [TimeSeries.merge()](../../api/kineticstoolkit.TimeSeries.merge.rst). For example, if we wanted to add the marker `BodyL:HandL` to this subset:

```{code-cell}
markers_subset = markers_subset.merge(
    markers.get_subset('BodyL:HandL')
)

markers_subset.data
```

Alternatively, we could directly add the data to the TimeSeries `data` attribute using:

```
markers_subset.data['BodyL:HandL'] = markers.data['BodyL:HandL']
```

However, using the `merge` method is slightly safer since it ensures that the time vector is identical in both TimeSeries before merging.

## Splitting TimeSeries

### Using indexes

The [TimeSeries.get_ts_before_index()](../../api/kineticstoolkit.TimeSeries.get_ts_before_index.rst),
[TimeSeries.get_ts_after_index()](../../api/kineticstoolkit.TimeSeries.get_ts_after_index.rst), and
[TimeSeries.get_ts_between_indexes()](../../api/kineticstoolkit.TimeSeries.get_ts_between_indexes.rst)
allow splitting the TimeSeries based on time indexes. For example, if we plot the previous markers subset, we see that the main action (the oscillating signals) starts at about 12 seconds and stops at about 18 seconds. At 120 samples per second, this means from indexes 1440 to 2160.

```{code-cell}
markers_subset.plot()
```

We could therefore split the TimeSeries between indexes 1440 and 2160:

```{code-cell}
ts = markers_subset.get_ts_between_indexes(1440, 2160)
ts.plot()
```

### Using time

We could also use the time directly to do the same split, using one of
[TimeSeries.get_ts_before_time()](../../api/kineticstoolkit.TimeSeries.get_ts_before_time.rst),
[TimeSeries.get_ts_after_time()](../../api/kineticstoolkit.TimeSeries.get_ts_after_time.rst), and
[TimeSeries.get_ts_between_times()](../../api/kineticstoolkit.TimeSeries.get_ts_between_times.rst).

```{code-cell}
ts = markers_subset.get_ts_between_times(12, 18)
ts.plot()
```

### Using events

A very powerful method to split a TimeSeries is to use events. For this example, we will rebuilt the wheelchair kinetics TimeSeries of the previous tutorial.

```{code-cell}
ts = ktk.load(ktk.config.root_folder + '/data/timeseries/smartwheel.ktk.zip')
ts = ts.add_event(4.35, 'sync')
ts = ts.add_event(8.56, 'push')
ts = ts.add_event(9.93, 'recovery')
ts = ts.add_event(10.50, 'push')
ts = ts.add_event(11.12, 'recovery')
ts = ts.add_event(11.78, 'push')
ts = ts.add_event(12.33, 'recovery')
ts = ts.add_event(13.39, 'push')
ts = ts.add_event(13.88, 'recovery')
ts = ts.add_event(14.86, 'push')
ts = ts.add_event(15.30, 'recovery')
ts = ts.sync_event('sync')

ts.plot()
```

If we want to analyze data of the four first pushes and get rid of any other data, we would use one of
[TimeSeries.get_ts_before_event()](../../api/kineticstoolkit.TimeSeries.get_ts_before_event.rst),
[TimeSeries.get_ts_after_event()](../../api/kineticstoolkit.TimeSeries.get_ts_after_event.rst), and
[TimeSeries.get_ts_between_events()](../../api/kineticstoolkit.TimeSeries.get_ts_between_events.rst):

```{code-cell}
# inclusive=True to ensure that the push 0 and push 4 events are included in
# the resulting time vector
first_four_pushes = ts.get_ts_between_events(
    'push', 'push', 0, 4, inclusive=True
)

# Remove events not inside the resulting time vector
first_four_pushes = first_four_pushes.trim_events()

first_four_pushes.plot()
```

We could also, for instance, extract only the push phase of the second cycle:

```{code-cell} ipython3
second_push_phase = ts.get_ts_between_events(
    'push', 'recovery', 1, 1, inclusive=True
)

second_push_phase = second_push_phase.trim_events()

second_push_phase.plot()
```
