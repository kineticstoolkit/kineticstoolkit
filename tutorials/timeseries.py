# %% markdown
"""
TimeSeries
==========
KTK provides a standard class for expressing timeseries. This
data format is very largely inspired by the Matlab's timeseries object. It
provides a way to express several multidimensional data that varies in time,
along with a time vector and a list of events. It also allows subsetting and
merging with other TimeSeries, and extracting sub-TimeSeries based on events.
"""
import ktk
import numpy as np
import matplotlib.pyplot as plt
import os

# %%
"""
Let's start by looking at the attributes of the TimeSeries class.

Attributes
----------
    time : 1-dimension np.array. Default value is [].
        Contains the time vector

    data : dict. Default value is {}.
        Contains the data, where each element contains a np.array which
        first dimension corresponds to time.

    time_info : dict. Default value is {'Unit': 's'}
        Contains metadata relative to time.

    data_info : dict. Default value is {}.
        Contains facultative metadata relative to data. For example, the
        data_info attribute could indicate the unit of data['Forces']:

        >>> data['Forces'] = {'Unit': 'N'}.

        To facilitate the management of data_info, please refer to the
        class method:

        ``ktk.TimeSeries.add_data_info``


To create a new, empty TimeSeries:
"""
ts = ktk.TimeSeries()

ts

# %% exclude
assert isinstance(ts.time, np.ndarray)
assert isinstance(ts.data, dict)
assert isinstance(ts.time_info, dict)
assert isinstance(ts.data_info, dict)
assert isinstance(ts.events, list)
assert ts.time_info['Unit'] == 's'

# %%
"""
This TimeSeries is empty, and is therefore pretty much useless. Let's put some
data in there. We will allocate the time vector and add some random data.
"""
ts.time = np.arange(100)
ts.data['signal1'] = np.sin(ts.time / 10)
ts.data['signal2'] = np.cos(ts.time / 10)
ts.data['signal3'] = np.hstack((np.sin(ts.time / 10)[:, np.newaxis],
                                np.cos(ts.time / 10)[:, np.newaxis]))

ts

# %%
"""
Plotting a TimeSeries
---------------------
There are several ways to plot a TimeSeries, the easiest are either by using
standard matplotlib.pyplot functions, or using the TimeSeries' ``plot``
function.

Using standard matplotlib.pyplot functions:
"""
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(ts.time, ts.data['signal1'])
plt.subplot(3, 1, 2)
plt.plot(ts.time, ts.data['signal2'])
plt.subplot(3, 1, 3)
plt.plot(ts.time, ts.data['signal3'])

# %%
"""
Using the TimeSeries' plot function (note that the axes are automatically
labelled):
"""
plt.figure()
ts.plot()

# %%
"""
We can also select what signals to plot:
"""
plt.figure()
ts.plot(['signal1', 'signal2'])

# %%
"""
Adding time and data information to a TimeSeries
------------------------------------------------
We see in the previous plots that the axes are labelled according to the
name of the data key. We also see that the time unit is in seconds. Unit
information is specified in the dictionaries time_info and data_info. For
example, if we look at the contents of ``ts.time_info``:
"""
ts.time_info

# %%
"""
There is a dict entry named 'Unit' with a value of 's'. We can also add units
to the TimeSeries' data. For example, if ``signal1`` is in newtons,
``signal2`` is in volts and ``signal3`` is in m/s:
"""
ts.add_data_info('signal1', 'Unit', 'N')
ts.add_data_info('signal2', 'Unit', 'V')
ts.add_data_info('signal3', 'Unit', 'm/s')
ts.plot()

# %% exclude
assert ts.data_info['signal1']['Unit'] == 'N'
assert ts.data_info['signal2']['Unit'] == 'V'
assert ts.data_info['signal3']['Unit'] == 'm/s'

# %%
"""
The time_info and data_info fields are not constrained to time/data units. We
can add anything that can be helpful to describe the data.
"""

# %%
"""
Adding events to a TimeSeries
-----------------------------
In addition to add information to time and data, we can also add events to
a TimeSeries. Events are a very helpful concept that help a lot in
synchronizing different TimeSeries, or extracting smaller TimeSeries by slicing
TimeSeries between two events.
"""
ts.add_event(15.34, 'some_event_1')
ts.add_event(99.2, 'some_event_2')
ts.add_event(1, 'some_event_3')
ts.events

# %% exclude
assert ts.events[0].name == 'some_event_1'
assert ts.events[1].name == 'some_event_2'
assert ts.events[2].name == 'some_event_3'
assert ts.events[0].time == 15.34
assert ts.events[1].time == 99.2
assert ts.events[2].time == 1

# %%
"""
As we can see, the ``events`` attribute of the TimeSeries is a list of lists,
where the inner list contains the time at which the event happened followed
by the name of the event. In addition, an event can be accessed using its
properties ``time`` and ``name``.
"""
ts.events[0]

# %%
ts.events[0][0]

# %%
ts.events[0].time

# %%
ts.events[0].name

# %%
"""
When we plot a TimeSeries that contains events using the TimeSeries' ``plot``
method, the events are also drawn. It is also possible to print out the
events' names on the plot:

    """
plt.figure()
ts.plot('signal1')

plt.figure()
ts.plot('signal1', plot_event_names=True)
plt.tight_layout()  # To resize the figure so we see the text completely.

# %%
"""
Using events
------------
The events of a TimeSeries are powerful to extract new TimeSeries after,
before or between events. Among many methods that use events, the most
important are ``get_ts_before_event``, ``get_ts_between_events`` and
``get_ts_after_event``.

"""
ts1 = ts.get_ts_before_event('some_event_1')
plt.figure()
ts1.plot(plot_event_names=True)
plt.tight_layout()  # To resize the figure so we see the text completely.

# %%
ts2 = ts.get_ts_between_events('some_event_3', 'some_event_1')
plt.figure()
ts2.plot(plot_event_names=True)
plt.tight_layout()  # To resize the figure so we see the text completely.

# %%
ts3 = ts.get_ts_after_event('some_event_1')
plt.figure()
ts3.plot(plot_event_names=True)
plt.tight_layout()  # To resize the figure so we see the text completely.

"""
An event can be used to define the time zero of a TimeSeries, which can
help syncing different TimeSeries recorded using different instruments:

"""

plt.figure()
ts.plot(plot_event_names=True)
plt.tight_layout()  # To resize the figure so we see the text completely.

"""
Now we sync on event 'some_event_1', which becomes the new time zero:

"""

ts.sync_on_event('some_event_1')

plt.figure()
ts.plot(plot_event_names=True)
plt.tight_layout()  # To resize the figure so we see the text completely.

# %%
"""
Subsetting and merging TimeSeries
---------------------------------
Subsetting and merging can be done with the ``get_subset`` and ``merge```
methods.

We start by creating a random TimeSeries with three data keys:
"""
ts1 = ktk.TimeSeries(time=np.arange(10))
ts1.data['signal1'] = ts1.time ** 2
ts1.data['signal2'] = np.sin(ts1.time)
ts1.data['signal3'] = np.cos(ts1.time)
ts1.add_data_info('signal1', 'Unit', 'Unit1')
ts1.add_data_info('signal2', 'Unit', 'Unit2')
ts1.add_data_info('signal3', 'Unit', 'Unit3')
ts1.add_event(1.53, 'test_event1')
ts1.add_event(9.2, 'test_event2')
ts1.add_event(1, 'test_event3')

plt.figure()
ts1.plot()

"""
The method ``get_subset`` allows extracting data, data_info and events from
a TimeSeries to get a subset of this TimeSeries. For example, here we
define a new TimeSeries that contains only signal1, and another that
contains both signal2 and signal3.
"""
ts2 = ts1.get_subset('signal1')
ts3 = ts1.get_subset(['signal2', 'signal3'])

plt.figure()
ts2.plot()

plt.figure()
ts3.plot()

"""
The method ``merge`` does the opposite: it merges the data, data_info and
events from two TimeSeries.
"""
ts2.merge(ts3)

plt.figure()
ts2.plot()

"""
Options are available to allow resampling in case both TimeSeries do not share
a same time vector, or to select the conflict resolution in case both
TimeSeries share a same data key. Please see the ``ktk.TimeSeries.merge`` help
for these special cases.
"""

# %% exclude
assert ts1 == ts2

# %%
"""
Resampling a TimeSeries
-----------------------

The method ``resample`` allows a TimeSeries to be resampled over a new time
vector. Any interpolation method supported by ``scipy.interpolate.interp1d``
is supported.

"""
plt.figure()
ts1.plot()

# %%
ts2 = ts1.copy()
ts2.resample(np.arange(0, 9, 0.01), 'linear')
plt.figure()
ts2.plot()

# %%
ts3 = ts1.copy()
ts3.resample(np.arange(0, 9, 0.01), 'nearest')
plt.figure()
ts3.plot()

# %%
ts4 = ts1.copy()
ts4.resample(np.arange(0, 9, 0.01), 'zero')
plt.figure()
ts4.plot()

# %%
ts5 = ts1.copy()
ts5.resample(np.arange(0, 9, 0.01), 'slinear')
plt.figure()
ts5.plot()

# %%
ts6 = ts1.copy()
ts6.resample(np.arange(0, 9, 0.01), 'quadratic')
plt.figure()
ts6.plot()

# %%
ts7 = ts1.copy()
ts7.resample(np.arange(0, 9, 0.01), 'cubic')
plt.figure()
ts7.plot()

# %%
"""
Filling missing data
--------------------

The method ``fill_missing_data`` allows reconstructing TimeSeries with
missing data, for example trajectories with marker occlusion.

"""
ts1 = ktk.TimeSeries(time=np.arange(100.0))

"""
We build a TimeSeries where the first data key will have no missing data:
"""
ts1.data['data1'] = np.sin(ts1.time / 10)

"""
and where the second data keys will have missing data of various sizes:
"""
ts1.data['data2'] = np.cos(ts1.time / 10)
ts1.data['data2'][15:30] = np.nan
ts1.data['data2'][35:40] = np.nan
ts1.data['data2'][95:] = np.nan
ts1.data['data2'][0:5] = np.nan

plt.figure()
ts1.plot()

"""
The ``fill_missing_samples`` allows filling the gaps up to a given length,
using any interpolation method that is supported by
``scipy.interpolate.interp1d``.
"""

ts2 = ts1.copy()
ts2.fill_missing_samples(max_missing_samples=10, method='linear')
plt.figure()
ts2.plot()

# %%
ts3 = ts1.copy()
ts3.fill_missing_samples(max_missing_samples=100, method='linear')
plt.figure()
ts3.plot()

# %%
ts4 = ts1.copy()
ts4.fill_missing_samples(max_missing_samples=10, method='cubic')
plt.figure()
ts4.plot()

# %%
ts5 = ts1.copy()
ts5.fill_missing_samples(max_missing_samples=100, method='cubic')
plt.figure()
ts5.plot()

# %% exclude

# TODO Put some testing code for fill_missing_sample

# %% exclude
def test_empty_constructor():
    ts = ktk.TimeSeries()
    assert isinstance(ts.time, np.ndarray)
    assert isinstance(ts.data, dict)
    assert isinstance(ts.time_info, dict)
    assert isinstance(ts.data_info, dict)
    assert isinstance(ts.events, list)
    assert ts.time_info['Unit'] == 's'
test_empty_constructor()

def test_add_data_info():
    ts = ktk.TimeSeries()
    ts.add_data_info('Force', 'Unit', 'N')
    assert ts.data_info['Force']['Unit'] == 'N'
test_add_data_info()

def test_add_event():
    ts = ktk.TimeSeries()
    ts.add_event(15.34, 'test_event1')
    ts.add_event(99.2, 'test_event2')
    ts.add_event(1, 'test_event3')
    assert ts.events[0].name == 'test_event3'
    assert ts.events[1].name == 'test_event1'
    assert ts.events[2].name == 'test_event2'
    assert ts.events[0].time == 1
    assert ts.events[1].time == 15.34
    assert ts.events[2].time == 99.2
test_add_event()

def test_get_index_before_time():
    ts = ktk.TimeSeries(time=np.array([0, 0.5, 1, 1.5, 2]))
    assert ts.get_index_before_time(0.9) == 1
    assert ts.get_index_before_time(1) == 2
    assert ts.get_index_before_time(1.1) == 2
    assert np.isnan(ts.get_index_before_time(-1))
test_get_index_before_time()

def test_get_index_at_time():
    ts = ktk.TimeSeries(time=np.array([0, 0.5, 1, 1.5, 2]))
    assert ts.get_index_at_time(0.9) == 2
    assert ts.get_index_at_time(1) == 2
    assert ts.get_index_at_time(1.1) == 2
test_get_index_at_time()

def test_get_index_after_time():
    ts = ktk.TimeSeries(time=np.array([0, 0.5, 1, 1.5, 2]))
    assert ts.get_index_after_time(0.9) == 2
    assert ts.get_index_after_time(1) == 2
    assert ts.get_index_after_time(1.1) == 3
    assert np.isnan(ts.get_index_after_time(13))
test_get_index_after_time()

def test_get_event_time():
    ts = ktk.TimeSeries()
    ts.add_event(5.5, 'event1')
    ts.add_event(10.8, 'event2')
    ts.add_event(2.3, 'event2')
    assert ts.get_event_time('event1') == 5.5
    assert ts.get_event_time('event2', 0) == 2.3
    assert ts.get_event_time('event2', 1) == 10.8
test_get_event_time()

def test_get_ts_at_event___get_ts_at_time():
    ts = ktk.TimeSeries()
    ts.time = np.linspace(0, 99, 100)
    time_as_column = np.reshape(ts.time, (-1, 1))
    ts.data['Forces'] = np.block(
            [time_as_column, time_as_column**2, time_as_column**3])
    ts.data['Moments'] = np.block(
            [time_as_column**2, time_as_column**3, time_as_column**4])
    ts.add_event(5.5, 'event1')
    ts.add_event(10.8, 'event2')
    ts.add_event(2.3, 'event2')
    new_ts = ts.get_ts_at_event('event1')
    assert new_ts.time == 5
    new_ts = ts.get_ts_at_event('event2')
    assert new_ts.time == 2
    new_ts = ts.get_ts_at_event('event2', 1)
    assert new_ts.time == 11
test_get_ts_at_event___get_ts_at_time()

def tes_get_ts_before_time():
    ts = ktk.TimeSeries(time=np.linspace(0, 9, 10))
    new_ts = ts.get_ts_before_time(3)
    assert new_ts.time.tolist() == [0., 1., 2., 3.]
    new_ts = ts.get_ts_before_time(3.5)
    assert new_ts.time.tolist() == [0., 1., 2., 3.]
    new_ts = ts.get_ts_before_time(-2)
    assert new_ts.time.tolist() == []
    new_ts = ts.get_ts_before_time(13)
    assert new_ts.time.tolist() == [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]
tes_get_ts_before_time()

def test_get_ts_after_time():
    ts = ktk.TimeSeries(time=np.linspace(0, 9, 10))
    new_ts = ts.get_ts_after_time(3)
    assert new_ts.time.tolist() == [3., 4., 5., 6., 7., 8., 9.]
    new_ts = ts.get_ts_after_time(3.5)
    assert new_ts.time.tolist() == [4., 5., 6., 7., 8., 9.]
    new_ts = ts.get_ts_after_time(-2)
    assert new_ts.time.tolist() == [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]
    new_ts = ts.get_ts_after_time(13)
    assert new_ts.time.tolist() == []
test_get_ts_after_time()

def test_get_ts_between_times():
    ts = ktk.TimeSeries(time=np.linspace(0, 9, 10))
    new_ts = ts.get_ts_between_times(3, 6)
    assert new_ts.time.tolist() == [3., 4., 5., 6.]
    new_ts = ts.get_ts_between_times(3.5, 5.5)
    assert new_ts.time.tolist() == [4., 5.]
    new_ts = ts.get_ts_between_times(-2, 13)
    assert new_ts.time.tolist() == [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]
    new_ts = ts.get_ts_between_times(-2, -1)
    assert new_ts.time.tolist() == []
test_get_ts_between_times()

def test_merge_and_resample():
    # Begin with two timeseries with identical times
    ts1 = ktk.TimeSeries()
    ts1.time = np.linspace(0, 99, 100)
    ts1.data['signal1'] = np.random.rand(100, 2)
    ts1.data['signal2'] = np.random.rand(100, 2)
    ts1.data['signal3'] = np.random.rand(100, 2)
    ts1.add_data_info('signal1', 'Unit', 'Unit1')
    ts1.add_data_info('signal2', 'Unit', 'Unit2')
    ts1.add_data_info('signal3', 'Unit', 'Unit3')
    ts1.add_event(1.54, 'test_event1')
    ts1.add_event(10.2, 'test_event2')
    ts1.add_event(100, 'test_event3')

    ts2 = ktk.TimeSeries()
    ts2.time = np.linspace(0, 99, 100)
    ts2.data['signal4'] = np.random.rand(100, 2)
    ts2.data['signal5'] = np.random.rand(100, 2)
    ts2.data['signal6'] = np.random.rand(100, 2)
    ts2.add_data_info('signal4', 'Unit', 'Unit1')
    ts2.add_data_info('signal5', 'Unit', 'Unit2')
    ts2.add_data_info('signal6', 'Unit', 'Unit3')
    ts2.add_event(1.54, 'test_event4')
    ts2.add_event(10.2, 'test_event5')
    ts2.add_event(100, 'test_event6')

    ts1.merge(ts2)

    assert np.all(ts1.data['signal4'] == ts2.data['signal4'])
    assert np.all(ts1.data['signal5'] == ts2.data['signal5'])
    assert np.all(ts1.data['signal6'] == ts2.data['signal6'])
    assert np.all(ts1.data_info['signal4']['Unit'] ==
                  ts2.data_info['signal4']['Unit'])
    assert np.all(ts1.data_info['signal5']['Unit'] ==
                  ts2.data_info['signal5']['Unit'])
    assert np.all(ts1.data_info['signal6']['Unit'] ==
                  ts2.data_info['signal6']['Unit'])

    # Try with two timeseries that don't fit in time. It must generate an
    # exception.
    ts1 = ktk.TimeSeries()
    ts1.time = np.linspace(0, 99, 100, endpoint=False)
    ts1.data['signal1'] = np.random.rand(100, 2)
    ts1.data['signal2'] = np.random.rand(100, 2)
    ts1.data['signal3'] = np.random.rand(100, 2)
    ts1.add_data_info('signal1', 'Unit', 'Unit1')
    ts1.add_data_info('signal2', 'Unit', 'Unit2')
    ts1.add_data_info('signal3', 'Unit', 'Unit3')
    ts1.add_event(1.54, 'test_event1')
    ts1.add_event(10.2, 'test_event2')
    ts1.add_event(100, 'test_event3')

    ts2 = ktk.TimeSeries()
    ts2.time = np.linspace(0, 99, 300, endpoint=False)
    ts2.data['signal4'] = ts2.time ** 2
    ts2.data['signal5'] = np.random.rand(300, 2)
    ts2.data['signal6'] = np.random.rand(300, 2)
    ts2.add_data_info('signal4', 'Unit', 'Unit1')
    ts2.add_data_info('signal5', 'Unit', 'Unit2')
    ts2.add_data_info('signal6', 'Unit', 'Unit3')
    ts2.add_event(1.54, 'test_event4')
    ts2.add_event(10.2, 'test_event5')
    ts2.add_event(100, 'test_event6')

    try:
        ts1.merge(ts2)
        raise Exception('This command should have raised a ValueError.')
    except ValueError:
        pass

    # Try the same thing but with linear resampling
    ts1.merge(ts2, resample=True)

    def _assert_almost_equal(one, two):
        assert np.max(np.abs(one - two)) < 1E-6

    _assert_almost_equal(ts1.data['signal4'], ts2.data['signal4'][0::3])
    _assert_almost_equal(ts1.data['signal5'], ts2.data['signal5'][0::3])
    _assert_almost_equal(ts1.data['signal6'], ts2.data['signal6'][0::3])
    assert ts1.data_info['signal4']['Unit'] == ts2.data_info['signal4']['Unit']
    assert ts1.data_info['signal5']['Unit'] == ts2.data_info['signal5']['Unit']
    assert ts1.data_info['signal6']['Unit'] == ts2.data_info['signal6']['Unit']
test_merge_and_resample()

def test_rename_data():
    ts = ktk.TimeSeries(time=np.arange(100))
    ts.data['data1'] = ts.time.copy()
    ts.data['data2'] = ts.time.copy()
    ts.add_data_info('data2', 'Unit', 'N')

    ts.rename_data('data2', 'data3')

    assert 'data2' not in ts.data
    assert 'data2' not in ts.data_info
    assert np.all(ts.data['data3'] == ts.time)
    assert ts.data_info['data3']['Unit'] == 'N'
test_rename_data()

