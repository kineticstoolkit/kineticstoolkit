# %% markdown
"""
ktk.TimeSeries Tutorial
=======================
KTK provides a standard class for expressing timeseries: ktk.TimeSeries. This
data format is very largely inspired by the Matlab's timeseries object. It
provides a way to express several multidimensional data that varies in time,
along with a time vector and a list of events. It also allows subsetting and
merging with other TimeSeries, and extracting sub-TimeSeries based on events.
"""

# %%

import ktk
import numpy as np
import matplotlib.pyplot as plt
import os

# %% markdown
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
# %%
ts = ktk.TimeSeries()

ts

# %% exclude
assert isinstance(ts.time, np.ndarray)
assert isinstance(ts.data, dict)
assert isinstance(ts.time_info, dict)
assert isinstance(ts.data_info, dict)
assert isinstance(ts.events, list)
assert ts.time_info['Unit'] == 's'

# %% markdown
"""
This TimeSeries is empty, and is therefore pretty much useless. Let's put some
data in there. We will allocate the time vector and add some random data.
"""

# %%
ts.time = np.arange(100)
ts.data['signal1'] = np.random.rand(100, 2)
ts.data['signal2'] = np.random.rand(100, 2)
ts.data['signal3'] = np.random.rand(100, 2)

ts

# %% markdown
"""
Plotting a TimeSeries
---------------------
There are several ways to plot a TimeSeries, the easiest are either by using
standard matplotlib.pyplot functions, or using the TimeSeries' ``plot``
function.

Using standard matplotlib.pyplot functions:
"""

# %%
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(ts.time, ts.data['signal1'])
plt.subplot(3, 1, 2)
plt.plot(ts.time, ts.data['signal2'])
plt.subplot(3, 1, 3)
plt.plot(ts.time, ts.data['signal3'])

# %% markdown
"""
Using the TimeSeries' plot function (note that the axes are automatically
labelled):
"""
# %%
plt.figure()
ts.plot()

# %% markdown
"""
We can also select what signals to plot:
"""
# %%
plt.figure()
ts.plot(['signal1', 'signal2'])

# %% markdown
"""
Adding time and data information to a TimeSeries
------------------------------------------------
We see in the previous plots that the axes are labelled according to the
name of the data key. We also see that the time unit is in seconds. Unit
information is specified in the dictionaries time_info and data_info. For
example, if we look at the contents of ``ts.time_info``:
"""

# %%
ts.time_info

# %% markdown
"""
There is a dict entry named 'Unit' with a value of 's'. We can also add units
to the TimeSeries' data. For example, if ``signal1`` is in newtons,
``signal2`` is in volts and ``signal3`` is in m/s:
"""

# %%
ts.add_data_info('signal1', 'Unit', 'N')
ts.add_data_info('signal2', 'Unit', 'V')
ts.add_data_info('signal3', 'Unit', 'm/s')
ts.plot()

# %% exclude
assert ts.data_info['signal1']['Unit'] == 'N'
assert ts.data_info['signal2']['Unit'] == 'V'
assert ts.data_info['signal3']['Unit'] == 'm/s'

# %% markdown
"""
The time_info and data_info fields are not constrained to time/data units. We
can add anything that can be helpful to describe the data.

Adding events to a TimeSeries
-----------------------------
In addition to add information to time and data, we can also add events to
a TimeSeries. Events are a very helpful concept that help a lot in
synchronizing different TimeSeries, or extracting smaller TimeSeries by slicing
TimeSeries between two events.
"""

# %%
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

# %% markdown
"""
As we can see, the ``events`` attribute of the TimeSeries is a list of lists,
where the inner list contains the time at which the event happened followed
by the name of the event. In addition, an event can be accessed using its
properties ``time`` and ``name``.
"""

# %%
ts.events[0]

# %%
ts.events[0][0]

# %%
ts.events[0].time

# %%
ts.events[0].name

# %% markdown
"""
When we plot a TimeSeries that contains events using the TimeSeries' ``plot``
function, the events are also drawn. It is also possible to print out the
events' names on the plot:
"""

# %%
plt.figure()
ts.plot('signal1')

plt.figure()
ts.plot('signal1', plot_event_names=True)
plt.tight_layout()  # To resize the figure so we see the text completely.

# %%

def test_get_index_before_time():
    ts = ktk.TimeSeries(time=np.array([0, 0.5, 1, 1.5, 2]))
    assert ts.get_index_before_time(0.9) == 1
    assert ts.get_index_before_time(1) == 2
    assert ts.get_index_before_time(1.1) == 2
    assert np.isnan(ts.get_index_before_time(-1))


def test_get_index_at_time():
    ts = ktk.TimeSeries(time=np.array([0, 0.5, 1, 1.5, 2]))
    assert ts.get_index_at_time(0.9) == 2
    assert ts.get_index_at_time(1) == 2
    assert ts.get_index_at_time(1.1) == 2


def test_get_index_after_time():
    ts = ktk.TimeSeries(time=np.array([0, 0.5, 1, 1.5, 2]))
    assert ts.get_index_after_time(0.9) == 2
    assert ts.get_index_after_time(1) == 2
    assert ts.get_index_after_time(1.1) == 3
    assert np.isnan(ts.get_index_after_time(13))


def test_get_event_time():
    ts = ktk.TimeSeries()
    ts.add_event(5.5, 'event1')
    ts.add_event(10.8, 'event2')
    ts.add_event(2.3, 'event2')
    assert ts.get_event_time('event1') == 5.5
    assert ts.get_event_time('event2', 0) == 2.3
    assert ts.get_event_time('event2', 1) == 10.8


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
    ts1.add_event(15.34, 'test_event1')
    ts1.add_event(99.2, 'test_event2')
    ts1.add_event(1, 'test_event3')

    ts2 = ktk.TimeSeries()
    ts2.time = np.linspace(0, 99, 100)
    ts2.data['signal4'] = np.random.rand(100, 2)
    ts2.data['signal5'] = np.random.rand(100, 2)
    ts2.data['signal6'] = np.random.rand(100, 2)
    ts2.add_data_info('signal4', 'Unit', 'Unit1')
    ts2.add_data_info('signal5', 'Unit', 'Unit2')
    ts2.add_data_info('signal6', 'Unit', 'Unit3')
    ts2.add_event(15.34, 'test_event4')
    ts2.add_event(99.2, 'test_event5')
    ts2.add_event(1, 'test_event6')

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
    assert ts1.events[3] == ts2.events[0]
    assert ts1.events[4] == ts2.events[1]
    assert ts1.events[5] == ts2.events[2]

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
    ts1.add_event(15.34, 'test_event1')
    ts1.add_event(99.2, 'test_event2')
    ts1.add_event(1, 'test_event3')

    ts2 = ktk.TimeSeries()
    ts2.time = np.linspace(0, 99, 300, endpoint=False)
    ts2.data['signal4'] = ts2.time ** 2
    ts2.data['signal5'] = np.random.rand(300, 2)
    ts2.data['signal6'] = np.random.rand(300, 2)
    ts2.add_data_info('signal4', 'Unit', 'Unit1')
    ts2.add_data_info('signal5', 'Unit', 'Unit2')
    ts2.add_data_info('signal6', 'Unit', 'Unit3')
    ts2.add_event(15.34, 'test_event4')
    ts2.add_event(99.2, 'test_event5')
    ts2.add_event(1, 'test_event6')

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
    assert ts1.events[3] == ts2.events[0]
    assert ts1.events[4] == ts2.events[1]
    assert ts1.events[5] == ts2.events[2]
