# This is the code to generate both samples.

import kineticstoolkit.lab as ktk
import numpy as np
import scipy as sp

ts_source = ktk.TimeSeries(time=np.linspace(0, 5, 1000))

#%% Generate some fancy functions
ts_source.data['square'] = np.array([int(t) % 2 == 0 for t in ts_source.time]) + 4.5
ts_source.data['triangle'] = np.block([
    0.01 * (sp.integrate.cumtrapz(ts_source.data['square'] - 5)) + 3, 4.])
ts_source.data['sine'] = 0.5 * np.sin(ts_source.time * np.pi) + 2
ts_source.data['pulse'] = np.array([float((i + 50) % 50 == 0) for i in range(1000)])

ts_source.plot()

ktk.save('sample_clean.ktk.zip', ts_source)


#%% Add white noise
np.random.seed(0)
noise = (np.random.rand(1000) - 0.5) * 1E-1
ts = ts_source.copy()
ts.data['square'] += noise
ts.data['triangle'] += noise
ts.data['sine'] += noise
ts.data['pulse'] += noise

# Quantification noise: resolution of 1E-2
for key in ts.data:
    ts.data[key] = np.floor(ts.data[key] * 1E2) * 1E-2

ktk.save('sample_noisy.ktk.zip', ts)


#%% Make a nice signal for moving average
ts_ma = ktk.TimeSeries(time=np.arange(50))
ts_ma.data['signal'] = np.round(
    (np.sin(ts_ma.time / 7) +
     ts_ma.time * np.sin(ts_ma.time / 2) / 30 +
     np.mod(ts_ma.time, 5) / 10 +
     noise[0:50]),
    1)
ts_ma.plot()

ktk.save('sample_moving_average.ktk.zip', ts_ma)

