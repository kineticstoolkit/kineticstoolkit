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

# Smoothing signals using a moving average

The moving average is an excellent filter to remove noise that is related to a specific time pattern. The classic example is the day-to-day evaluation of a process that is sensible to week-ends (for example, the number of workers who enter a building). A moving average with a window length of 7 days is ideal to evaluate the generic trend of this signal without considering intra-week fluctuations. Although its use in biomechanics is less obvious, this filter may be useful in some situation. This tutorial will show how to use the [ktk.filters.smooth()](/api/kineticstoolkit.filters.smooth.rst) function on TimeSeries data.

```{code-cell}
import kineticstoolkit.lab as ktk
import matplotlib.pyplot as plt
```

We will first load some noisy data:

```{code-cell}
ts = ktk.load(
    ktk.config.root_folder + '/data/filters/sample_noises.ktk.zip')

# Plot it
ts.plot(['clean', 'periodic_noise'], marker='.')
plt.grid(True)
plt.tight_layout()
```

In this signal, we observe that appart from random noise, there seems to be a periodic signal with a period of five seconds, that we may consider as noise. Since we consider these variations as noise and their period is constants, the moving average is a nice candidate for filtering out this noise.

```{code-cell}
filtered = ktk.filters.smooth(ts, window_length=5)

ts.plot(['clean', 'periodic_noise'], marker='.')

filtered.plot('periodic_noise', marker='.', color='k')

plt.title('Removing the fast, constant rate variation (black curve)')
plt.grid(True)
plt.tight_layout()
```

As expected, the 5-sample period noise was completely removed. Some signal was however averaged and we therefore lost some dynamics in the signal.
