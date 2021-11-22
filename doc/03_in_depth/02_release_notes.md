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

# Release Notes

## Version 0.7 (Upcoming)

### New Features

### Improvements

### Breaking Changes

- New default for `TimeSeries.merge`'s `overwrite` option. It changed from True to False.

## Version 0.6 (August 2021)

### New Features

- Rigid body tracking and reconstruction of virtual markers in module `kinematics`: functions `define_rigid_body()`, `define_virtual_marker()`, and `track_rigid_body()`.
- Interactive method to edit events in a TimeSeries: method `TimeSeries.ui_edit_events()`.

### Improvements

- All TimeSeries methods now work on copies, see breaking changes below.

### Breaking Changes

- **WARNING - Important breaking change** - Every `TimeSeries` method now works on a copy of the TimeSeries. The original TimeSeries is never modified itself. It was already the case for methods such as `get_ts_between_events()` etc. which would not affect the original TimeSeries, but not for others such as `add_event()`, `rename_data()`, etc. This modification was made in an effort to standardize the TimeSeries behaviour and to allow safe method chaining. This choice is a bit drastic and was implemented quickly because we are still early in Kinetics Toolkit's development. It has been encouraged by Pandas, which seems to also go in a similar direction (https://github.com/pandas-dev/pandas/issues/16529). To migrate your code to the new behaviour, please change your method calls accordingly by adding an assignation before the call. For example: `ts.add_event(...)` becomes `ts = ts.add_event(...)`. The full list of methods that have been changed is:
    - `add_data_info()`
    - `add_event()`
    - `fill_missing_samples()`
    - `merge()`
    - `remove_data()`
    - `remove_data_info()`
    - `remove_event()`
    - `rename_data()`
    - `rename_event()`
    - `shift()`
    - `sort_events()`
    - `sync_event()`
    - `trim_events()`
    - `ui_add_event()` (now discontinued in favour of `ui_edit_events`)

**Please do not update to 0.6 until you are ready to find and modify those statements in your current code**.

## Version 0.5 (June 2021)

### New Features

- Coordinate system origins are now clickable in Player.

### Improvements

- Most TimeSeries arguments can now be used either by position or keyword (removed superflous slash operators in signatures).
- TimeSeriesEvent class is now a proper data class. This has no implication on usability, but the API is cleaner and more robust.
- Bug fixes.

### Breaking Changes

- `cycle.detect_cycles` was changed back to experimental. Its argument pairs xxxx1, xxxx2 have been changed to sequences [xxxx1, xxxx2] in prevision of possible cycle detections with more than 2 phases, or even only one phase. This method is now experimental because it may be separated into different functions (one to detect cycles, another to search cycles with given criteria, and another to remove found cycles).


## Version 0.4 (April 2021)

### New Features

- Added the `geometry` module to perform rigid body geometry operations such as creating frames, homogeneous transforms, multiplying series of matrices, converting from local to global coordinates and vice-versa, and extracting Euler angles from homogeneous transforms.
- Added the `span` option to `cycles.time_normalize`, so that cycles can be normalized between other ranges than 0% to 100%. Both reducing span (e.g., 10% to 90%) and increasing span (e.g., -25 to 125%) work.
- Added the `to_html5` method to `Player`, which allows visualizing 3d markers and bodies in Jupyter.
- Added the `rename_event` method to `TimeSeries`.

### Improvements

- Added test coverage measurement for continuous improvement of code robustness.
- Added warnings when using private or unstable functions.
- Changed the website to use the ReadTheDoc theme, and changed its structure to facilitate continuous improvements of the website without needing to wait for releases.

### Breaking Changes

- The default behaviour for `TimeSeries.remove_event` changed when no occurrence is defined. Previously, only the first occurrence was removed. Now every occurrence is removed.
- In `cycles.time_normalize`, the way to time-normalize between two events of the same name changed from `event_name, _` to `event_name, event_name`.

## Version 0.3 (October 2020)

### New Features

- Added the `cycles` module to detect, time-normalize and stack cycles (e.g., gait cycles, wheelchair propulsion cycles, etc.)
- Added the `pushrimkinetics` module to read files from instrumented wheelchair wheels, reconstruct the pushrim kinetics,
  remove dynamic offsets in kinetic signals, and perform speed and power calculation for analysing spatiotemporal and kinetic
  parameters of wheelchair propulsion.
- Added lab mode to allow importing ktk without changing defaults.
- Added `ktk.filters.deriv()` to derivate TimeSeries.
- Added `ktk.filters.median()`, which is a running median filter function.

### Improvements

- `TimeSeries.plot()` now shows the event occurrences besides the event names.
- Nicer tutorial for the `filters` module.
- Improved unit tests for the `filters` module.

### Breaking Changes

- The module name has been changed from `ktk` to `kineticstoolkit`. Importing using `import ktk` is now deprecated and the standard way to import is now either `import kineticstoolkit as ktk` or `import kineticstoolkit.lab as ktk`.
- Now importing Kinetics Toolkit does not change IPython's representation of dicts or matplotlib's defaults. This allows using ktk's functions without modifying the current working environment. The old behaviour is now the lab mode and is the recommended way to import Kinetics Toolkit in an IPython-based environment: `import kineticstoolkit.lab as ktk`.


## Version 0.2 (August 2020)

### New Features

- Added the functions `ktk.load` and `ktk.save`.
- Introduced the `ktk.zip` file format.
- Added the `gui` module to show messages, input dialogs and file/folder pickers.
- Added the `filters` module with TimeSeries wrappers for scipy's butterworth and savitsky-golay filters.
- Added interactive methods to TimeSeries: `TimeSeries.ui_add_event()`, `TimeSeries.ui_get_ts_between_clicks()` and `TimeSeries.ui_sync()` (experimental).
- Added `TimeSeries.remove_event()` method.
- Added `TimeSeries.resample()` (experimental).

### Improvements

- Updated the documentation system using sphinx and jupyter-lab.
- Improved performance of `TimeSeries.from_dataframe()`
- ktk is now typed.

### Breaking Changes
- `TimeSeries.from_dataframe()` is now a class function and not an instance method anymore. Therefore we need to call `ktk.TimeSeries.from_dataframe(dataframe)` instead of `ktk.TimeSeries().from_dataframe(dataframe)`.
- Now depends on python 3.8 instead of 3.7.


## Version 0.1 (May 2020)

### New Features

- Added the `TimeSeries` class.
- Added the `kinematics` module, to read c3d and n3d files.
- Added the `Player` class, to view markers in 3d.
