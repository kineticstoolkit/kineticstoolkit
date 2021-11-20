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

# Developer manual

This is a draft for an upcoming coding style for Kinetics Toolkit and other projects at the Research Lab on Mobility in Adaptive Sports.

## Coding style

### Standard Python conventions

We try, when possible, to match the guidelines presented in these documents:

- [Style Guide for Python Code (PEP8)](https://pep8.org);
- [Numpy Docstring](https://numpydoc.readthedocs.io/en/latest/format.html).

Those are precious references and all other sections are additions to these references. Integrated desktop environments may help programmers to follow these conventions. For example, in Spyder, one could enable:

- In `Preferences : Completion in linting : Code style and formatting : Code style`, check `Enable code style linting` to enable PEP8 linting;
- In `Preferences : Completion in linting : Code style and formatting : Code formatting`, Select `autopep8` and check `Autoformat files on save` to ensure minimal PEP8 compliance at all times;
- In `Preferences : Completion in linting : Docstring style`, check `Enable Docstring style linting` and select `Numpy`  to enable Numpy docstring linting.

### Quote style

Strings are single-quoted or double-quoted following their meaning:

- Most strings have single quotes:
    - `kinetics['Forces'] = [0, 0, 0, 0]`
- Strings that contain text in the form of messages have double quotes:
    - `warnings.warn("This sample contains missing data")`
    - `dictionary['key'] = "Please select an option."`

### Naming conventions

The following PEP8 conventions are used:

- All code, comments and documentation are in **English**.
- **Function names** are active (begin by a verb), and are in snake_case (lowercase words separated by underscores):
    - close
    - calculate_power
    - detect_cycles
- **Variable names** are passive, and are also in snake_case:
    - forces
    - detected_markers
- **Class names** are passive, and are in PascalCase (Capital first letters):
    - TimeSeries
    - Player
- **Constants** are passive, and are in UPPER_SNAKE (uppercase words separated by underscores):
    - CALIBRATION_MATRIX
    - WHEEL_RADIUS

In addition, the following convention is used:

- **Strings contents** for data flow and coding (e.g., keys, signal names) are passive form, and are in MixedCase:
    - contents['Forces'], kinematics.data['UpperArmR']
    - dataframe.columns = ['WithBall', 'WithoutBall']

### Type hints

Kinetics Toolkit is type-hinted, with static type checking performed by `mypy`. It does not use python 3.9 contained types yet, and therefore relies on the standard `typing` library.

```{code-cell}
import pandas as pd
import numpy as np
from typing import Dict

def dataframe_to_dict_of_arrays(
        dataframe: pd.DataFrame) -> Dict[str, np.ndarray]:
    pass
```

### Function life from development to deprecation

New public functions appear and live in the following order:

#### Unstable

These functions are currently being developed. They are considered public in the development API, and private in the stable API. They are decorated by the `@unstable` decorator, which automates the documentation of their unstable status both in their doctring and in the API documentation. This decorator automates the inclusion of these functions in the development API, and their exclusion from the stable API.

```{code-cell}
from kineticstoolkit.decorators import unstable


@unstable
def function_name(arguments: str) -> None:
    """Perform an operation."""
    pass  # Function contents
```

#### Experimental

These objects are part of the API but are not considered stable yet. They are documented accordingly in their docstring, with the following text just before the parameters section (by replacing the version number):

```
    Warning
    -------
    This function, which has been introduced in 0.4, is still experimental and
    may change signature or behaviour in the future.
```

They don't have a dedicated decorator.

#### Stable

Standard production function, without specific decorator or warning.

#### Deprecated

Standard function, but decorated with the `@deprecated` decorator:

```{code-cell}
from kineticstoolkit.decorators import deprecated


@deprecated(since='0.1', until='0.2',
            details='It has been replaced by `better_function` because '
                    'the latter is much better.')
def function_name(arguments: str) --> None:
    """Perform an operation."""
    pass  # Function contents
```

## Feature development to release cycle

### Feature branches

Features are developed on `feature/feature_name` branches. All of the above should be done before going further:

- Feature completed
- Docstring completed
- Doctest completed and passing (if relevant)
- Unit test completed and passing

Still on the feature branch, `ktk.dev.release()` should be run with success. This function runs `autopep8`, `mypy`, every unit test, builds all the tutorials, build the API and generates the static website.

On success, the API should be navigated on the generated website to see if it rendered well.

When all is done, the feature branch can be merged onto `master`.

### Master branch

This is the main development branch, which hosts the most recent code and documentation. The development website's documentation automatically pulls from the master branch regularly.

Everything on `master` should work, but new features (marked with the @unstable decorator) are not final.

### Stable branch

This is the release branch that corresponds to the packages distributed on PyPI and conda-forge. This branch also hosts the main website's documentation. This site automatically pulls from the stable branch regularly.

To make a release, one must:
    - Merge the master branch onto the stable branch;
    - Remove the @unstable decorators from the to-be-released function declarations;
    - Bump the version number in `kineticstoolkit/VERSION`
    - Run `ktk.dev.release()` and check the generated website, as for a normal commit on master;
    - Run `ktk.dev.compile_for_pypi()`;
    - Run `ktk.dev.upload_to_pypi()`.
    - Commit and tag the release on the stable branch.
    - Wait for conda-forge to detect the change on PyPI, and automatically create and merge a pull request on conda-forge.