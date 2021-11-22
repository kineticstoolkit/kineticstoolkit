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

# Saving and loading

Contrarily to Matlab with its `.mat` file, python does not come with a single standard way to save data. To ease saving and sharing data, Kinetics Toolkit provides two functions that load and save its own `.ktk.zip` format:

- [save()](/api/kineticstoolkit.save.rst)
- [load()](/api/kineticstoolkit.load.rst)

These functions are straightforward to use:

```{code-cell}
import kineticstoolkit.lab as ktk
import numpy as np

variable = {
    'some_array': np.arange(0, 5, 0.5),
    'some_text': 'hello',
}
```

Saving the variable:

```{code-cell}
ktk.save('filename.ktk.zip', variable)
```

Loading back the variable:

```{code-cell}
loaded_variable = ktk.load('filename.ktk.zip')

loaded_variable
```

## File format and supported types

The `ktk.zip` file format is built to be as portable and simple as possible. It is a standard zip file that contains two JSON files:

- `metadata.json`: The file metadata such as the save date, the computer's operating system, etc.
- `data.json`: The data. The data types that are not supported natively by the JSON file format (e.g., numpy, pandas and ktk objects) are converted to supported objects so that they are fully readable in other environments such as Matlab.

The `ktk.zip` file format supports any combination of the following types:

| Already supported by JSON          | Extended for Kinetics Toolkit |
| ---------------------------------- | ----------------------------- |
| dict containing any supported type | numpy.array                   |
| list containing any supported type | pandas.DataFrame              |
| str                                | pandas.Series                 |
| int                                | kineticstoolkit.TimeSeries    |
| float                              |                               |
| True, False, None                  |                               |

Tuples can also be saved but will be loaded back as lists.

## Loading a ktk.zip file in Matlab

    % Create a temporary folder to unzip to
    mkdir('temp');
    unzip(filename, 'temp');

    % Load the file contents
    data = jsondecode(fileread('temp/data.json'));
    metadata = jsondecode(fileread('temp/metadata.json'));

Since the types are not the same between python, JSON and Matlab, here is how Matlab will reconstruct the following python types:

| Python                         | Matlab                    |
| ------------------------------ | ------------------------- |
| None                           | NaN                       |
| True, False                    | true, false               |
| int, float                     | double                    |
| str                            | char                      |
| dict                           | struct                    |
| list of different data types   | cell array                |
| list of boolean                | array of logical          |
| list of int or float           | array of double           |
| list of str                    | cell array of char        |
| list of dict, same field names | structure array           |
| list of dict, diff field names | cell array of struct      |
| numpy.array                    | array of double           |
| kineticstoolkit.TimeSeries     | struct                    |
| complex                        | struct with real and imag |

*tuples are saved as lists in JSON.