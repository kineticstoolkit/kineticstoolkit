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

# Converting between TimeSeries and Pandas DataFrames

To ensure a great compatibility between Kinetics Toolkit and other frameworks, TimeSeries can be converted from and to Pandas DataFrame using the [TimeSeries.from_dataframe()](../api/kineticstoolkit.TimeSeries.from_dataframe.rst) and [TimeSeries.to_dataframe()](../api/kineticstoolkit.TimeSeries.to_dataframe.rst) methods, and thus benefit from the myriad of options offered by Pandas.

In this tutorial, we will learn how to import comma-separated-value (csv) files as TimeSeries, and export back to csv.

```{code-cell}
import kineticstoolkit.lab as ktk
import pandas as pd
```

## Example 1: Importing a csv file

We import the contents of this csv file:

```
    Time,Force,Position
    0,0,0
    0.1,0,0.01
    0.2,0,0.02
    0.3,0,0.03
    0.4,0.3,0.04
    0.5,0.5,0.05
    0.6,0.6,0.06
    0.7,0.5,0.07
    0.8,0.3,0.08
    0.9,0,0.09
    1,0,0.1
```

First, by opening it as a Pandas DataFrame using `pd.read_csv()`, then by converting it to a TimeSeries:

```{code-cell}
ts = ktk.TimeSeries.from_dataframe(
    pd.read_csv(
        ktk.config.root_folder + '/data/timeseries/sample1.csv',
        index_col='Time',
    )
)

ts
```

```{code-cell}
ts.data
```

### Multidimensional data

TimeSeries are well suited for multidimensional data. In the last example, the force sensor was unidimensional. For a tridimensional force sensor, we would expect three signals (x, y, z).

In this second example, we will import the following csv file:

```
    Time,Fx,Fy,Fz,Position
    0,0,-9.81,0,0
    0.1,0,-9.81,0,0.01
    0.2,0,-9.81,0,0.02
    0.3,0,-9.81,0,0.03
    0.4,0.3,-9.81,1.5,0.04
    0.5,0.5,-9.81,2.5,0.05
    0.6,0.6,-9.81,3,0.06
    0.7,0.5,-9.81,2.5,0.07
    0.8,0.3,-9.81,1.5,0.08
    0.9,0,-9.81,0,0.09
    1,0,-9.81,0,0.1
```

```{code-cell}
ts = ktk.TimeSeries.from_dataframe(
    pd.read_csv(
        ktk.config.root_folder + '/data/timeseries/sample2.csv',
        index_col='Time',
    )
)

ts
```

```{code-cell}
ts.data
```

As for the previous example, the csv file was correctly read as a TimeSeries. However, force components are scattered into three separate signals. Instead, we would like to process these signals as three components of a same data.

A trick to combine the three force components into a single signal is to rename the columns of the DataFrame, either in the original csv file or after reading it, using index brackets. Let's start over by loading the DataFrame first:

```{code-cell}
df = pd.read_csv(
    ktk.config.root_folder + '/data/timeseries/sample2.csv',
    index_col='Time',
)

df
```

Now, we rename the columns using indexing:

```{code-cell}
df.columns = ['Forces[0]', 'Forces[1]', 'Forces[2]', 'Position']

df
```

Finally, we can import this new DataFrame as a TimeSeries. The forces signals are combined into one Nx3 array:

```{code-cell}
ts = ktk.TimeSeries.from_dataframe(df)

ts.data
```

```{code-cell}
ts.data['Forces']
```

For series of arrays with more than one dimension, the brackets would have multiple indexes. For example, a series of Nx4x4 homogeneous matrices would require 16 columns and the indexes would go from [0,0] to [3,3].

## Example 2: Converting a c3d file to csv

For saving a TimeSeries to a `csv`, we create a DataFrame using the [TimeSeries.to_dataframe()](../api/kineticstoolkit.TimeSeries.to_dataframe.rst) method, then we can use Pandas' `to_csv()` method.

In this example, we will read 3d marker positions from a sample `c3d` file, and export these positions to a `csv` file. We first read the `c3d` file using the [kinematics](../api/kineticstoolkit.kinematics.rst) module. This results in a TimeSeries with 26 markers:

```{code-cell}
markers = ktk.kinematics.read_c3d_file(
    ktk.config.root_folder
    + '/data/kinematics/sprintbasket.c3d'
)

markers
```

```{code-cell}
markers.data
```

To convert this TimeSeries to a `csv`, we first create a DataFrame:

```{code-cell}
df = markers.to_dataframe()

df
```

Then we export this DataFrame to a `csv` file. Let's print the first 3 lines of this file:

```{code-cell}
df.to_csv('output.csv', index_label='Time')

!head -3 output.csv
```
