# Kinetics Toolkit

*An Open-Source Python Package to Facilitate Research in Biomechanics*

[![JOSS](https://joss.theoj.org/papers/10.21105/joss.03714/status.svg)](https://doi.org/10.21105/joss.03714)
[![Anaconda](https://anaconda.org/conda-forge/kineticstoolkit/badges/version.svg)](https://anaconda.org/conda-forge/kineticstoolkit)
[![Latest release](https://anaconda.org/conda-forge/kineticstoolkit/badges/latest_release_date.svg)](https://anaconda.org/conda-forge/kineticstoolkit)

---

:::{ifconfig} release in ['master']
```{warning}
You are currently on Kinetics Toolkit's development website. This website may refer to unreleased and unstable material.
```
:::

[Kinetics Toolkit (ktk)](https://kineticstoolkit.uqam.ca) is an open-source, pure-python package of integrated classes and functions that aims to facilitate research in biomechanics. It is mainly addressed to researchers and students in biomechanics with minimal experience in programming, who want to learn processing and controlling their research data using python. A special attention is made on documentation and interoperability with other environments (using pandas Dataframes and JSON files as intermediate data containers).

This is a long term project that is focused not only on the tool itself, but also a lot on learning. Please consult the "What is Kinetics Toolkit" section for more information on the project.

![GIF Animation](https://felixchenier.uqam.ca/wp-content/uploads/2020/05/Sample_ktk.Player_Wheelchair.gif)

Please ask questions and submit bugs or feature requests on the [git-hub issue tracker](https://github.com/felixchenier/kineticstoolkit/issues). While I develop Kinetics Toolkit primarily for my lab and have limited resources for troubleshooting, it will be my great pleasure to help if I can.

## Credits

Kinetics Toolkit is developed at the [Mobility and Adaptive Sports Research Lab](https://felixchenier.uqam.ca) by Professor Félix Chénier at [Université du Québec à Montréal](https://uqam.ca), Canada.

Thanks to Clay Flannigan for his [icp](https://github.com/ClayFlannigan/icp) method, to Benjamin Michaud for his [ezc3d](https://github.com/pyomeca/ezc3d) module, and to the dedicated people behind major software and packages used by Kinetics Toolkit, such as python, numpy, matplotlib, pandas, jupyter, pytest, sphinx, etc.


## Site map

```{toctree}
:caption: GETTING STARTED
:maxdepth: 2
11_what_is_kinetics_toolkit
12_install
13_conventions
```

```{toctree}
:caption: TUTORIALS
:maxdepth: 2
21_timeseries/00_timeseries
22_loadsave/00_loadsave
23_filters/00_filters
24_cycles/00_cycles
25_geometry/00_geometry
26_kinematics/00_kinematics
27_pushrimkinetics/00_pushrimkinetics
```

```{toctree}
:caption: IN DEPTH
31_lab_mode
32_release_notes
api_reference
Development website <https://felixchenier.uqam.ca/ktk_develop>
GitHub <https://github.com/felixchenier/kineticstoolkit>
```

:::{ifconfig} release in ['master']
    ```{toctree}
    :caption: FOR DEVELOPERS
    :maxdepth: 2
    41_install
    42_developer_manual
    43_dev/00_tutorials
    ```
:::
