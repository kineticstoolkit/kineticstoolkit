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

# What is Kinetics Toolkit

*The Summary and Statement of need sections are from the white paper published in*
[Journal of Open Source Software](https://joss.theoj.org/papers/10.21105/joss.03714).

## Summary

Kinetics Toolkit is a Python package for generic biomechanical analysis of human motion that is easily accessible by new programmers. The only prerequisite for using this toolkit is having minimal to moderate skills in Python and Numpy.

While Kinetics Toolkit provides a dedicated class for containing and manipulating data (`TimeSeries`), it loosely follows a procedural programming paradigm where processes are grouped as interrelated functions in different submodules, which is consistent with how people are generally introduced to programming. Each function has a limited and well-defined scope, making Kinetics Toolkit generic and expandable. Particular care is given to documentation, with extensive tutorials and API references. Special attention is also given to interoperability with other software programs by using Pandas Dataframes (and therefore CSV files, Excel files, etc.), JSON files or C3D files as intermediate data containers.

Kinetics Toolkit is accessible at https://kineticstoolkit.uqam.ca and is distributed via conda and pip.


## Statement of need

The last decade has been marked by the development of several powerful open-source software programs in biomechanics. Examples include:
[OpenSim](https://doi.org/10.1371/journal.pcbi.1006223),
[SimBody](https://doi.org/10.1016/j.piutam.2011.04.023),
[Biordb](https://doi.org/10.21105/joss.02562),
[BiomechZoo](https://doi.org/10.1016/j.cmpb.2016.11.007),
[Pinocchio](https://doi.org/10.1109/SII.2019.8700380),
[FreeBody](https://doi.org/10.1098/rsos.140449),
[CusToM](https://doi.org/10.21105/joss.00927),
as well as many others. However, many of these tools are rather specific (e.g., musculoskeletal modelling, neuromuscular optimization, etc.) and not especially well suited for performing generic processing of human motion data such as filtering data, segmenting cycles, changing coordinate systems, etc. Other software programs, while being open source, rely on expensive closed-source software such as Matlab (Mathworks LCC, Naticks, USA).

While Matlab has a long and successful history in biomechanical analysis, it is quickly becoming challenged by the free and open-source Python scientific ecosystem, particularly by powerful packages, including
[Numpy](https://doi.org/10.1038/s41586-020-2649-2),
[Matplotlib](https://doi.org/10.1109/MCSE.2007.55),
[SciPy](https://doi.org/10.1038/s41592-019-0686-2) and
[Pandas](https://pandas.pydata.org).
Since Python is regarded as a [robust introductory programming language for algorithm development](https://doi.org/10.1007/978-3-540-25944-2_157), it may be an ideal tool for new programmers in biomechanics.

The [Pyomeca](https://doi.org/10.21105/joss.02431) toolbox is a Python library for biomechanical analysis. It uses an object-oriented programming paradigm where each data class (`Angles`, `Rototrans`, `Analogs`, `Markers`) subclasses [xarray](https://doi.org/10.5334/jors.148), and where the data processing functions are accessible as class methods. While this paradigm may be compelling from a programmer's perspective, it requires users to master xarray and object-oriented concepts such as class inheritance, which are not as straightforward to learn, especially for new programmers who may just be starting out with Python and Numpy.

With this beginner audience in mind, Kinetics Toolkit is a Python package for generic biomechanical analysis of human motion. It is a user-friendly tool for people with little experience in programming, yet elegant, fun to use and still appealing to experienced programmers. Designed with a mainly procedural programming paradigm, its data processing functions can be used directly as examples so that users can build their own scripts, functions, and even modules, and therefore make Kinetics Toolkit fit their own specific needs.


## Objectives of this project

- Build an intuitive and elegant Python package that leverages the power of biomechanical analysis to people who are not formed in programming or engineering;

- Build a strong documentation around this package to:
    - Teach or remind the basic principles of biomechanics to newcomers;
    - Explain how Kinetics Toolkit implements these principles; 
    - Teach users how to use Kinetics Toolkit on their data without relying on do-it-all, commercial graphical interfaces.

## How to contribute

At the moment, I think Kinetics Toolkit has made a moderate way into its first objective. There is however still much to do. If you find this long-term project appealing and want to contribute, here are different ways to help:

- Try it and tell me your thoughts and ideas on the interface, features and documentation;
- Test it and report bugs;
- Communicate with me to help with documentation;
- Extend it to develop missing functionality and share the results;
- Citing it in your work;
- Possibly many other ways to help.

A good start would be to check out the [development website](https://felixchenier.uqam.ca/ktk_develop) or fork the repository on [GitHub](https://github.com/felixchenier/kineticstoolkit).


## How to cite

If you find Kinetics Toolkit useful and want to credit it in your work (which I would much appreciate), then please cite this paper:

Chénier, F., 2021. Kinetics Toolkit: An Open-Source Python Package to Facilitate Research in Biomechanics.
*Journal of Open Source Software* 6(66), 3714. https://doi.org/10.21105/joss.03714

---

I hope Kinetics Toolkit may help you in your biomechanical analyses; have a good read!
