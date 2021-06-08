kineticstoolkit_joss
====================

This is the repository for the manuscript entitled "Kinetics Toolkit, an open-source python package to facilitate research in biomechanics", for publication in Journal of Open Source Software.

Author: Félix Chénier

Some review criteria (to help the reviewers)
--------------------------------------------

### Hosting

Kinetics Toolkit source is hosted on github at this address: https://github.com/felixchenier/kineticstoolkit

### Software license

Kinetics Toolkit is distributed under the Apache 2.0 license. See LICENSE.txt in the main repository.

### Substantial scholarly effort

Kinetics Toolkit has been developed first as a Matlab toolkit between 2015 and 2019, and was then converted and expanded as a Python toolkit from 2019 up to now. It has been open source since July 2019. Initially started as a software framework for my emerging research lab, it matured in a form that I now consider helpful for lots of other researchers.

Having a formation in Engineering, I have a professor position in a Physical Activity Science department and I think I have a good understanding of the knowledge and abilities of typical students and researchers in this domain. Kinetics Toolkit is oriented directly toward this clientele, and no other package I know of have this same objective.

Kinetics Toolkit has yet to be cited in literature because I wanted to publish a paper in JOSS before publicizing the toolkit, so that it can be referred using this paper. While at our lab, we personally used and tweaked Kinetics Toolkit over the years, I now consider this toolkit mature enough to be used with confidence by other researchers.

### Documentation

All documentation, including biomechanics basics, tutorials, API, statement of need and installation instructions, community guidelines (how to contribute) are available at https://kineticstoolkit.uqam.ca.

The development website, which refers to the most up to date master branch, also documents unstable features being developed, and is available at https://kineticstoolkit.uqam.ca/master.

### Tests

Kinetics Toolkit uses several way to ensure the code integrity. For each commit on the master branch, the following items are checked:

- Types are checked using mypy;
- API documentation is tested using doctest;
- Modules and class methods are tested using unit tests;
- All tutorials are rebuilt and checked for failure using nbconvert;
- The whole website is rebuilt and checked for failure using sphinx.

Unit tests are provided into the `test` folder, and data for tests and tutorials are provided in the `data` folder.
