Kinematics analysis
===================

The kinematics_ module allows opening trajectories of markers from c3d or n3d
files, and process those trajectory to:

- create rigid body configurations based on a static acquisitions;
- register the markers trajectories to rigid body trajectories;
- create virtual marker configuration based on probing acquisitions;
- reconstruct these virtual markers from rigid body trajectories;
- and other operations that will be implemented in the future.

Here is a quick example of how to reconstruct complete kinematics during a task, based on a static acquisition and probing acquisitions.

.. _kinematics: ../api/kineticstoolkit.kinematics.rst

.. toctree::
    :maxdepth: 2

    load_visualize
    simple_analysis
    reconstruction
