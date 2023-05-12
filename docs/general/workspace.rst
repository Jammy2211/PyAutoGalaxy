.. _workspace:

Workspace Tour
==============

You should have downloaded and configured the `autogalaxy workspace <https://github.com/Jammy2211/autogalaxy_workspace>`_
when you installed **PyAutoGalaxy**. If you didn't, checkout the
`installation instructions <https://pyautogalaxy.readthedocs.io/en/latest/general/installation.html#installation-with-pip>`_
for how to downloaded and configure the workspace.

The ``README.rst`` files distributed throughout the workspace describe every folder and file, and specify if
examples are for beginner or advanced users.

New users should begin by checking out the following parts of the workspace.

HowToGalaxy
-----------

The **HowToGalaxy** lecture series are a collection of Jupyter notebooks describing how to build a **PyAutoGalaxy** model
fitting project and giving illustrations of different statistical methods and techniques.

Checkout the
`tutorials section <file:///Users/Jammy/Code/PyAuto/PyAutoGalaxy/docs/_build/howtogalaxy/howtogalaxy.html>`_ for a
full description of the lectures and online examples of every notebook.

Scripts / Notebooks
-------------------

There are numerous example describing how to perform calculations, galaxy modeling, and many other
**PyAutoGalaxy** features. All examples are provided as Python scripts and Jupyter notebooks.

A full description of the scripts available is given on
the `autogalaxy workspace GitHub page <https://github.com/Jammy2211/autogalaxy_workspace>`_.

Config
------

Here, you'll find the configuration files which customize:

    - The default settings used by every non-linear search.
    - Visualization, including the backend used by *matplotlib*.
    - The priors and notation configs associated with the light and mass profiles used for lens modeling.
    - The behaviour of different (y,x) Cartesian grids used to perform lens calculations.
    - The generaltrue config which customizes other aspects of **PyAutoGalaxy**.

Descriptions of every configuration file and their input parameters are provided in the ``README.rst`` in
the `config directory of the workspace <https://github.com/Jammy2211/autogalaxy_workspace/tree/release/config>`_

Dataset
-------

Contains the dataset's used to perform lens modeling. Example datasets using simulators included with the workspace
are included here by default.

Output
------

The folder where  modeling results are stored.
