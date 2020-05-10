.. _installation:

Installation
============

Installation with pip
---------------------

The simplest way to install **PyAutoGalaxy** is via pip which installs **PyAutoGalaxy** with the following dependencies:

**PyAutoConf** https://github.com/rhayes777/PyAutoConf

**Dynesty** https://github.com/joshspeagle/dynesty

**emcee** https://github.com/dfm/emcee

**astropy** https://www.astropy.org/

**GetDist** https://getdist.readthedocs.io/en/latest/

**matplotlib** https://matplotlib.org/

**numpy** https://numpy.org/

**scipy** https://www.scipy.org/

You can install **PyAutoGalaxy** via pip as follows:

.. code-block:: bash

    pip install autogalaxy

Clone autogalaxy workspace & set WORKSPACE environment model:

.. code-block:: bash

    cd /path/where/you/want/autogalaxy_workspace
    git clone https://github.com/Jammy2211/autogalaxy_workspace
    export WORKSPACE=/path/to/autogalaxy_workspace/

Set PYTHONPATH to include the autogalaxy_workspace directory:

.. code-block:: bash

    export PYTHONPATH=/path/to/autogalaxy_workspace

Matplotlib uses the default backend on your computer, as set in the config file:

.. code-block:: bash

    autogalaxy_workspace/config/visualize/general.ini

If unchanged, the backend is set to 'default', meaning it will use the backend automatically set up for Python on
your system.

.. code-block:: bash

    [general]
    backend = default

There have been reports that using the default backend causes crashes when running the test script below (either the
code crashes without a error or your computer restarts). If this happens, change the config's backend until the test
works (TKAgg has worked on Linux machines, Qt5Agg has worked on new MACs). For example:

.. code-block:: bash

    [general]
    backend = TKAgg

You can test everything is working by running the following command in the autogalaxy_workspace:

.. code-block:: bash

    python3 /path/to/autogalaxy_workspace/api/simple/galaxy.py

PyMultiNest
-----------

Installation via pip omits an optional dependency, the nested sampling algorithm
`PyMultiNest <http://johannesbuchner.github.io/pymultinest-tutorial/install.html>`_. If you require **PyMultiNest** you
either need too install **PyAutoGalaxy** via conda following the instructions below or will need to install **MultiNest**
`at this link <http://johannesbuchner.github.io/pymultinest-tutorial/install.html>`_.

Installation with conda
-----------------------

First, install `conda <https://conda.io/miniconda.html>`_.

Create a conda environment:

.. code-block:: bash

    >> conda create -n autogalaxy python=3.7 anaconda


Activate the conda environment:

.. code-block:: bash

    conda activate autogalaxy


Install multinest:

.. code-block:: bash

    conda install -c conda-forge multinest


Install autogalaxy:

.. code-block:: bash

    pip install autogalaxy


Clone the autogalaxy workspace & set WORKSPACE environment model:

.. code-block:: bash

    cd /path/where/you/want/autogalaxy_workspace
    git clone https://github.com/Jammy2211/autogalaxy_workspace
    export WORKSPACE=/path/to/autogalaxy_workspace/


Set PYTHONPATH to include the autogalaxy_workspace directory:

.. code-block:: bash

    export PYTHONPATH=/path/to/autogalaxy_workspace/

Matplotlib uses the default backend on your computer, as set in the config file:

.. code-block:: bash

    autogalaxy_workspace/config/visualize/general.ini

If unchanged, the backend is set to 'default', meaning it will use the backend automatically set up for Python on
your system.

.. code-block:: bash

    [general]
    backend = default

There have been reports that using the default backend causes crashes when running the test script below (either the
code crashes without a error or your computer restarts). If this happens, change the config's backend until the test
works (TKAgg has worked on Linux machines, Qt5Agg has worked on new MACs). For example:

.. code-block:: bash

    [general]
    backend = TKAgg


You can test everything is working by running the example pipeline runner in the autogalaxy_workspace

.. code-block:: bash

    python3 /path/to/autogalaxy_workspace/runners/beginner/no_galaxy_light/galaxy_sie__source_inversion.py

Forking / Cloning
-----------------

Alternatively, you can fork or clone the **PyAutoGalaxy** github repository. Note that **PyAutoGalaxy** requires a valid
config to run. Therefore, if you fork or clone the **PyAutoGalaxy** repository, you need the
`autogalaxy_workspace <https://github.com/Jammy2211/autogalaxy_workspace>`_ with the PYTHONPATH and WORKSPACE environment
variables set up as described on the `autogalaxy_workspace <https://github.com/Jammy2211/autogalaxy_workspace>`_ repository
or the installation instructions below.