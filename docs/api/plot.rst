========
Plotting
========

**PyAutoGalaxy** custom visualization library.

Step-by-step Juypter notebook guides illustrating all objects listed on this page are
provided on the `autogalaxy_workspace: plot tutorials <https://github.com/Jammy2211/autogalaxy_workspace/tree/release/notebooks/plot>`_ and
it is strongly recommended you use those to learn plot customization.

**Examples / Tutorials:**

- `autogalaxy_workspace: plot tutorials <https://github.com/Jammy2211/autogalaxy_workspace/tree/release/notebooks/plot>`_

Plotters [aplt]
---------------

Create figures and subplots showing quantities of standard **PyAutoGalaxy** objects.

.. currentmodule:: autogalaxy.plot

**Basic Plot Functions:**

.. autosummary::
   :toctree: _autosummary

    plot_array
    plot_grid

**Galaxy and Light / Mass Profile Subplots:**

.. autosummary::
   :toctree: _autosummary

    subplot_galaxy_light_profiles
    subplot_galaxy_mass_profiles
    subplot_basis_image
    subplot_galaxies
    subplot_galaxy_images
    subplot_adapt_images

**Imaging Fit Subplots:**

.. autosummary::
   :toctree: _autosummary

    subplot_fit_imaging
    subplot_fit_imaging_of_galaxy

**Interferometer Fit Subplots:**

.. autosummary::
   :toctree: _autosummary

    subplot_fit_interferometer
    subplot_fit_dirty_images
    subplot_fit_real_space

**Quantity Fit Subplots:**

.. autosummary::
   :toctree: _autosummary

    subplot_fit_quantity

**Ellipse Fit Subplots:**

.. autosummary::
   :toctree: _autosummary

    subplot_fit_ellipse
    subplot_ellipse_errors

Non-linear Search Plotters [aplt]
---------------------------------

Create figures and subplots of non-linear search specific visualization of every search algorithm supported
by **PyAutoGalaxy**.

.. currentmodule:: autogalaxy.plot

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   NestPlotter
   MCMCPlotter
   MLEPlotter

Plot Customization [aplt]
-------------------------

Customize figures created via ``Plotter`` objects, including changing ``matplotlib`` settings and adding
visuals to figures.

.. currentmodule:: autogalaxy.plot

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   MatPlot1D
   MatPlot2D
   Visuals1D
   Visuals2D

Matplot Lib Wrappers [aplt]
---------------------------

Wrappers for every ``matplotlib`` function used by a ``Plotter``, allowing for detailed customization of
every figure and subplot.

.. currentmodule:: autogalaxy.plot

**Matplotlib Wrapper Base Objects:**

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   Units
   Figure
   Axis
   Cmap
   Colorbar
   ColorbarTickParams
   TickParams
   YTicks
   XTicks
   Title
   YLabel
   XLabel
   Legend
   Output

**Matplotlib Wrapper 1D Objects:**

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   YXPlot

**Matplotlib Wrapper 2D Objects:**

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   ArrayOverlay
   GridScatter
   GridPlot
   VectorYXQuiver
   PatchOverlay
   VoronoiDrawer
   OriginScatter
   MaskScatter
   BorderScatter
   PositionsScatter
   IndexScatter
   MeshGridScatter
   ParallelOverscanPlot
   SerialPrescanPlot
   SerialOverscanPlot