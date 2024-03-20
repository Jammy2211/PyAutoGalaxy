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

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

    Array2DPlotter
    Grid2DPlotter
    MapperPlotter
    YX1DPlotter
    InversionPlotter
    ImagingPlotter
    InterferometerPlotter
    LightProfilePlotter
    LightProfilePDFPlotter
    GalaxyPlotter
    FitImagingPlotter
    FitInterferometerPlotter
    GalaxiesPlotter
    AdaptPlotter
    FitImagingPlotter
    FitInterferometerPlotter
    MultiFigurePlotter
    MultiYX1DPlotter

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
   OptimizePlotter

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
   Include1D
   Include2D
   Visuals1D
   Visuals2D

Matplot Lib Wrappers [aplt]
---------------------------

Wrappers for every ``matplotlib`` function used by a ``Plotter``, allowing for detailed customizaiton of
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