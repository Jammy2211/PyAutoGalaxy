What is PyAutoGalaxy?
=====================

Nearly a century ago, Edwin Hubble famously classified galaxies into three distinct groups: ellipticals, spirals and
irregulars. Today, by analysing millions of galaxies with advanced image processing techniques Astronomers have
expanded on this picture and revealed the rich diversity of galaxy morphology both in the nearby and distant
Universe.

``PyAutoGalaxy`` is an open-source Python 3.8 - 3.11 package for analysing the morphologies and structures of large
multi-wavelength galaxy samples. **PyAutoGalaxy** makes it simple to model galaxies, for example this Hubble Space
Telescope imaging of a spiral galaxy:

|pic1|

.. |pic1| image:: https://github.com/Jammy2211/PyAutoGalaxy/blob/main/paper/hstcombined.png?raw=true

**PyAutoGalaxy** also fits interferometer data from observatories such as ALMA:

|pic2|

.. |pic2| image:: https://github.com/Jammy2211/PyAutoGalaxy/blob/main/paper/almacombined.png?raw=true


Getting Started
===============

The following links are useful for new starters:

- `The PyAutoGalaxy readthedocs <https://pyautogalaxy.readthedocs.io/en/latest>`_, which includes `an installation guide <https://pyautogalaxy.readthedocs.io/en/latest/installation/overview.html>`_ and `an overview of PyAutoGalaxy's core features <https://pyautogalaxy.readthedocs.io/en/latest/overview/overview_1_galaxies.html>`_.

- `The introduction Jupyter Notebook on Binder <https://mybinder.org/v2/gh/Jammy2211/autogalaxy_workspace/release?filepath=introduction.ipynb>`_, where you can try **PyAutoGalaxy** in a web browser (without installation).

- `The autogalaxy_workspace GitHub repository <https://github.com/Jammy2211/autogalaxy_workspace>`_, which includes example scripts and the `HowToGalaxy Jupyter notebook lectures <https://github.com/Jammy2211/autogalaxy_workspace/tree/master/notebooks/howtogalaxy>`_ which give new users a step-by-step introduction to **PyAutoGalaxy**.

Features
========

Core features include fully automated Bayesian model-fitting of galaxy two-dimensional surface brightness profiles,
support for dataset and interferometer datasets and comprehensive tools for simulating galaxy images. The software
places a focus on **big data** analysis, including support for hierarchical models that simultaneously fit thousands of
galaxies, massively parallel model-fitting and an SQLite3 database that allows large suites of modeling results to be
loaded, queried and analysed.

The software comes distributed with the **HowToGalaxy** Jupyter notebook lectures, which are written assuming no
previous knowledge about galaxy structure and teach a new user theory and statistics required to analyse
galaxy data. Checkout `the howtogalaxy section of
the readthedocs <https://pyautogalaxy.readthedocs.io/en/latest/howtogalaxy/howtogalaxy.html>`_.

An overview of **PyAutoGalaxy**'s core features can be found in
the `overview section of the readthedocs <https://pyautogalaxy.readthedocs.io/en/latest/overview/lensing.html>`_.

Galaxy Morphology & Structure
=============================

The study of galaxy morphology aims to understand the different luminous structures that galaxies are composed of.
Using large CCD imaging datasets of galaxies observed at ultraviolet, optical and near-infrared wavelengths from
instruments like the Hubble Space Telescope (HST), Astronomers have uncovered the plentiful structures that make up
a galaxy (e.g. bars, bulges, disks and rings) and revealed that evolving galaxies transition from disk-like structures
to bulge-like elliptical galaxies. At sub-mm and radio wavelengths interferometer datasets from instruments like the
Atacama LAM (ALMA) have revealed the integral role that dust plays in forming a galaxy in the distant Universe, early
in its lifetime. Studies typically represent a galaxy's light using analytic functions such as the Sersic
profile, which quantify the global appearance of most galaxies into one of three groups: (i) bulge-like
structures which follow a dev Vaucouleurs profile; (ii) disk-like structures which follow an Exponential profile
or; (iii) irregular morphologies which are difficult to quantify with symmetric and smooth analytic profiles. Galaxies
are often composed of many sub-components which may be a combination of these different structures.

In the next decade, wide field surveys such as Euclid, the Vera Rubin Observatory and Square Kilometer array are
poised to observe images of _billions_ of galaxies. Analysing these extremely large galaxy datasets demands
advanced Bayesian model-fitting techniques which can scale-up in a fully automated manner. Equally, the James Webb
Space Telescope, thirty-meter class ground telescopes and ?radio? will observe galaxies at an
unprecedented resolution and level of detail. This demands more flexible modeling techniques that can accurately
represent the complex irregular structures one such high resolution observations reveal. ``PyAutoGalaxy`` aims to meet
both these needs, by interfacing galaxy model-fitting with the probabilistic programming language ``PyAutoFit`` to
provide Bayesian fitting tools suited to big data analysis alongside image processing tools that represent irregular
galaxy structures using non-parametric models.

How does PyAutoGalaxy Work?
===========================

At the heart of the `PyAutoGalaxy` API is the `Galaxy` object, which groups together one or more `LightProfile` objects
at an input redshift. Passing these objects a `Grid2D` returns an image of the galaxy(s), which can subsequently
be passed through `Operator` objects to apply a 2D convolution or Fast Fourier Transform and thereby compare
the `Galaxy`'s image to an imaging or interferometer dataset. The `inversion` package contains a range of non-parametric
models which fit a galaxy's light using a Bayesian linear matrix inversion. The `astropy` cosmology module is
used to handle unit conversions and calculations are optimized using the packages `NumPy` [@numpy], `numba` [@numba]
and `PyNUFFT` [@pynufft].

.. code-block:: python

    import autogalaxy as ag
    import autogalaxy.plot as aplt

    """
    To describe the galaxy emission two-dimensional grids of (y,x) Cartesian
    coordinates are used.
    """
    grid = ag.Grid2D.uniform(
        shape_native=(50, 50),
        pixel_scales=0.05,  # <- Conversion from pixel units to arc-seconds.
    )

    """
    The galaxy has an elliptical sersic light profile representing its bulge.
    """
    bulge = ag.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=3.0,
    )

    """
    The galaxy also has an elliptical exponential disk
    """
    disk = ag.lp.Exponential(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.7, angle=30.0),
        intensity=0.5,
        effective_radius=1.6,
    )

    """
    We combine the above light profiles to compose a galaxy at redshift 1.0.
    """
    galaxy = ag.Galaxy(redshift=1.0, bulge=bulge, disk=disk)

    """
    We can use the grid and galaxies to perform many calculations, for example
    plotting the image of the galaxies.
    """
    galaxies_plotter = aplt.GalaxiesPlotter(galaxies=[galaxy], grid=grid)
    galaxies_plotter.figures_2d(image=True)


To perform model-fitting, `PyAutoGalaxy` adopts the probabilistic programming
language `PyAutoFit` (https://github.com/rhayes777/PyAutoFit). `PyAutoFit` allows users to compose a
model from `LightProfile` and `Galaxy` objects, customize the model parameterization and fit it to data via a
non-linear search (e.g., `dynesty` [@dynesty], `emcee` [@emcee], `PySwarms` [@pyswarms]). By composing a model with
`Pixelization`  objects, the galaxy's light is reconstructed using a non-parametric rectangular
grid or Voronoi mesh that accounts for irregular galaxy morphologies.

.. code-block:: python

    import autofit as af
    import autogalaxy as ag

    import os

    """
    Load Imaging data of the galaxy from the dataset folder of the workspace.
    """
    dataset = ag.Imaging.from_fits(
        data_path="/path/to/dataset/image.fits",
        noise_map_path="/path/to/dataset/noise_map.fits",
        psf_path="/path/to/dataset/psf.fits",
        pixel_scales=0.1,
    )

    """
    Create a mask for the data, which we setup as a 3.0" circle.
    """
    mask = ag.Mask2D.circular(
        shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
    )

    """
    We model the galaxy using an Sersic LightProfile.
    """
    light_profile = ag.lp.Sersic

    """
    We next setup this profile as model components whose parameters are free & fitted for
    by setting up a Galaxy as a Model.
    """
    galaxy_model = af.Model(ag.Galaxy, redshift=1.0, light=light_profile)
    model = af.Collection(galaxy=galaxy_model)

    """
    We define the non-linear search used to fit the model to the data (in this case, Dynesty).
    """
    search = af.Nautilus(name="search[example]", n_live=50)

    """
    We next set up the `Analysis`, which contains the `log likelihood function` that the
    non-linear search calls to fit the lens model to the data.
    """
    analysis = ag.AnalysisImaging(dataset=masked_dataset)

    """
    To perform the model-fit we pass the model and analysis to the search's fit method. This will
    output results (e.g., dynesty samples, model parameters, visualization) to hard-disk.
    """
    result = search.fit(model=model, analysis=analysis)

    """
    The results contain information on the fit, for example the maximum likelihood
    model from the Dynesty parameter space search.
    """
    print(result.samples.max_log_likelihood())

.. toctree::
   :caption: Overview:
   :maxdepth: 1
   :hidden:

   overview/overview_1_galaxies
   overview/overview_2_fitting
   overview/overview_3_modeling
   overview/overview_4_simulate
   overview/overview_5_pixelizations
   overview/overview_6_interferometry
   overview/overview_7_multi_wavelength

.. toctree::
   :caption: Installation:
   :maxdepth: 1
   :hidden:

   installation/overview
   installation/conda
   installation/pip
   installation/numba
   installation/source
   installation/troubleshooting

.. toctree::
   :caption: General:
   :maxdepth: 1
   :hidden:

   general/workspace
   general/configs
   general/model_cookbook
   general/likelihood_function
   general/citations
   general/credits

.. toctree::
   :caption: Tutorials:
   :maxdepth: 1
   :hidden:

   howtogalaxy/howtogalaxy
   howtogalaxy/chapter_1_introduction
   howtogalaxy/chapter_2_modeling
   howtogalaxy/chapter_3_search_chaining
   howtogalaxy/chapter_4_pixelizations
   howtogalaxy/chapter_optional

.. toctree::
   :caption: API Reference:
   :maxdepth: 1
   :hidden:

   api/data
   api/light
   api/galaxy
   api/fitting
   api/modeling
   api/pixelization
   api/plot
   api/source

.. toctree::
   :caption: Advanced:
   :maxdepth: 1
   :hidden:

   advanced/database
   advanced/chaining
   advanced/graphical
