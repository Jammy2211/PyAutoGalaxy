PyAutoGalaxy: Open-Source Multi Wavelength Galaxy Structure & Morphology
========================================================================

.. |nbsp| unicode:: 0xA0
    :trim:

.. |binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/Jammy2211/autogalaxy_workspace/HEAD

.. |code-style| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

|binder| |code-style|

`Installation Guide <https://pyautogalaxy.readthedocs.io/en/latest/installation/overview.html>`_ |
`readthedocs <https://pyautogalaxy.readthedocs.io/en/latest/index.html>`_ |
`Introduction on Binder <https://mybinder.org/v2/gh/Jammy2211/autogalaxy_workspace/release?filepath=introduction.ipynb>`_ |
`HowToGalaxy <https://pyautogalaxy.readthedocs.io/en/latest/howtogalaxy/howtogalaxy.html>`_

The study of a galaxy's structure and morphology is at the heart of modern day Astrophysical research.

**PyAutoGalaxy** makes it simple to model galaxies, for example this Hubble Space Telescope imaging of a spiral
galaxy:

|pic1|

.. |pic1| image:: https://github.com/Jammy2211/PyAutoGalaxy/blob/master/paper/hstcombined.png

**PyAutoGalaxy** also fits interferometer data from observatories such as ALMA:


|pic2|

.. |pic2| image:: https://github.com/Jammy2211/PyAutoGalaxy/blob/master/paper/almacombined.png

Getting Started
---------------

The following links are useful for new starters:

- `The introduction Jupyter Notebook on Binder <https://mybinder.org/v2/gh/Jammy2211/autogalaxy_workspace/release?filepath=introduction.ipynb>`_, where you can try **PyAutoGalaxy** in a web browser (without installation).

- `The PyAutoGalaxy readthedocs <https://pyautogalaxy.readthedocs.io/en/latest>`_, which includes `an installation guide <https://pyautogalaxy.readthedocs.io/en/latest/installation/overview.html>`_ and an overview of **PyAutoGalaxy**'s core features.

- `The autogalaxy_workspace GitHub repository <https://github.com/Jammy2211/autogalaxy_workspace>`_, which includes example scripts and the `HowToGalaxy Jupyter notebook tutorials <https://github.com/Jammy2211/autogalaxy_workspace/tree/master/notebooks/howtogalaxy>`_ which give new users a step-by-step introduction to **PyAutoGalaxy**.


API Overview
------------

Galaxy morphology calculations are performed in **PyAutoGalaaxy** by building a ``Plane`` object from ``LightProfile``
and ``Galaxy`` objects. Below, we create a simple galaxy system where a redshift 0.5
``Galaxy`` with an ``EllSersic`` ``LightProfile`` representing a bulge and an ``EllExponential`` ``LightProfile``
representing a disk.

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
    bulge=ag.lp.EllSersic(
        centre=(0.0, 0.0),
        elliptical_comps=ag.convert.elliptical_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=3.0,
    )

    """
    The galaxy also has an elliptical exponential disk
    """
    disk = ag.lp.EllExponential(
        centre=(0.0, 0.0),
        elliptical_comps=ag.convert.elliptical_comps_from(axis_ratio=0.7, angle=30.0),
        intensity=0.5,
        effective_radius=1.6,
    )

    """
    We combine the above light profiles to compose a galaxy at redshift 1.0.
    """
    galaxy = ag.Galaxy(redshift=1.0, bulge=bulge, disk=disk)

    """
    We create a Plane, which in this example has just one galaxy but can
    be extended for datasets with many galaxies.
    """
    plane = ag.Plane(
        galaxies=[galaxy],
    )

    """
    We can use the Grid2D and Plane to perform many calculations, for example
    plotting the image of the galaxyed source.
    """
    plane_plotter = aplt.PlanePlotter(plane=plane, grid=grid)
    plane_plotter.figures_2d(image=True)


With **PyAutoGalaxy**, you can begin modeling a galaxy in just a couple of minutes. The example below demonstrates a
simple analysis which fits a galaxy's light.

.. code-block:: python

    import autofit as af
    import autogalaxy as ag

    import os

    """
    Load Imaging data of the strong galaxy from the dataset folder of the workspace.
    """
    imaging = ag.Imaging.from_fits(
        image_path="/path/to/dataset/image.fits",
        noise_map_path="/path/to/dataset/noise_map.fits",
        psf_path="/path/to/dataset/psf.fits",
        pixel_scales=0.1,
    )

    """
    Create a mask for the data, which we setup as a 3.0" circle.
    """
    mask = ag.Mask2D.circular(
        shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.0
    )

    """
    We model the galaxy using an EllSersic LightProfile.
    """
    light_profile = ag.lp.EllSersic

    """
    We next setup this profile as model components whose parameters are free & fitted for
    by setting up a Galaxy as a Model.
    """
    galaxy_model = af.Model(ag.Galaxy, redshift=1.0, light=light_profile)
    model = af.Collection(galaxy=galaxy_model)

    """
    We define the non-linear search used to fit the model to the data (in this case, Dynesty).
    """
    search = af.DynestyStatic(name="search[example]", nlive=50)
    
    """
    We next set up the `Analysis`, which contains the `log likelihood function` that the
    non-linear search calls to fit the galaxy model to the data.
    """
    analysis = ag.AnalysisImaging(dataset=masked_imaging)

    """
    To perform the model-fit we pass the model and analysis to the search's fit method. This will
    output results (e.g., dynesty samples, model parameters, visualization) to hard-disk.
    """
    result = search.fit(model=model, analysis=analysis)

    """
    The results contain information on the fit, for example the maximum likelihood
    model from the Dynesty parameter space search.
    """
    print(result.samples.max_log_likelihood_instance)


Support
-------

Support for installation issues, help with galaxy modeling and using **PyAutoGalaxy** is available by
`raising an issue on the GitHub issues page <https://github.com/Jammy2211/PyAutoGalaxy/issues>`_.

We also offer support on the **PyAutoGalaxy** `Slack channel <https://pyautogalaxy.slack.com/>`_, where we also provide the
latest updates on **PyAutoGalaxy**. Slack is invitation-only, so if you'd like to join send
an `email <https://github.com/Jammy2211>`_ requesting an invite.
