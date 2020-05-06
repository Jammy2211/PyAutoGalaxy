PyAutoGalaxy
==========

The study of a galaxy's light, structure and dynamics is at the heart of modern day Astrophysical research.
**PyAutoGalaxy** makes it simple to model galaxies, like this oidkne:

Missing for now :(

Example
-------

With **PyAutoGalaxy**, you can begin modeling a galaxy in just a couple of minutes. The example below demonstrates a
simple analysis which fits a galaxy's light.

.. code-block:: python

    import autofit as af
    import autogalaxy as ag

    import os

    # In this example, we'll fit an image of a single galaxy .
    dataset_path = '{}/../data/'.format(os.path.dirname(os.path.realpath(__file__)))

    galaxy_name = 'example_galaxy'

    # Use the relative path to the dataset to load the imaging data.
    imaging = ag.Imaging.from_fits(
        image_path=dataset_path + galaxy_name + '/image.fits',
        psf_path=dataset_path+galaxy_name+'/psf.fits',
        noise_map_path=dataset_path+galaxy_name+'/noise_map.fits',
        pixel_scales=0.1)

    # Create a mask for the data, which we setup as a 3.0" circle.
    mask = ag.Mask.circular(shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0)

    # We model our galaxy using a light profile (an elliptical Sersic).
    light_profile = ag.lp.EllipticalSersic

    # To setup our model galaxy, we use the GalaxyModel class, which represents a galaxy whose parameters
    # are free & fitted for by PyAutoGalaxy. The galaxy is also assigned a redshift.
    galaxy_model = ag.GalaxyModel(redshift=1.0, light=light_profile)

    # To perform the analysis we set up a phase, which takes our galaxy model & fits its parameters using a non-linear
    # search (in this case, MultiNest).
    phase = ag.PhaseImaging(
        galaxies=dict(galaxy=galaxy_model),
        phase_name='example/phase_example',
        non_linear_class=af.MultiNest
        )

    # We pass the imaging data and mask to the phase, thereby fitting it with the galaxy model & plot the resulting fit.
    result = phase.run(data=imaging, mask=mask)
    ag.plot.FitImaging.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

Features
--------

**PyAutoGalaxy's** advanced modeling features include:

- **Profiles** - Use light & mass profiles to make galaxies & perform studies of galaxy light, structure and dynamics.
- **Pipelines** - Write automated analysis pipelines to fit complex models to large samples of galaxies.
- **Pixelizations** - Reconstruct complex galaxy morphologies on a variety of pixel-grids.
- **Adaption** - Adapt the galaxy analysis to the features of the observed galaxy imaging.
- **Interferometry** - Model radio / sub-mm interferometer data directly in the uv-plane.
- **Visualization** - Custom visualization libraries for plotting physical galaxying quantities & modeling results.
- **PyAutoFit** - Perform fits using many non-linear searches (MCMC, Nested Sampling) and manipulate large result outputs
                  via the probablistic programming language `PyAutoFit <https://github.com/rhayes777/PyAutoFit>`_

HowToGalaxy
---------

Included with **PyAutoGalaxy** is the **HowToGalaxy** lecture series, which provides an introduction to galaxy modeling 
with **PyAutoGalaxy**. It can be found in the workspace & consists of 5 chapters:

- **Introduction** - An introduction to galaxy fitting & **PyAutoGalaxy**.
- **Galaxy Modeling** - How to perform model fits of galaxies, including a primer on Bayesian non-linear analysis.
- **Pipelines** - How to build model-fitting pipelines & tailor them to your own science case.
- **Inversions** - How to perform pixelized reconstructions of a galaxy.
- **Hyper-Mode** - How to use **PyAutoGalaxy** advanced modeling features that adapt the model to the galaxy being analysed.

Workspace
---------

**PyAutoGalaxy** comes with a workspace, which can be found `here <https://github.com/Jammy2211/autogalaxy_workspace>`_ & which includes:

- **Aggregator** - Manipulate large suites of modeling results via Jupyter notebooks, using **PyAutoFit**'s in-built results database.
- **API** - Illustrative scripts of the **PyAutoGalaxy** interface, for examples on how to make plots, perform galaxy calculations, etc.
- **Config** - Configuration files which customize **PyAutoGalaxy**'s behaviour.
- **Dataset** - Where data is stored, including example datasets distributed with **PyAutoGalaxy**.
- **HowToGalaxy** - The **HowToGalaxy** lecture series.
- **Output** - Where the **PyAutoGalaxy** analysis and visualization are output.
- **Pipelines** - Example pipelines for modeling galaxies.
- **Preprocess** - Tools to preprocess data before an analysis (e.g. convert units, create masks).
- **Quick Start** - A quick start guide, so you can begin modeling galaxies within hours.
- **Runners** - Scripts for running **PyAutoGalaxy** pipelines.
- **Simulators** - Scripts for simulating galaxy datasets with **PyAutoGalaxy**.

Slack
-----

We're building a **PyAutoGalaxy** community on Slack, so you should contact us on our
`Slack channel <https://pyautogalaxy.slack.com/>`_ before getting started. Here, I will give you the latest updates on
the software & discuss how best to use **PyAutoGalaxy** for your science case.

Unfortunately, Slack is invitation-only, so first send me an `email <https://github.com/Jammy2211>`_ requesting an
invite.

Documentation & Installation
----------------------------

The PyAutoGalaxy documentation can be found at our `readthedocs  <https://pyautogalaxy.readthedocs.io/en/master>`_,
including instructions on `installation <https://pyautogalaxy.readthedocs.io/en/master/installation.html>`_.

Contributing
------------

If you have any suggestions or would like to contribute please get in touch.

Papers
------

A list of published articles using **PyAutoGalaxy** can be found
`here <https://pyautogalaxy.readthedocs.io/en/master/papers.html>`_ .

Credits
-------

**Developers**:

`James Nightingale <https://github.com/Jammy2211>`_ - Lead developer & PyAutoGalaxy guru.

`Richard Hayes <https://github.com/rhayes777>`_ - Lead developer &
`PyAutoFit <https://github.com/rhayes777/PyAutoFit>`_ guru.

`Ashley Kelly <https://github.com/AshKelly>`_ - Developer of `pyquad <https://github.com/AshKelly/pyquad>`_ for fast
numerical integration.

`Amy Etherington <https://github.com/amyetherington>`_ - Mass profile calcluation methods.

`Xiaoyue Cao <https://github.com/caoxiaoyue>`_ - Analytic Ellipitcal Power-Law Calculations.

Qiuhan He  - NFW Profile Calculations.

`Nan Li <https://github.com/linan7788626>`_ - Docker integration & support.

**Code Donors**:

Mattia Negrello - Visibility models in the uv-plane via direct Fourier transforms.

`Andrea Enia <https://github.com/AndreaEnia>`_ - Voronoi source-plane plotting tools.

`Aristeidis Amvrosiadis <https://github.com/Sketos>`_ - ALMA imaging data loading.
