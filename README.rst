PyAutoGalaxy
==========

The study of a galaxy's light, structure and dynamics is at the heart of modern day Astrophysical research.
**PyAutoGalaxy** makes it simple to model galaxies, like this one:

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
        search=af.DynestyStatic()
        )

    # We pass the imaging data and mask to the phase, thereby fitting it with the galaxy model & plot the resulting fit.
    result = phase.run(data=imaging, mask=mask)
    ag.plot.FitImaging.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

Getting Started
---------------

Please contact us via email or on our SLACK channel if you are interested in using **PyAutoGalaxy**, as project
is still a work in progress whilst we focus n **PyAutoFit** and **PyAutoLens**.

Slack
-----

We're building a **PyAutoGalaxy** community on Slack, so you should contact us on our
`Slack channel <https://pyautogalaxy.slack.com/>`_ before getting started. Here, I will give you the latest updates on
the software & discuss how best to use **PyAutoGalaxy** for your science case.

Unfortunately, Slack is invitation-only, so first send me an `email <https://github.com/Jammy2211>`_ requesting an
invite.