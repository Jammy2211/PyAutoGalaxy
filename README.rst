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

    """
    Load Imaging data of the strong lens from the dataset folder of the workspace.
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
    model = af.Collection(galaxy=_galaxy_model)

    """
    We define the non-linear search used to fit the model to the data (in this case, Dynesty).
    """
    search = af.DynestyStatic(name="search[example]", nlive=50)
    
    """
    We next set up the `Analysis`, which contains the `log likelihood function` that the
    non-linear search calls to fit the lens model to the data.
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