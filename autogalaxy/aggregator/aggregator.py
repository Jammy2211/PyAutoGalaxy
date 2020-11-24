import autogalaxy as ag


def plane_generator_from_aggregator(aggregator):
    """
    Returns a generator of `Plane` objects from an input aggregator, which generates a list of the `Plane` objects
    for every set of results loaded in the aggregator.

    This is performed by mapping the *plane_from_agg_obj* with the aggregator, which sets up each plane using only
    generators ensuring that manipulating the planes of large sets of results is done in a memory efficient way.

    Parameters
    ----------
    aggregator : af.Aggregator
        A PyAutoFit aggregator object containing the results of PyAutoGalaxy model-fits."""
    return aggregator.map(func=plane_from_agg_obj)


def plane_from_agg_obj(agg_obj):
    """
    Returns a `Plane` object from an aggregator's *PhaseOutput* class, which we call an 'agg_obj' to describe that
     it acts as the aggregator object for one result in the *Aggregator*. This uses the aggregator's generator outputs
     such that the function can use the *Aggregator*'s map function to to create a `Plane` generator.

     The `Plane` is created following the same method as the PyAutoGalaxy `Phase` classes using an instance of the
     maximum log likelihood model's galaxies. These galaxies have their hyper-images added (if they were used in the
     fit) and passed into a Plane object.

    Parameters
    ----------
    agg_obj : af.PhaseOutput
        A PyAutoFit aggregator's PhaseOutput object containing the generators of the results of PyAutoGalaxy model-fits.
    """

    samples = agg_obj.samples
    attributes = agg_obj.attributes
    max_log_likelihood_instance = samples.max_log_likelihood_instance
    galaxies = max_log_likelihood_instance.galaxies

    if attributes.hyper_galaxy_image_path_dict is not None:

        for (
            galaxy_path,
            galaxy,
        ) in max_log_likelihood_instance.path_instance_tuples_for_class(ag.Galaxy):
            if galaxy_path in attributes.hyper_galaxy_image_path_dict:
                galaxy.hyper_model_image = attributes.hyper_model_image
                galaxy.hyper_galaxy_image = attributes.hyper_galaxy_image_path_dict[
                    galaxy_path
                ]

    return ag.Plane(galaxies=galaxies)


def masked_imaging_generator_from_aggregator(aggregator):
    """
    Returns a generator of `MaskImaging` objects from an input aggregator, which generates a list of the
    `MaskImaging` objects for every set of results loaded in the aggregator.

    This is performed by mapping the *masked_imaging_from_agg_obj* with the aggregator, which sets up each masked
    imaging using only generators ensuring that manipulating the masked imaging of large sets of results is done in a
    memory efficient way.

    Parameters
    ----------
    aggregator : af.Aggregator
        A PyAutoFit aggregator object containing the results of PyAutoGalaxy model-fits."""
    return aggregator.map(func=masked_imaging_from_agg_obj)


def masked_imaging_from_agg_obj(agg_obj):
    """
    Returns a `MaskImaging` object from an aggregator's *PhaseOutput* class, which we call an 'agg_obj' to describe
     that it acts as the aggregator object for one result in the *Aggregator*. This uses the aggregator's generator
     outputs such that the function can use the *Aggregator*'s map function to to create a `MaskImaging` generator.

     The `MaskImaging` is created following the same method as the PyAutoGalaxy `Phase` classes, including using the
     *SettingsMaskedImaging* instance output by the phase to load inputs of the `MaskImaging` (e.g. psf_shape_2d).

    Parameters
    ----------
    agg_obj : af.PhaseOutput
        A PyAutoFit aggregator's PhaseOutput object containing the generators of the results of PyAutoGalaxy model-fits.
    """

    return ag.MaskedImaging(
        imaging=agg_obj.dataset,
        mask=agg_obj.mask,
        settings=agg_obj.settings.settings_masked_imaging,
    )


def fit_imaging_generator_from_aggregator(aggregator):
    """
    Returns a generator of `FitImaging` objects from an input aggregator, which generates a list of the
    `FitImaging` objects for every set of results loaded in the aggregator.

    This is performed by mapping the *fit_imaging_from_agg_obj* with the aggregator, which sets up each fit using
    only generators ensuring that manipulating the fits of large sets of results is done in a memory efficient way.

    Parameters
    ----------
    aggregator : af.Aggregator
        A PyAutoFit aggregator object containing the results of PyAutoGalaxy model-fits."""
    return aggregator.map(func=fit_imaging_from_agg_obj)


def fit_imaging_from_agg_obj(agg_obj):
    """
    Returns a `FitImaging` object from an aggregator's *PhaseOutput* class, which we call an 'agg_obj' to describe
     that it acts as the aggregator object for one result in the *Aggregator*. This uses the aggregator's generator
     outputs such that the function can use the *Aggregator*'s map function to to create a `FitImaging` generator.

     The `FitImaging` is created following the same method as the PyAutoGalaxy `Phase` classes.

    Parameters
    ----------
    agg_obj : af.PhaseOutput
        A PyAutoFit aggregator's PhaseOutput object containing the generators of the results of PyAutoGalaxy model-fits.
    """

    masked_imaging = masked_imaging_from_agg_obj(agg_obj=agg_obj)
    plane = plane_from_agg_obj(agg_obj=agg_obj)

    return ag.FitImaging(
        masked_imaging=masked_imaging,
        plane=plane,
        settings_pixelization=agg_obj.settings.settings_pixelization,
        settings_inversion=agg_obj.settings.settings_inversion,
    )


def masked_interferometer_generator_from_aggregator(aggregator):
    """
    Returns a generator of *MaskedInterferometer* objects from an input aggregator, which generates a list of the
    *MaskedInterferometer* objects for every set of results loaded in the aggregator.

    This is performed by mapping the *masked_interferometer_from_agg_obj* with the aggregator, which sets up each masked
    interferometer object using only generators ensuring that manipulating the masked interferometer objects of large
    sets of results is done in a memory efficient  way.

    Parameters
    ----------
    aggregator : af.Aggregator
        A PyAutoFit aggregator object containing the results of PyAutoGalaxy model-fits."""
    return aggregator.map(func=masked_interferometer_from_agg_obj)


def masked_interferometer_from_agg_obj(agg_obj):
    """
    Returns a *MaskedInterferometer* object from an aggregator's *PhaseOutput* class, which we call an 'agg_obj' to
    describe that it acts as the aggregator object for one result in the *Aggregator*. This uses the aggregator's
    generator outputs such that the function can use the *Aggregator*'s map function to to create a
    *MaskedInterferometer* generator.

    The *MaskedInterferometer* is created following the same method as the PyAutoGalaxy `Phase` classes, including
    using the *SettingsMaskedInterferometer* instance output by the phase to load inputs of the *MaskedInterferometer*
    (e.g. psf_shape_2d).

    Parameters
    ----------
    agg_obj : af.PhaseOutput
        A PyAutoFit aggregator's PhaseOutput object containing the generators of the results of PyAutoGalaxy model-fits.
    """

    return ag.MaskedInterferometer(
        interferometer=agg_obj.dataset,
        visibilities_mask=agg_obj.mask,
        real_space_mask=agg_obj.attributes.real_space_mask,
        settings=agg_obj.settings.settings_masked_interferometer,
    )


def fit_interferometer_generator_from_aggregator(aggregator):
    """
    Returns a *FitInterferometer* object from an aggregator's *PhaseOutput* class, which we call an 'agg_obj' to
    describe that it acts as the aggregator object for one result in the *Aggregator*. This uses the aggregator's
    generator outputs such that the function can use the *Aggregator*'s map function to to create a *FitInterferometer*
    generator.

    The *FitInterferometer* is created following the same method as the PyAutoGalaxy `Phase` classes.

    Parameters
    ----------
    agg_obj : af.PhaseOutput
        A PyAutoFit aggregator's PhaseOutput object containing the generators of the results of PyAutoGalaxy model-fits.
    """
    return aggregator.map(func=fit_interferometer_from_agg_obj)


def fit_interferometer_from_agg_obj(agg_obj):
    """
    Returns a generator of *FitInterferometer* objects from an input aggregator, which generates a list of the
    *FitInterferometer* objects for every set of results loaded in the aggregator.

    This is performed by mapping the *fit_interferometer_from_agg_obj* with the aggregator, which sets up each fit
    using only generators ensuring that manipulating the fits of large sets of results is done in a memory efficient
    way.

    Parameters
    ----------
    aggregator : af.Aggregator
        A PyAutoFit aggregator object containing the results of PyAutoGalaxy model-fits."""
    masked_interferometer = masked_interferometer_from_agg_obj(agg_obj=agg_obj)
    plane = plane_from_agg_obj(agg_obj=agg_obj)

    return ag.FitInterferometer(
        masked_interferometer=masked_interferometer,
        plane=plane,
        settings_pixelization=agg_obj.settings.settings_pixelization,
        settings_inversion=agg_obj.settings.settings_inversion,
    )
