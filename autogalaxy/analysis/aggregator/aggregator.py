import autogalaxy as ag


def plane_generator_from_aggregator(aggregator):
    """
    Returns a generator of `Plane` objects from an input aggregator, which generates a list of the `Plane` objects
    for every set of results loaded in the aggregator.

    This is performed by mapping the `plane_from_agg_obj` with the aggregator, which sets up each plane using only
    generators ensuring that manipulating the planes of large sets of results is done in a memory efficient way.

    Parameters
    ----------
    aggregator : af.Aggregator
        A PyAutoFit aggregator object containing the results of PyAutoGalaxy model-fits."""
    return aggregator.map(func=plane_from_agg_obj)


def plane_from_agg_obj(agg_obj):
    """
    Returns a `Plane` object from an aggregator's `SearchOutput` class, which we call an 'agg_obj' to describe that
     it acts as the aggregator object for one result in the `Aggregator`. This uses the aggregator's generator outputs
     such that the function can use the `Aggregator`'s map function to to create a `Plane` generator.

     The `Plane` is created following the same method as the PyAutoGalaxy `Search` classes using an instance of the
     maximum log likelihood model's galaxies. These galaxies have their hyper-images added (if they were used in the
     fit) and passed into a Plane object.

    Parameters
    ----------
    agg_obj : af.SearchOutput
        A PyAutoFit aggregator's SearchOutput object containing the generators of the results of PyAutoGalaxy model-fits.
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


def imaging_generator_from_aggregator(aggregator):
    """
    Returns a generator of `Imaging` objects from an input aggregator, which generates a list of the
    `Imaging` objects for every set of results loaded in the aggregator.

    This is performed by mapping the `imaging_from_agg_obj` with the aggregator, which sets up each
    imaging using only generators ensuring that manipulating the imaging of large sets of results is done in a
    memory efficient way.

    Parameters
    ----------
    aggregator : af.Aggregator
        A PyAutoFit aggregator object containing the results of PyAutoGalaxy model-fits."""
    return aggregator.map(func=imaging_from_agg_obj)


def imaging_from_agg_obj(agg_obj):
    """
    Returns a `Imaging` object from an aggregator's `SearchOutput` class, which we call an 'agg_obj' to describe
     that it acts as the aggregator object for one result in the `Aggregator`. This uses the aggregator's generator
     outputs such that the function can use the `Aggregator`'s map function to to create a `Imaging` generator.

     The `Imaging` is created following the same method as the PyAutoGalaxy `Search` classes, including using the
     `SettingsImaging` instance output by the Search to load inputs of the `Imaging` (e.g. psf_shape_2d).

    Parameters
    ----------
    agg_obj : af.SearchOutput
        A PyAutoFit aggregator's SearchOutput object containing the generators of the results of PyAutoGalaxy model-fits.
    """

    return agg_obj.dataset


def fit_imaging_generator_from_aggregator(aggregator):
    """
    Returns a generator of `FitImaging` objects from an input aggregator, which generates a list of the
    `FitImaging` objects for every set of results loaded in the aggregator.

    This is performed by mapping the `fit_imaging_from_agg_obj` with the aggregator, which sets up each fit using
    only generators ensuring that manipulating the fits of large sets of results is done in a memory efficient way.

    Parameters
    ----------
    aggregator : af.Aggregator
        A PyAutoFit aggregator object containing the results of PyAutoGalaxy model-fits."""
    return aggregator.map(func=fit_imaging_from_agg_obj)


def fit_imaging_from_agg_obj(agg_obj):
    """
    Returns a `FitImaging` object from an aggregator's `SearchOutput` class, which we call an 'agg_obj' to describe
     that it acts as the aggregator object for one result in the `Aggregator`. This uses the aggregator's generator
     outputs such that the function can use the `Aggregator`'s map function to to create a `FitImaging` generator.

     The `FitImaging` is created following the same method as the PyAutoGalaxy `Search` classes.

    Parameters
    ----------
    agg_obj : af.SearchOutput
        A PyAutoFit aggregator's SearchOutput object containing the generators of the results of PyAutoGalaxy model-fits.
    """

    imaging = imaging_from_agg_obj(agg_obj=agg_obj)
    plane = plane_from_agg_obj(agg_obj=agg_obj)

    return ag.FitImaging(
        imaging=imaging,
        plane=plane,
        settings_pixelization=agg_obj.settings_pixelization,
        settings_inversion=agg_obj.settings_inversion,
    )


def interferometer_generator_from_aggregator(aggregator):
    """
    Returns a generator of `Interferometer` objects from an input aggregator, which generates a list of the
    `Interferometer` objects for every set of results loaded in the aggregator.

    This is performed by mapping the `interferometer_from_agg_obj` with the aggregator, which sets up each
    interferometer object using only generators ensuring that manipulating the interferometer objects of large
    sets of results is done in a memory efficient  way.

    Parameters
    ----------
    aggregator : af.Aggregator
        A PyAutoFit aggregator object containing the results of PyAutoGalaxy model-fits."""
    return aggregator.map(func=interferometer_from_agg_obj)


def interferometer_from_agg_obj(agg_obj):
    """
    Returns a `Interferometer` object from an aggregator's `SearchOutput` class, which we call an 'agg_obj' to
    describe that it acts as the aggregator object for one result in the `Aggregator`. This uses the aggregator's
    generator outputs such that the function can use the `Aggregator`'s map function to to create a
    `Interferometer` generator.

    The `Interferometer` is created following the same method as the PyAutoGalaxy `Search` classes, including
    using the `SettingsInterferometer` instance output by the Search to load inputs of the `Interferometer`
    (e.g. psf_shape_2d).

    Parameters
    ----------
    agg_obj : af.SearchOutput
        A PyAutoFit aggregator's SearchOutput object containing the generators of the results of PyAutoGalaxy
        model-fits.
    """

    return agg_obj.dataset


def fit_interferometer_generator_from_aggregator(aggregator):
    """
    Returns a `FitInterferometer` object from an aggregator's `SearchOutput` class, which we call an 'agg_obj' to
    describe that it acts as the aggregator object for one result in the `Aggregator`. This uses the aggregator's
    generator outputs such that the function can use the `Aggregator`'s map function to to create a `FitInterferometer`
    generator.

    The `FitInterferometer` is created following the same method as the PyAutoGalaxy `Search` classes.

    Parameters
    ----------
    agg_obj : af.SearchOutput
        A PyAutoFit aggregator's SearchOutput object containing the generators of the results of PyAutoGalaxy model-fits.
    """
    return aggregator.map(func=fit_interferometer_from_agg_obj)


def fit_interferometer_from_agg_obj(agg_obj):
    """
    Returns a generator of `FitInterferometer` objects from an input aggregator, which generates a list of the
    `FitInterferometer` objects for every set of results loaded in the aggregator.

    This is performed by mapping the `fit_interferometer_from_agg_obj` with the aggregator, which sets up each fit
    using only generators ensuring that manipulating the fits of large sets of results is done in a memory efficient
    way.

    Parameters
    ----------
    aggregator : af.Aggregator
        A PyAutoFit aggregator object containing the results of PyAutoGalaxy model-fits."""
    interferometer = interferometer_from_agg_obj(agg_obj=agg_obj)
    plane = plane_from_agg_obj(agg_obj=agg_obj)

    return ag.FitInterferometer(
        interferometer=interferometer,
        plane=plane,
        settings_pixelization=agg_obj.settings_pixelization,
        settings_inversion=agg_obj.settings_inversion,
    )
