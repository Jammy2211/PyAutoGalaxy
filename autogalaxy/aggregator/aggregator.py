import autogalaxy as ag
from autofit import exc


def plane_generator_from_aggregator(aggregator):
    """Compute a generator of *Plane* objects from an input aggregator, which generates a list of the *Plane* objects 
    for every set of results loaded in the aggregator.

    This is performed by mapping the *plane_from_agg_obj* with the aggregator, which sets up each plane using only
    generators ensuring that manipulating the planes of large sets of results is done in a memory efficient way.

    Parameters
    ----------
    aggregator : af.Aggregator
        A PyAutoFit aggregator object containing the results of PyAutoGalaxy model-fits."""
    return aggregator.map(func=plane_from_agg_obj)


def plane_from_agg_obj(agg_obj):
    """Compute a *Plane* object from an aggregator's *PhaseOutput* class, which we call an 'agg_obj' to describe that
     it acts as the aggregator object for one result in the *Aggregator*. This uses the aggregator's generator outputs
     such that the function can use the *Aggregator*'s map function to to create a *Plane* generator.

     The *Plane* is created following the same method as the PyAutoGalaxy *Phase* classes using an instance of the
     maximum log likelihood model's galaxies. These galaxies have their hyper-images added (if they were used in the
     fit) and passed into a Plane object.

    Parameters
    ----------
    agg_obj : af.PhaseOutput
        A PyAutoFit aggregator's PhaseOutput object containing the generators of the results of PyAutoGalaxy model-fits.
    """

    samples = agg_obj.samples
    phase_attributes = agg_obj.phase_attributes
    max_log_likelihood_instance = samples.max_log_likelihood_instance
    galaxies = max_log_likelihood_instance.galaxies

    if phase_attributes.hyper_galaxy_image_path_dict is not None:

        for (
            galaxy_path,
            galaxy,
        ) in max_log_likelihood_instance.path_instance_tuples_for_class(ag.Galaxy):
            if galaxy_path in phase_attributes.hyper_galaxy_image_path_dict:
                galaxy.hyper_model_image = phase_attributes.hyper_model_image
                galaxy.hyper_galaxy_image = phase_attributes.hyper_galaxy_image_path_dict[
                    galaxy_path
                ]

    return ag.Plane(galaxies=galaxies)


def masked_imaging_generator_from_aggregator(aggregator):
    """Compute a generator of *MaskedImaging* objects from an input aggregator, which generates a list of the 
    *MaskedImaging* objects for every set of results loaded in the aggregator.

    This is performed by mapping the *masked_imaging_from_agg_obj* with the aggregator, which sets up each masked
    imaging using only generators ensuring that manipulating the masked imaging of large sets of results is done in a
    memory efficient way.

    Parameters
    ----------
    aggregator : af.Aggregator
        A PyAutoFit aggregator object containing the results of PyAutoGalaxy model-fits."""
    return aggregator.map(func=masked_imaging_from_agg_obj)


def masked_imaging_from_agg_obj(agg_obj):
    """Compute a *MaskedImaging* object from an aggregator's *PhaseOutput* class, which we call an 'agg_obj' to describe 
     that it acts as the aggregator object for one result in the *Aggregator*. This uses the aggregator's generator 
     outputs such that the function can use the *Aggregator*'s map function to to create a *MaskedImaging* generator.

     The *MaskedImaging* is created following the same method as the PyAutoGalaxy *Phase* classes, including using the
     *SettingsMaskedImaging* instance output by the phase to load inputs of the *MaskedImaging* (e.g. psf_shape_2d).

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
    """Compute a generator of *FitImaging* objects from an input aggregator, which generates a list of the 
    *FitImaging* objects for every set of results loaded in the aggregator.

    This is performed by mapping the *fit_imaging_from_agg_obj* with the aggregator, which sets up each fit using
    only generators ensuring that manipulating the fits of large sets of results is done in a memory efficient way.

    Parameters
    ----------
    aggregator : af.Aggregator
        A PyAutoFit aggregator object containing the results of PyAutoGalaxy model-fits."""
    return aggregator.map(func=fit_imaging_from_agg_obj)


def fit_imaging_from_agg_obj(agg_obj):
    """Compute a *FitImaging* object from an aggregator's *PhaseOutput* class, which we call an 'agg_obj' to describe 
     that it acts as the aggregator object for one result in the *Aggregator*. This uses the aggregator's generator 
     outputs such that the function can use the *Aggregator*'s map function to to create a *FitImaging* generator.

     The *FitImaging* is created following the same method as the PyAutoGalaxy *Phase* classes. 

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
    """Compute a generator of *MaskedInterferometer* objects from an input aggregator, which generates a list of the 
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
    """Compute a *MaskedInterferometer* object from an aggregator's *PhaseOutput* class, which we call an 'agg_obj' to
    describe that it acts as the aggregator object for one result in the *Aggregator*. This uses the aggregator's
    generator outputs such that the function can use the *Aggregator*'s map function to to create a
    *MaskedInterferometer* generator.

    The *MaskedInterferometer* is created following the same method as the PyAutoGalaxy *Phase* classes, including
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
        real_space_mask=agg_obj.phase_attributes.real_space_mask,
        settings=agg_obj.settings.settings_masked_interferometer,
    )


def fit_interferometer_generator_from_aggregator(aggregator):
    """Compute a *FitInterferometer* object from an aggregator's *PhaseOutput* class, which we call an 'agg_obj' to 
    describe that it acts as the aggregator object for one result in the *Aggregator*. This uses the aggregator's 
    generator outputs such that the function can use the *Aggregator*'s map function to to create a *FitInterferometer* 
    generator.

    The *FitInterferometer* is created following the same method as the PyAutoGalaxy *Phase* classes. 

    Parameters
    ----------
    agg_obj : af.PhaseOutput
        A PyAutoFit aggregator's PhaseOutput object containing the generators of the results of PyAutoGalaxy model-fits.
    """
    return aggregator.map(func=fit_interferometer_from_agg_obj)


def fit_interferometer_from_agg_obj(agg_obj):
    """Compute a generator of *FitInterferometer* objects from an input aggregator, which generates a list of the 
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


def grid_search_result_as_array(aggregator, use_max_log_likelihoods=True):

    grid_search_result_gen = aggregator.values("grid_search_result")

    grid_search_results = list(filter(None, list(grid_search_result_gen)))

    if len(grid_search_results) != 1:
        raise exc.AggregatorException(
            "There is more than one grid search result in the aggregator - please filter the"
            "aggregator."
        )

    return grid_search_result_as_array_from_grid_search_result(
        grid_search_result=grid_search_results[0],
        use_max_log_likelihoods=use_max_log_likelihoods,
    )


def grid_search_result_as_array_from_grid_search_result(
    grid_search_result, use_max_log_likelihoods=True
):

    if grid_search_result.no_dimensions != 2:
        raise exc.AggregatorException(
            "The GridSearchResult is not dimensions 2, meaning a 2D array cannot be made."
        )

    if use_max_log_likelihoods:
        values = [
            value
            for values in grid_search_result.max_log_likelihood_values
            for value in values
        ]
    else:
        values = [
            value
            for values in grid_search_result.log_evidence_values
            for value in values
        ]

    return ag.Array.manual_yx_and_values(
        y=[centre[0] for centre in grid_search_result.physical_centres_lists],
        x=[centre[1] for centre in grid_search_result.physical_centres_lists],
        values=values,
        pixel_scales=grid_search_result.physical_step_sizes,
        shape_2d=grid_search_result.shape,
    )
