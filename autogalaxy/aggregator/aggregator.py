import autogalaxy as ag
from autofit import exc


def plane_generator_from_aggregator(aggregator):
    return aggregator.map(func=plane_from_agg_obj)


def plane_from_agg_obj(agg_obj):

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
    return aggregator.map(func=masked_imaging_from_agg_obj)


def masked_imaging_from_agg_obj(agg_obj):

    return ag.MaskedImaging(
        imaging=agg_obj.dataset,
        mask=agg_obj.mask,
        psf_shape_2d=agg_obj.meta_dataset.psf_shape_2d,
        inversion_pixel_limit=agg_obj.meta_dataset.inversion_pixel_limit,
    )


def fit_imaging_generator_from_aggregator(aggregator):
    return aggregator.map(func=fit_imaging_from_agg_obj)


def fit_imaging_from_agg_obj(agg_obj):

    masked_imaging = masked_imaging_from_agg_obj(agg_obj=agg_obj)
    plane = plane_from_agg_obj(agg_obj=agg_obj)

    return ag.FitImaging(masked_imaging=masked_imaging, plane=plane)


def masked_interferometer_generator_from_aggregator(aggregator):
    return aggregator.map(func=masked_interferometer_from_agg_obj)


def masked_interferometer_from_agg_obj(agg_obj):

    return ag.MaskedInterferometer(
        interferometer=agg_obj.dataset,
        visibilities_mask=agg_obj.mask,
        real_space_mask=agg_obj.meta_dataset.real_space_mask,
        transformer_class=agg_obj.meta_dataset.transformer_class,
        primary_beam_shape_2d=agg_obj.meta_dataset.primary_beam_shape_2d,
        inversion_pixel_limit=agg_obj.meta_dataset.inversion_pixel_limit,
    )


def fit_interferometer_generator_from_aggregator(aggregator):
    return aggregator.map(func=fit_interferometer_from_agg_obj)


def fit_interferometer_from_agg_obj(agg_obj):

    masked_interferometer = masked_interferometer_from_agg_obj(agg_obj=agg_obj)
    plane = plane_from_agg_obj(agg_obj=agg_obj)

    return ag.FitInterferometer(
        masked_interferometer=masked_interferometer, plane=plane
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
