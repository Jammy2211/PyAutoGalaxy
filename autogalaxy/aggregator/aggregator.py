import autogalaxy as ag

import numpy as np
import os


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
