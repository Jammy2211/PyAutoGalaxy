from autofit.database.model.fit import Fit
import autogalaxy as ag

from typing import Optional

from functools import partial


def plane_gen_from(aggregator):
    """
    Returns a generator of `Plane` objects from an input aggregator, which generates a list of the `Plane` objects
    for every set of results loaded in the aggregator.

    This is performed by mapping the `plane_from_agg_obj` with the aggregator, which sets up each plane using only
    generators ensuring that manipulating the planes of large sets of results is done in a memory efficient way.

    Parameters
    ----------
    aggregator : af.Aggregator
        A PyAutoFit aggregator object containing the results of PyAutoGalaxy model-fits."""
    return aggregator.map(func=plane_via_database_from)


def plane_via_database_from(fit: Fit):
    """
    Returns a `Plane` object from an aggregator's `SearchOutput` class, which we call an 'agg_obj' to describe that
     it acts as the aggregator object for one result in the `Aggregator`. This uses the aggregator's generator outputs
     such that the function can use the `Aggregator`'s map function to to create a `Plane` generator.

     The `Plane` is created following the same method as the PyAutoGalaxy `Search` classes using an instance of the
     maximum log likelihood model's galaxies. These galaxies have their hyper-images added (if they were used in the
     fit) and passed into a Plane object.

    Parameters
    ----------
    fit : af.SearchOutput
        A PyAutoFit aggregator's SearchOutput object containing the generators of the results of PyAutoGalaxy model-fits.
    """

    galaxies = fit.instance.galaxies

    hyper_model_image = fit.value(name="hyper_model_image")
    hyper_galaxy_image_path_dict = fit.value(name="hyper_galaxy_image_path_dict")

    if hyper_galaxy_image_path_dict is not None:

        for (galaxy_path, galaxy) in fit.instance.path_instance_tuples_for_class(
            ag.Galaxy
        ):
            if galaxy_path in hyper_galaxy_image_path_dict:
                galaxy.hyper_model_image = hyper_model_image
                galaxy.hyper_galaxy_image = hyper_galaxy_image_path_dict[galaxy_path]

    return ag.Plane(galaxies=galaxies)


def imaging_gen_from(aggregator, settings_imaging: Optional[ag.SettingsImaging] = None):
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

    func = partial(imaging_via_database_from, settings_imaging=settings_imaging)

    return aggregator.map(func=func)


def imaging_via_database_from(
    fit: Fit, settings_imaging: Optional[ag.SettingsImaging] = None
):
    """
    Returns a `Imaging` object from an aggregator's `SearchOutput` class, which we call an 'agg_obj' to describe
    that it acts as the aggregator object for one result in the `Aggregator`. This uses the aggregator's generator
    outputs such that the function can use the `Aggregator`'s map function to to create a `Imaging` generator.

     The `Imaging` is created following the same method as the PyAutoGalaxy `Search` classes, including using the
    `SettingsImaging` instance output by the Search to load inputs of the `Imaging` (e.g. psf_shape_2d).

    Parameters
    ----------
    fit : af.SearchOutput
        A PyAutoFit aggregator's SearchOutput object containing the generators of the results of PyAutoGalaxy model-fits.
    """

    data = fit.value(name="data")
    noise_map = fit.value(name="noise_map")
    psf = fit.value(name="psf")
    settings_imaging = settings_imaging or fit.value(name="settings_dataset")

    imaging = ag.Imaging(
        image=data,
        noise_map=noise_map,
        psf=psf,
        settings=settings_imaging,
        setup_convolver=True,
    )

    imaging.apply_settings(settings=settings_imaging)

    return imaging


def fit_imaging_gen_from(
    aggregator,
    settings_imaging: Optional[ag.SettingsImaging] = None,
    settings_pixelization: Optional[ag.SettingsPixelization] = None,
    settings_inversion: Optional[ag.SettingsInversion] = None,
):
    """
    Returns a generator of `FitImaging` objects from an input aggregator, which generates a list of the
    `FitImaging` objects for every set of results loaded in the aggregator.

    This is performed by mapping the `fit_imaging_from_agg_obj` with the aggregator, which sets up each fit using
    only generators ensuring that manipulating the fits of large sets of results is done in a memory efficient way.

    Parameters
    ----------
    aggregator : af.Aggregator
        A PyAutoFit aggregator object containing the results of PyAutoGalaxy model-fits."""

    func = partial(
        fit_imaging_via_database_from,
        settings_imaging=settings_imaging,
        settings_pixelization=settings_pixelization,
        settings_inversion=settings_inversion,
    )

    return aggregator.map(func=func)


def fit_imaging_via_database_from(
    fit: Fit,
    settings_imaging: Optional[ag.SettingsImaging] = None,
    settings_pixelization: Optional[ag.SettingsPixelization] = None,
    settings_inversion: Optional[ag.SettingsInversion] = None,
):
    """
    Returns a `FitImaging` object from an aggregator's `SearchOutput` class, which we call an 'agg_obj' to describe
     that it acts as the aggregator object for one result in the `Aggregator`. This uses the aggregator's generator
     outputs such that the function can use the `Aggregator`'s map function to to create a `FitImaging` generator.

     The `FitImaging` is created following the same method as the PyAutoGalaxy `Search` classes.

    Parameters
    ----------
    fit : af.SearchOutput
        A PyAutoFit aggregator's SearchOutput object containing the generators of the results of PyAutoGalaxy model-fits.
    """

    imaging = imaging_via_database_from(fit=fit, settings_imaging=settings_imaging)
    plane = plane_via_database_from(fit=fit)

    settings_pixelization = settings_pixelization or fit.value(
        name="settings_pixelization"
    )
    settings_inversion = settings_inversion or fit.value(name="settings_inversion")

    return ag.FitImaging(
        imaging=imaging,
        plane=plane,
        settings_pixelization=settings_pixelization,
        settings_inversion=settings_inversion,
    )


def interferometer_gen_from(
    aggregator,
    real_space_mask: Optional[ag.Mask2D] = None,
    settings_interferometer: Optional[ag.SettingsInterferometer] = None,
):
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

    func = partial(
        interferometer_via_database_from,
        real_space_mask=real_space_mask,
        settings_interferometer=settings_interferometer,
    )

    return aggregator.map(func=func)


def interferometer_via_database_from(
    fit: Fit,
    real_space_mask: Optional[ag.Mask2D] = None,
    settings_interferometer: Optional[ag.SettingsInterferometer] = None,
):
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
    fit : af.SearchOutput
        A PyAutoFit aggregator's SearchOutput object containing the generators of the results of PyAutoGalaxy
        model-fits.
    """

    data = fit.value(name="data")
    noise_map = fit.value(name="noise_map")
    uv_wavelengths = fit.value(name="uv_wavelengths")
    real_space_mask = real_space_mask or fit.value(name="real_space_mask")
    settings_interferometer = settings_interferometer or fit.value(
        name="settings_dataset"
    )

    interferometer = ag.Interferometer(
        visibilities=data,
        noise_map=noise_map,
        uv_wavelengths=uv_wavelengths,
        real_space_mask=real_space_mask,
    )

    interferometer = interferometer.apply_settings(settings=settings_interferometer)

    return interferometer


def fit_interferometer_gen_from(
    aggregator,
    real_space_mask: Optional[ag.Mask2D] = None,
    settings_interferometer: Optional[ag.SettingsInterferometer] = None,
    settings_pixelization: Optional[ag.SettingsPixelization] = None,
    settings_inversion: Optional[ag.SettingsInversion] = None,
):
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

    func = partial(
        fit_interferometer_via_database_from,
        real_space_mask=real_space_mask,
        settings_interferometer=settings_interferometer,
        settings_pixelization=settings_pixelization,
        settings_inversion=settings_inversion,
    )

    return aggregator.map(func=func)


def fit_interferometer_via_database_from(
    fit: Fit,
    real_space_mask: Optional[ag.Mask2D] = None,
    settings_interferometer: Optional[ag.SettingsInterferometer] = None,
    settings_pixelization: Optional[ag.SettingsPixelization] = None,
    settings_inversion: Optional[ag.SettingsInversion] = None,
):
    """
    Returns a generator of `FitInterferometer` objects from an input aggregator, which generates a list of the
    `FitInterferometer` objects for every set of results loaded in the aggregator.

    This is performed by mapping the `fit_interferometer_from_agg_obj` with the aggregator, which sets up each fit
    using only generators ensuring that manipulating the fits of large sets of results is done in a memory efficient
    way.

    Parameters
    ----------
    aggregator : af.Aggregator
        A PyAutoFit aggregator object containing the results of PyAutoGalaxy model-fits.
    """

    settings_pixelization = settings_pixelization or fit.value(
        name="settings_pixelization"
    )
    settings_inversion = settings_inversion or fit.value(name="settings_inversion")

    interferometer = interferometer_via_database_from(
        fit=fit,
        real_space_mask=real_space_mask,
        settings_interferometer=settings_interferometer,
    )
    plane = plane_via_database_from(fit=fit)

    return ag.FitInterferometer(
        interferometer=interferometer,
        plane=plane,
        settings_pixelization=settings_pixelization,
        settings_inversion=settings_inversion,
    )
