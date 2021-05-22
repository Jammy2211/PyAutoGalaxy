import numpy as np
from scipy.stats import norm

import autofit as af
from autoarray.inversion import pixelizations as pix, regularization as reg
from autofit.exc import PriorException
from autogalaxy.galaxy import galaxy as g
from autogalaxy.profiles import light_profiles as lp
from autogalaxy.profiles import mass_profiles as mp


def isprior(obj):
    if isinstance(obj, af.Model):
        return True
    return False


def isinstance_or_prior(obj, cls):
    if isinstance(obj, cls):
        return True
    if isinstance(obj, af.Model) and obj.cls == cls:
        return True
    return False


def pixelization_from(model: af.Collection) -> pix.Pixelization:
    """
    For a model containing one or more galaxies, inspect its attributes and return the `pixelization` of a galaxy
    provided one galaxy has a pixelization, otherwise it returns none. There cannot be more than one `Pixelization` in
    a model.
    
    This function expects that the input model is a `Collection` where the first model-component has the
    name `galaxies`, and is itself a `Collection` of `Galaxy` instances. This is the
    standard API for creating a model in PyAutoGalaxy.

    The result of `pixelization_from_model` is used by the preloading to determine whether certain parts of a
    calculation can be cached before the non-linear search begins for efficiency.

    Parameters
    ----------
    model : af.Collection
        Contains the `galaxies` in the model that will be fitted via the non-linear search.

    Returns
    -------
    pix.Pixelization or None:
        The `Pixelization` of a galaxy, provided one galaxy has a `Pixelization`.
    """

    for galaxy in model.galaxies:
        if hasattr(galaxy, "pixelization"):
            if galaxy.pixelization is not None:
                if isinstance(galaxy.pixelization, af.Model):
                    return galaxy.pixelization.cls
                else:
                    return galaxy.pixelization


def has_pixelization_from_model(model: af.Collection):
    """
    For a model containing one or more galaxies, inspect its attributes and return `True` if a galaxy has a
    `Pixelization` otherwise return `False`.

    This function expects that the input model is a `Collection` where the first model-component has the
    name `galaxies`, and is itself a `Collection` of `Galaxy` instances. This is the
    standard API for creating a model in PyAutoGalaxy.

    The result of `has_pixelization_from_model` is used by the preloading to determine whether certain parts of a
    calculation can be cached before the non-linear search begins for efficiency.

    Parameters
    ----------
    model : af.Collection
        Contains the `galaxies` in the model that will be fitted via the non-linear search.

    Returns
    -------
    pix.Pixelization or None:
        The `Pixelization` of a galaxy, provided one galaxy has a `Pixelization`.
    """
    pixelization = pixelization_from(model=model)

    return pixelization is not None


def pixelization_is_model_from(model: af.Collection):
    """
    For a model containing one or more galaxies, inspect its attributes and return `True` if a galaxy has a
    `Pixelization` which is a model-component with free parameters, otherwise return `False`. Therefore, a `False`
    may be returned if a galaxy has a `Pixelization` but it is an `instance` where no parameters are free parameters
    in the non-linear search.

    This function expects that the input model is a `Collection` where the first model-component has the
    name `galaxies`, and is itself a `Collection` of `Galaxy` instances. This is the
    standard API for creating a model in PyAutoGalaxy.

    The result of `pixelization_is_model_from_model` is used by the preloading to determine whether certain parts of a
    calculation can be cached before the non-linear search begins for efficiency.

    Parameters
    ----------
    model : af.Collection
        Contains the `galaxies` in the model that will be fitted via the non-linear search.

    Returns
    -------
    pix.Pixelization or None:
        The `Pixelization` of a galaxy, provided one galaxy has a `Pixelization`.
    """
    if model.galaxies:
        for galaxy in model.galaxies:
            if isprior(galaxy.pixelization):
                return True
    return False


def hyper_model_from(
    setup_hyper, result: af.Result, include_hyper_image_sky: bool = False
) -> af.Collection:
    """
    Make a hyper model from the `Result` of a model-fit, where the hyper-model is the maximum log likelihood instance
    of the inferred model but turns the following hyper components of the model to free parameters:

    1) The `Pixelization` of any `Galaxy` in the model.
    2) The `Regularization` of any `Galaxy` in the model.
    3) Hyper data components like a `HyperImageSky` or `HyperBackgroundNoise` if input into the function.
    4) `HyperGalaxy` components of the `Galaxy`'s in the model, which are used to scale the noise in regions of the
    data which are fit poorly.

    The hyper model is typically used in pipelines to refine and improve an `Inversion` after model-fits that fit the
    `Galaxy` light and mass components.

    Parameters
    ----------
    setup_hyper : SetupHyper
        The setup of the hyper analysis if used (e.g. hyper-galaxy noise scaling).
    result : af.Result
        The result of a previous `Analysis` search whose maximum log likelihood model forms the basis of the hyper model.
    include_hyper_image_sky : hd.HyperImageSky
        This must be true to include the hyper-image sky in the model, even if it is turned on in `setup_hyper`.

    Returns
    -------
    af.Collection
        The hyper model, which has an instance of the input results maximum log likelihood model with certain hyper
        model components now free parameters.
    """

    model = result.instance.as_model((pix.Pixelization, reg.Regularization))

    if setup_hyper is None:
        return None

    if setup_hyper.hyper_galaxy_names is None:
        if not has_pixelization_from_model(model=model):
            if setup_hyper.hypers_all_off:
                return None
            if setup_hyper.hypers_all_except_image_sky_off:
                if not include_hyper_image_sky:
                    return None

    model.hyper_image_sky = setup_hyper.hyper_image_sky
    model.hyper_background_noise = setup_hyper.hyper_background_noise

    if setup_hyper.hyper_galaxy_names is not None:

        for path_galaxy, galaxy in result.path_galaxy_tuples:
            if path_galaxy[-1] in setup_hyper.hyper_galaxy_names:
                if not np.all(result.hyper_galaxy_image_path_dict[path_galaxy] == 0):

                    galaxy = getattr(model.galaxies, path_galaxy[-1])

                    setattr(galaxy, "hyper_galaxy", af.Model(g.HyperGalaxy))

    return model


def hyper_fit(hyper_model: af.Collection, setup_hyper, result: af.Result, analysis):
    """
    Perform a hyper-fit, which extends a model-fit with an additional fit which fixes the non-hyper components of the
    model (e.g., `LightProfile`'s, `MassProfile`) to the `Result`'s maximum likelihood fit. The hyper-fit then treats
    only the hyper-model components as free parameters, which are any of the following model components:

    1) The `Pixelization` of any `Galaxy` in the model.
    2) The `Regularization` of any `Galaxy` in the model.
    3) Hyper data components like a `HyperImageSky` or `HyperBackgroundNoise` if input into the function.
    4) `HyperGalaxy` components of the `Galaxy`'s in the model, which are used to scale the noise in regions of the
    data which are fit poorly.

    The hyper model is typically used in pipelines to refine and improve an `Inversion` after model-fits that fit the
    `Galaxy` light and mass components.

    Parameters
    ----------
    hyper_model : Collection
        The hyper model used by the hyper-fit, which models hyper-components like a `Pixelization` or `HyperGalaxy`'s.
    setup_hyper : SetupHyper
        The setup of the hyper analysis if used (e.g. hyper-galaxy noise scaling).
    result : af.Result
        The result of a previous `Analysis` search whose maximum log likelihood model forms the basis of the hyper model.
    analysis : Analysis
        An analysis class used to fit imaging or interferometer data with a model.

    Returns
    -------
    af.Result
        The result of the hyper model-fit, which has a new attribute `result.hyper` that contains updated parameter
        values for the hyper-model components for passing to later model-fits.
    """

    if hyper_model is None:
        return result

    search = setup_hyper.search_cls(
        path_prefix=result.search.path_prefix_no_unique_tag,
        name=f"{result.search.paths.name}__hyper",
        unique_tag=result.search.paths.unique_tag,
        number_of_cores=result.search.number_of_cores,
        **setup_hyper.search_dict,
    )

    analysis.set_hyper_dataset(result=result)

    hyper_result = search.fit(model=hyper_model, analysis=analysis)

    setattr(result, "hyper", hyper_result)

    return result


def stochastic_model_from(
    result,
    include_lens_light=False,
    include_pixelization=False,
    include_regularization=False,
    subhalo_centre_width=None,
    subhalo_mass_at_200_log_uniform=True,
):
    """
    Make a stochastic model from  the `Result` of a model-fit, where the stochastic model uses the same model
    components as the original model but may switch certain components (e.g. the lens light, source pixelization)
    to free parameters.

    The stochastic model is used to perform a stochastic model-fit, which refits a model but introduces a log
    likelihood cap whereby all model-samples with a likelihood above this cap are rounded down to the value of the cap.

    This `log_likelihood_cap` is determined by sampling ~250 log likeilhood values from the original model's, but where
    each model evaluation uses a different KMeans seed of the pixelization to derive a unique pixelization with which
     to reconstruct the source galaxy (therefore a pixelization which uses the KMeans method, like the
     `VoronoiBrightnessImage` must be used to perform a stochastic fit).

     The cap is computed as the mean of these ~250 values and it is introduced to avoid underestimated errors due
     to artificial likelihood boosts.

    Parameters
    ----------
    result : af.Result
        The result of a previous `Analysis` search whose maximum log likelihood model forms the basis of the hyper model.
    include_lens_light : bool
        If `True` and the model includes any `LightProfile`'s, these are fitted for in the model.
    include_pixelization : bool
        If `True` the `VoronoiBrightnessImage` pixelization in the model is fitted for.
    include_regularization : bool
        If `True` the regularization in the model is fitted for.
    subhalo_centre_width : float
        The `sigma` value of the `GaussianPrior` on the centre of the subhalo, if it is included in the lens model.
    subhalo_mass_at_200_log_uniform : bool
        if `True`, the subhalo mass (if included) does not assume a `GaussianPrior` from the previous fit, but instead
        retains the default `LogUniformPrior`.

    Returns
    -------
    af.Collection
        The stochastic model, which is the same model as the input model but may fit for or fix additional parameters.
    """
    if not hasattr(result.model.galaxies, "lens"):
        raise PriorException(
            "Cannot extend a search with a stochastic search if the lens galaxy `Model` "
            "is not named `lens`. "
        )

    model_classes = [mp.MassProfile]

    if include_lens_light:
        model_classes.append(lp.LightProfile)

    if include_pixelization:
        model_classes.append(pix.Pixelization)

    if include_regularization:
        model_classes.append(reg.Regularization)

    model = result.instance.as_model(model_classes)

    model.galaxies.lens.take_attributes(source=result.model.galaxies.lens)

    if hasattr(model.galaxies, "subhalo"):
        model.galaxies.subhalo.take_attributes(source=result.model.galaxies.subhalo)

        if subhalo_centre_width is not None:
            model.galaxies.subhalo.mass.centre = result.model_absolute(
                a=subhalo_centre_width
            ).galaxies.subhalo.mass.centre

        if subhalo_mass_at_200_log_uniform:
            model.galaxies.subhalo.mass.mass_at_200 = af.LogUniformPrior(
                lower_limit=1e6, upper_limit=1e11
            )

    return model


def stochastic_fit(stochastic_model, search_cls, search_dict, result, analysis):
    """
    Perform a stochastic model-fit, which refits a model but introduces a log likelihood cap whereby all model-samples
    with a likelihood above this cap are rounded down to the value of the cap.

    This `log_likelihood_cap` is determined by sampling ~250 log likelihood values from the original model's maximum
    log likelihood model. However, the pixelization used to reconstruct the source of each model evaluation uses a
    different KMeans seed, such that each reconstruction uses a unique pixel-grid. The model must therefore use a
    pixelization which uses the KMeans method to construct the pixel-grid, for example the `VoronoiBrightnessImage`.

    The cap is computed as the mean of these ~250 values and it is introduced to avoid underestimated errors due
    to artificial likelihood boosts.

    Parameters
    ----------
    setup_hyper : SetupHyper
        The setup of the hyper analysis if used (e.g. hyper-galaxy noise scaling).
    result : af.Result
        The result of a previous `Analysis` search whose maximum log likelihood model forms the basis of the hyper model.
    include_hyper_image_sky : hd.HyperImageSky
        This must be true to include the hyper-image sky in the model, even if it is turned on in `setup_hyper`.

    Returns
    -------
    af.Collection
        The hyper model, which has an instance of the input results maximum log likelihood model with certain hyper
        model components now free parameters.
    """

    mean, sigma = norm.fit(result.stochastic_log_evidences)
    log_likelihood_cap = mean

    name = f"{result.search.paths.name}__stochastic_likelihood_cap_" + "{0:.1f}".format(
        log_likelihood_cap
    )

    search = search_cls(
        path_prefix=result.search.path_prefix_no_unique_tag,
        name=name,
        unique_tag=result.search.paths.unique_tag,
        number_of_cores=result.search.number_of_cores,
        **search_dict,
    )

    search.paths.save_object(
        "stochastic_log_evidences", result.stochastic_log_evidences
    )

    stochastic_result = search.fit(
        model=stochastic_model, analysis=analysis, log_likelihood_cap=log_likelihood_cap
    )

    setattr(result, "stochastic", stochastic_result)

    return result
