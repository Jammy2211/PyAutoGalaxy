from typing import Tuple, Optional, Union

import autofit as af
import autogalaxy as ag


def mass_from(mass, mass_result, unfix_mass_centre: bool = False) -> af.Model:
    """
    Returns an updated mass `Model` whose priors are initialized from a previous results in a pipeline.

    It includes an option to unfix the input `mass_centre` used previously (e.g. in the SLaM SOURCE PIPELINE), such
    that if the `mass_centre` were fixed (e.g. to (0.0", 0.0")) it becomes a free parameter.

    This function generically links any `MassProfile` to any `MassProfile`, pairing parameters which share the
    same path.

    Parameters
    ----------
    mass
        The mass profile about to be fitted, whose priors are updated based on the previous results.
    mass_result
        The mass profile inferred as a result of the previous pipeline, whose priors are used to update the
        input mass.
    unfix_mass_centre
        If the `mass_centre` was fixed to an input value in a previous pipeline, then `True` will unfix it and make it
        free parameters that are fitted for.

    Returns
    -------
    af.Model(mp.MassProfile)
        The total mass profile whose priors are initialized from a previous result.
    """

    mass.take_attributes(source=mass_result)

    if unfix_mass_centre and isinstance(mass.centre, tuple):

        centre_tuple = mass.centre

        mass.centre = af.Model(mass.cls).centre

        mass.centre.centre_0 = af.GaussianPrior(mean=centre_tuple[0], sigma=0.05)
        mass.centre.centre_1 = af.GaussianPrior(mean=centre_tuple[1], sigma=0.05)

    return mass


def source_custom_model_from(result: af.Result, source_is_model: bool = False) -> af.Model:
    """
    Setup the source model using the previous pipeline's source result.

    The source light model is not specified by the MASS PIPELINE and the previous SOURCE PIPELINE is used to
    determine whether the source model is parametric or an inversion.

    The source can be returned as an `instance` or `model`, depending on the optional input. The default behaviour is
    to return parametric sources as a model (give they must be updated to properly compute a new mass model) and
    return inversions as an instance (as they have sufficient flexibility to typically not required updating).

    Parameters
    ----------
    result
        The result of the previous source pipeline.
    source_is_model
        If `True` the source is returned as a *model* where the parameters are fitted for using priors of the
        search result it is loaded from. If `False`, it is an instance of that search's result.

    Returns
    -------
    af.Model(ag.Galaxy)
        The source galaxy with its light profile(s) and pixelization, set up as a model or instance according to
        the input `source_is_model`.
    """

    redshift = result.instance.galaxies.source.redshift

    if not hasattr(result.instance.galaxies.source, "pixelization"):

        if source_is_model:

            return af.Model(
                ag.Galaxy,
                redshift=redshift,
                bulge=result.model.galaxies.source.bulge,
                disk=result.model.galaxies.source.disk,
            )

        return af.Model(
            ag.Galaxy,
            redshift=redshift,
            bulge=result.instance.galaxies.source.bulge,
            disk=result.instance.galaxies.source.disk,
        )

    if hasattr(result, "adapt"):

        if source_is_model:

            pixelization = af.Model(
                ag.Pixelization,
                mesh=result.adapt.instance.galaxies.source.pixelization.mesh,
                regularization=result.adapt.model.galaxies.source.pixelization.regularization,
            )

            return af.Model(
                ag.Galaxy,
                redshift=redshift,
                pixelization=pixelization,
            )

        pixelization = af.Model(
            ag.Pixelization,
            mesh=result.adapt.instance.galaxies.source.pixelization.mesh,
            regularization=result.adapt.instance.galaxies.source.pixelization.regularization,
        )

        return af.Model(
            ag.Galaxy,
            redshift=redshift,
            pixelization=pixelization,
        )

    else:

        if source_is_model:

            pixelization = af.Model(
                ag.Pixelization,
                mesh=result.instance.galaxies.source.pixelization.mesh,
                regularization=result.model.galaxies.source.pixelization.regularization,
            )

            return af.Model(
                ag.Galaxy,
                redshift=redshift,
                pixelization=pixelization,
            )

        pixelization = af.Model(
            ag.Pixelization,
            mesh=result.instance.galaxies.source.pixelization.mesh,
            regularization=result.instance.galaxies.source.pixelization.regularization,
        )

        return af.Model(
            ag.Galaxy,
            redshift=redshift,
            pixelization=pixelization,
        )


def source_from(
    result: af.Result,
) -> af.Model:
    """
    Setup the source model for a MASS PIPELINE using the previous SOURCE PIPELINE results.

    The source light model is not specified by the  MASS PIPELINE and the previous SOURCE PIPELINE is used to
    determine whether the source model is parametric or an inversion.

    The source is returned as a model if it is parametric (given its parameters must be fitted for to properly compute
    a new mass model) whereas inversions are returned as an instance (as they have sufficient flexibility to not
    require updating). This behaviour can be customized in SLaM pipelines by replacing this method with the
    `source_from` method.

    Parameters
    ----------
    result
        The result of the previous source pipeline.

    Returns
    -------
    af.Model(ag.Galaxy)
        The source galaxy with its light profile(s) and pixelization, set up as a model or instance according to
        the input `result`.
    """

    if hasattr(result.instance.galaxies.source, "pixelization"):
        if result.instance.galaxies.source.pixelization is not None:
            return source_custom_model_from(result=result, source_is_model=False)
    return source_custom_model_from(result=result, source_is_model=True)