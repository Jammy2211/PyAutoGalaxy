from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import autoarray as aa
import autofit as af

if TYPE_CHECKING:
    from autogalaxy.analysis.result import Result

from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.profiles.mass.abstract.abstract import MassProfile
from autogalaxy.profiles.light_and_mass_profiles import LightMassProfile
from autogalaxy.galaxy.galaxy import Galaxy


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


def source_custom_model_from(result: Result, source_is_model: bool = False) -> af.Model:
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
    af.Model(Galaxy)
        The source galaxy with its light profile(s) and pixelization, set up as a model or instance according to
        the input `source_is_model`.
    """

    redshift = result.instance.galaxies.source.redshift

    if not hasattr(result.instance.galaxies.source, "pixelization"):
        if source_is_model:
            return af.Model(
                Galaxy,
                redshift=redshift,
                bulge=result.model.galaxies.source.bulge,
                disk=result.model.galaxies.source.disk,
            )

        return af.Model(
            Galaxy,
            redshift=redshift,
            bulge=result.instance.galaxies.source.bulge,
            disk=result.instance.galaxies.source.disk,
        )

    if source_is_model:
        pixelization = af.Model(
            aa.Pixelization,
            image_mesh=result.instance.galaxies.source.pixelization.image_mesh,
            mesh=result.instance.galaxies.source.pixelization.mesh,
            regularization=result.model.galaxies.source.pixelization.regularization,
        )

        return af.Model(
            Galaxy,
            redshift=redshift,
            pixelization=pixelization,
        )

    pixelization = af.Model(
        aa.Pixelization,
        image_mesh=result.instance.galaxies.source.pixelization.image_mesh,
        mesh=result.instance.galaxies.source.pixelization.mesh,
        regularization=result.instance.galaxies.source.pixelization.regularization,
    )

    return af.Model(
        Galaxy,
        redshift=redshift,
        pixelization=pixelization,
    )


def source_from(
    result: Result,
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
    af.Model(Galaxy)
        The source galaxy with its light profile(s) and pixelization, set up as a model or instance according to
        the input `result`.
    """

    if hasattr(result.instance.galaxies.source, "pixelization"):
        if result.instance.galaxies.source.pixelization is not None:
            return source_custom_model_from(result=result, source_is_model=False)
    return source_custom_model_from(result=result, source_is_model=True)


def clumps_from(
    result: Result,
    light_as_model: bool = False,
    mass_as_model: bool = False,
    free_centre: bool = False,
) -> af.Collection:
    """
    The clump API models the light and / or mass of additional galaxies surrouding the main galaxy or strong lens
    system.

    This function performs model composition of clumps for fits using the search chaining API. It makes it possible to
    pass the clump parameters from a previous search to a new search, such that the new clumps are either treated
    as an instance or model component.

    This function currently requires that mass profiles are `IsothermalSph` objects and that light profiles are
    `Sersic` objects. This will be generalised in the future.

    Parameters
    ----------
    result
        The result, which includes clumps, of the previous search, which via prior passing are used to create the new
        clump model.
    light_as_model
        If `True`, the clump light is passed as a model component, else it is a fixed instance.
    mass_as_model
        If `True`, the clump mass is passed as a model component, else it is a fixed instance.
    free_centre
        If `True`, the requested mass and/or light model centres are passed as a model, else they are fixed.

    Returns
    -------
    af.Collection
        A collection of clump `Galaxy` objects, where each clump is either an instance or model component.
    """
    # ideal API:

    # clumps = result.instance.clumps.as_model((LightProfile, mp.MassProfile,), fixed="centre", prior_pass=True)

    if mass_as_model:
        clumps = result.instance.clumps.as_model((MassProfile,))

        for clump_index in range(len(result.instance.clumps)):
            if hasattr(result.instance.clumps[clump_index], "mass"):
                clumps[clump_index].mass.centre = result.instance.clumps[
                    clump_index
                ].mass.centre
                clumps[clump_index].mass.einstein_radius = result.model.clumps[
                    clump_index
                ].mass.einstein_radius
                if free_centre:
                    clumps[clump_index].mass.centre = result.model.clumps[
                        clump_index
                    ].mass.centre

    elif light_as_model:
        clumps = result.instance.clumps.as_model((LightProfile,))

        for clump_index in range(len(result.instance.clumps)):
            if clumps[clump_index].light is not None:
                clumps[clump_index].light.centre = result.instance.clumps[
                    clump_index
                ].light.centre
                if free_centre:
                    clumps[clump_index].light.centre = result.model.clumps[
                        clump_index
                    ].light.centre

    else:
        clumps = result.instance.clumps.as_model(())

    return clumps


def mass_light_dark_from(
    lmp_model: af.Model(LightMassProfile),
    result_light_component: af.Model,
) -> Optional[af.Model]:
    """
    Returns an updated version of a `LightMassProfile` model (e.g. a bulge or disk) whose priors are initialized from
    previous results of a `Light` pipeline.

    This function generically links any `LightProfile` to any `LightMassProfile`, pairing parameters which share the
    same path.

    Parameters
    ----------
    lmp_model
        The light and mass profile whoses priors are passed from the LIGHT PIPELINE.
    result_light_component
        The `LightProfile` result of the LIGHT PIPELINE used to pass the priors.

    Returns
    -------
    af.Model(mp.LightMassProfile)
        The light and mass profile whose priors are initialized from a previous result.
    """

    if lmp_model is None:
        return lmp_model

    # TODO : Add support for linear light profiles + Basis

    lmp_model.take_attributes(source=result_light_component)

    return lmp_model
