from typing import Tuple, Optional, Union

import autofit as af
import autolens as al


def mass__from(mass, mass_result, unfix_mass_centre: bool = False) -> af.Model:
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
    results
        The result of a previous pipeline (e.g. SOURCE LP PIPELINE or SOURCE PIX PIPELINE in SLaM).
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