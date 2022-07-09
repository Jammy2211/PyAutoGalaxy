import autofit as af
from autofit.exc import *
from autoarray.exc import *


class ProfileException(Exception):
    """
    Raises exceptions associated with the `profile` modules and `LightProfile` / `MassProfile` classes.

    For example when a mass-profile deflection angle calculation goes wrong due to numerical issues.
    """

    pass


class GalaxyException(Exception):
    """
    Raises exceptions associated with the `galaxy` module and `Galaxy` class.

    For example if stellar-mass specific calculations are used for a galaxy which does not have a stellar mas
    component.
    """

    pass


class PlaneException(Exception):
    """
    Raises exceptions associated with the `plane` module and `Galaxy` class.

    For example if no galaxies or redshifts are input into a plane, such that the plane does not know its redshift
    relative to other planes.
    """

    pass


class AnalysisException(Exception):
    """
    Raises exceptions associated with the `analysis` modules in the `model` packages and `Analysis` classes.

    For example if the figure of merit of the analysis class's `log_likelihood_function` has changed for a resumed
    run from a previous run.
    """

    pass


class PixelizationException(af.exc.FitException):
    """
    Raises exceptions associated with the `inversion/pixelization` modules and `Pixelization` classes.

    For example if a `Rectangular` pixelization has dimensions below 3x3.

    This exception overwrites `autoarray.exc.PixelizationException` in order to add a `FitException`. This means that
    if this exception is raised during a model-fit in the analysis class's `log_likelihood_function` that model
    is resampled and does not terminate the code.
    """

    pass


class UnitsException(Exception):
    """
    Raises exceptions associated with unit conversions.

    For example if when constructing a dark matter profile the units and format of the redshifts are input
    incorrectly.
    """

    pass
