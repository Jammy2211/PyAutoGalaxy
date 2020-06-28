import autofit as af
from autofit.exc import PipelineException


class CosmologyException(Exception):
    pass


class GalaxyException(Exception):
    pass


class PlaneException(Exception):
    pass


class PlottingException(Exception):
    pass


class PixelizationException(af.exc.FitException):
    pass


class UnitsException(Exception):
    pass
