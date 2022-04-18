import autofit as af
from autofit.exc import *
from autoarray.exc import *


class CosmologyException(Exception):
    pass


class GUIException(Exception):
    pass


class ProfileException(Exception):
    pass


class GalaxyException(Exception):
    pass


class PlaneException(Exception):
    pass


class PlottingException(Exception):
    pass


class AnalysisException(Exception):
    pass


class PixelizationException(af.exc.FitException):
    pass


class UnitsException(Exception):
    pass


class SetupException(Exception):
    pass
