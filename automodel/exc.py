import autofit as af



class CosmologyException(Exception):
    pass


class GalaxyException(Exception):
    pass


class PixelizationException(af.exc.FitException):
    pass


class InversionException(af.exc.FitException):
    pass


class PlottingException(Exception):
    pass


class UnitsException(Exception):
    pass
