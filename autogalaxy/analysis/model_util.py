import autofit as af
from autoarray.inversion import pixelizations as pix
from autogalaxy.galaxy.galaxy_model import is_light_profile_class
from autogalaxy.pipeline.phase.abstract.result import Result
from astropy import cosmology as cosmo


def isprior(obj):
    if isinstance(obj, af.PriorModel):
        return True
    return False


def isinstance_or_prior(obj, cls):
    if isinstance(obj, cls):
        return True
    if isinstance(obj, af.PriorModel) and obj.cls == cls:
        return True
    return False


def pixelization_from_model(model):
    for galaxy in model.galaxies:
        if hasattr(galaxy, "pixelization"):
            if galaxy.pixelization is not None:
                if isinstance(galaxy.pixelization, af.PriorModel):
                    return galaxy.pixelization.cls
                else:
                    return galaxy.pixelization


def has_pixelization_from_model(model):

    pixelization = pixelization_from_model(model=model)

    return pixelization is not None


def pixelization_is_model_from_model(model):
    if model.galaxies:
        for galaxy in model.galaxies:
            if isprior(galaxy.pixelization):
                return True
    return False


@property
def uses_cluster_inversion(self):
    if self.galaxies:
        for galaxy in self.galaxies:
            if isinstance_or_prior(galaxy.pixelization, pix.VoronoiBrightnessImage):
                return True
    return False


