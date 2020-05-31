import autoarray as aa
import autofit as af
from autoconf import conf
from autoarray.operators.inversion import pixelizations as pix


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


class MetaDataset:
    def __init__(self, settings, model, is_hyper_phase=False):
        self.settings = settings
        self.is_hyper_phase = is_hyper_phase
        self.model = model

    def mask_with_phase_sub_size_from_mask(self, mask):

        if mask.sub_size != self.settings.sub_size:
            mask = aa.Mask.manual(
                mask=mask,
                pixel_scales=mask.pixel_scales,
                sub_size=self.settings.sub_size,
                origin=mask.origin,
            )

        return mask

    @property
    def pixelization(self):
        for galaxy in self.model.galaxies:
            if hasattr(galaxy, "pixelization"):
                if galaxy.pixelization is not None:
                    if isinstance(galaxy.pixelization, af.PriorModel):
                        return galaxy.pixelization.cls
                    else:
                        return galaxy.pixelization

    @property
    def has_pixelization(self):
        return self.pixelization is not None

    @property
    def uses_cluster_inversion(self):
        if self.model.galaxies:
            for galaxy in self.model.galaxies:
                if isinstance_or_prior(galaxy.pixelization, pix.VoronoiBrightnessImage):
                    return True
        return False

    @property
    def pixelizaition_is_model(self):
        if self.model.galaxies:
            for galaxy in self.model.galaxies:
                if isprior(galaxy.pixelization):
                    return True
        return False
