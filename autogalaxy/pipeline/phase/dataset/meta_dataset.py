import autoarray as aa
import autofit as af
from autoconf import conf
from autoarray.inversion import pixelizations as pix


class MetaDataset:
    def __init__(self, settings, model, is_hyper_phase=False):
        self.settings = settings
        self.is_hyper_phase = is_hyper_phase
        self.model = model
