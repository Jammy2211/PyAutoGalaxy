from . import aggregator as agg
from . import plot
from . import util
from .dataset.imaging import MaskedImaging, SettingsMaskedImaging, SimulatorImaging
from .dataset.interferometer import (
    MaskedInterferometer,
    SettingsMaskedInterferometer,
    SimulatorInterferometer,
)

from autoconf import conf

from .fit.fit import FitImaging, FitInterferometer
from .galaxy.fit_galaxy import FitGalaxy
from .galaxy.galaxy import Galaxy, HyperGalaxy, Redshift
from .galaxy.galaxy_data import GalaxyData
from .galaxy.galaxy_model import GalaxyModel
from .galaxy.masked_galaxy_data import MaskedGalaxyDataset
from .hyper import hyper_data
from .analysis.analysis import AnalysisImaging
from .analysis.analysis import AnalysisInterferometer
from autogalaxy.analysis.setup import (
    SetupPipeline,
    SetupHyper,
    SetupLightParametric,
    SetupLightInversion,
    SetupMassTotal,
    SetupMassLightDark,
    SetupSMBH,
)
from .plane.plane import Plane
from .profiles import (
    point_sources as ps,
    light_profiles as lp,
    mass_profiles as mp,
    light_and_mass_profiles as lmp,
)
from . import convert

conf.instance.register(__file__)

__version__ = "0.19.0"
