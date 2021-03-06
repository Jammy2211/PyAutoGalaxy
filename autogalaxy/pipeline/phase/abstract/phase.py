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


# noinspection PyAbstractClass
class AbstractPhase(af.AbstractPhase):

    Result = Result

    def __init__(self, *, settings, search, galaxies=None, cosmology=cosmo.Planck15):
        """
        A phase in an lens pipeline. Uses the set non_linear search to try to fit
        models and hyper_galaxies passed to it.

        Parameters
        ----------
        search: class
            The class of a non_linear search
        """

        self.use_as_hyper_dataset = False

        super().__init__(search=search)

        self.settings = settings
        self.galaxies = galaxies or []
        self.cosmology = cosmology

    def make_result(self, result, analysis):

        return self.Result(
            samples=result.samples,
            previous_model=result.previous_model,
            analysis=analysis,
            search=self.search,
            use_as_hyper_dataset=self.use_as_hyper_dataset,
        )
