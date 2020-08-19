import autofit as af
from autoarray.inversion import pixelizations as pix
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

    @af.convert_paths
    def __init__(
        self, paths, *, settings, search, galaxies=None, cosmology=cosmo.Planck15
    ):
        """
        A phase in an lens pipeline. Uses the set non_linear search to try to fit
        models and hyper_galaxies passed to it.

        Parameters
        ----------
        search: class
            The class of a non_linear search
        """

        self.use_as_hyper_dataset = False

        super().__init__(paths=paths, search=search)

        self.settings = settings
        self.galaxies = galaxies or []
        self.cosmology = cosmology

    @property
    def folders(self):
        return self.search.folders

    @property
    def phase_property_collections(self):
        """
        Returns
        -------
        phase_property_collections: [PhaseProperty]
            A list of phase property collections associated with this phase. This is
            used in automated prior passing and should be overridden for any phase that
            contains its own PhasePropertys.
        """
        return []

    @property
    def path(self):
        return self.search.path

    def make_result(self, result, analysis):

        return self.Result(
            samples=result.samples,
            previous_model=result.previous_model,
            analysis=analysis,
            search=self.search,
            use_as_hyper_dataset=self.use_as_hyper_dataset,
        )

    def run(self, dataset, mask, results=None):
        raise NotImplementedError()

    def modify_search_paths(self):
        raise NotImplementedError()

    @property
    def pixelization(self):
        for galaxy in self.galaxies:
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
        if self.galaxies:
            for galaxy in self.galaxies:
                if isinstance_or_prior(galaxy.pixelization, pix.VoronoiBrightnessImage):
                    return True
        return False

    @property
    def pixelization_is_model(self):
        if self.galaxies:
            for galaxy in self.galaxies:
                if isprior(galaxy.pixelization):
                    return True
        return False
