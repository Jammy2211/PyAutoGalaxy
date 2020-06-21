import autofit as af
from autogalaxy.pipeline.phase.abstract.result import Result


# noinspection PyAbstractClass


class AbstractPhase(af.AbstractPhase):
    Result = Result

    @af.convert_paths
    def __init__(self, paths, *, search):
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
