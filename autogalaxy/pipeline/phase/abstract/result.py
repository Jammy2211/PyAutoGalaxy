import autofit as af
from autogalaxy.galaxy import galaxy as g


class Result(af.Result):
    def __init__(
        self, samples, previous_model, analysis, search, use_as_hyper_dataset=False
    ):
        """
        The results of a non-linear search performed by a phase.

        Parameters
        ----------
        samples : af.Samples
            A class containing the samples of the non-linear search, including methods to get the maximum log
            likelihood model, errors, etc.
        previous_model : af.ModelMapper
            The model used in this result model-fit.
        analysis : Analysis
            The Analysis class used by this model-fit to fit the model to the data.
        search : af.NonLinearSearch
            The non-linear search search used by this model fit.
        use_as_hyper_dataset : bool
            Whether this result's phase contains hyper phases, allowing it to be used a hyper dataset.
        """
        super().__init__(samples=samples, previous_model=previous_model, search=search)

        self.analysis = analysis
        self.use_as_hyper_dataset = use_as_hyper_dataset

    @property
    def max_log_likelihood_plane(self):

        instance = self.analysis.associate_hyper_images(instance=self.instance)

        return self.analysis.plane_for_instance(instance=instance)

    @property
    def path_galaxy_tuples(self) -> [(str, g.Galaxy)]:
        """
        Tuples associating the names of galaxies with instances from the best fit
        """
        return self.instance.path_instance_tuples_for_class(cls=g.Galaxy)
