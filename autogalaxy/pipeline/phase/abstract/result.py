import autofit as af
from autogalaxy.galaxy import galaxy as g


class Result(af.Result):
    def __init__(
        self, samples, previous_model, analysis, optimizer, use_as_hyper_dataset=False
    ):
        """
        The result of a phase
        """
        super().__init__(samples=samples, previous_model=previous_model)

        self.analysis = analysis
        self.optimizer = optimizer
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
