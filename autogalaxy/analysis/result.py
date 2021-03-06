import autofit as af
from autogalaxy.galaxy import galaxy as g


def last_result_with_use_as_hyper_dataset(results):

    if results is not None:
        if results.last is not None:
            for index, result in enumerate(reversed(results)):
                if hasattr(result, "use_as_hyper_dataset"):
                    if result.use_as_hyper_dataset:
                        return result


class Result(af.Result):
    def __init__(self, samples, model, analysis, search, use_as_hyper_dataset=False):
        """
        The results of a `NonLinearSearch` performed by a phase.

        Parameters
        ----------
        samples : af.Samples
            A class containing the samples of the non-linear search, including methods to get the maximum log
            likelihood model, errors, etc.
        model : af.ModelMapper
            The model used in this result model-fit.
        analysis : Analysis
            The Analysis class used by this model-fit to fit the model to the data.
        search : af.NonLinearSearch
            The `NonLinearSearch` search used by this model fit.
        use_as_hyper_dataset : bool
            Whether this result's phase contains hyper phases, allowing it to be used a hyper dataset.
        """
        super().__init__(samples=samples, model=model, search=search)

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


class ResultDataset(Result):
    @property
    def max_log_likelihood_fit(self):

        hyper_image_sky = self.analysis.hyper_image_sky_for_instance(
            instance=self.instance
        )

        hyper_background_noise = self.analysis.hyper_background_noise_for_instance(
            instance=self.instance
        )

        return self.analysis.imaging_fit_for_plane(
            plane=self.max_log_likelihood_plane,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

    @property
    def mask(self):
        return self.max_log_likelihood_fit.mask

    @property
    def dataset(self):
        return self.max_log_likelihood_fit.masked_dataset

    @property
    def pixelization(self):
        for galaxy in self.max_log_likelihood_fit.galaxies:
            if galaxy.pixelization is not None:
                return galaxy.pixelization


class ResultImaging(ResultDataset):

    pass


class ResultInterferometer(ResultDataset):
    pass
