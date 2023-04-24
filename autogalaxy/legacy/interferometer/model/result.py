from autogalaxy.analysis.result import ResultDataset
from autogalaxy.interferometer.fit_interferometer import FitInterferometer


class ResultInterferometer(ResultDataset):
    """
    After the non-linear search of a fit to an interferometer dataset is complete it creates
    this `ResultInterferometer` object, which includes:

    - The samples of the non-linear search (E.g. MCMC chains, nested sampling samples) which are used to compute
    the maximum likelihood model, posteriors and other properties.

    - The model used to fit the data, which uses the samples to create specific instances of the model (e.g.
    an instance of the maximum log likelihood model).

    - The non-linear search used to perform the model fit.

    This class contains a number of methods which use the above objects to create the max log likelihood `Plane`,
    `FitInterferometer`, adapt-galaxy images,etc.

    Parameters
    ----------
    samples
        A PyAutoFit object which contains the samples of the non-linear search, for example the chains of an MCMC
        run of samples of the nested sampler.
    model
        The PyAutoFit model object, which includes model components representing the galaxies that are fitted to
        the interferometer data.
    search
        The non-linear search used to perform this model-fit.

    Returns
    -------
    ResultInterferometer
        The result of fitting the model to the interferometer dataset, via a non-linear search.
    """

    @property
    def max_log_likelihood_fit(self) -> FitInterferometer:
        """
        An instance of a `FitInterferometer` corresponding to the maximum log likelihood model inferred by the
        non-linear search.
        """
        hyper_background_noise = self.analysis.hyper_background_noise_via_instance_from(
            instance=self.instance_copy
        )

        instance = self.analysis.instance_with_associated_adapt_images_from(
            instance=self.instance_copy
        )

        plane = self.analysis.plane_via_instance_from(instance=instance)

        return self.analysis.fit_interferometer_via_plane_from(
            plane=plane, hyper_background_noise=hyper_background_noise
        )
