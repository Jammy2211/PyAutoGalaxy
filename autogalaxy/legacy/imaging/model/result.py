from autogalaxy.imaging.fit_imaging import FitImaging

from autogalaxy.imaging.model.result import ResultImaging as ResultImagingBase


class ResultImaging(ResultImagingBase):
    """
    After the non-linear search of a fit to an imaging dataset is complete it creates this `ResultImaging`, object
    which includes:

    - The samples of the non-linear search (E.g. MCMC chains, nested sampling samples) which are used to compute
    the maximum likelihood model, posteriors and other properties.

    - The model used to fit the data, which uses the samples to create specific instances of the model (e.g.
    an instance of the maximum log likelihood model).

    - The non-linear search used to perform the model fit.

    This class contains a number of methods which use the above objects to create the max log likelihood `Plane`,
    `FitImaging`, hyper-galaxy images,etc.

    Parameters
    ----------
    samples
        A PyAutoFit object which contains the samples of the non-linear search, for example the chains of an MCMC
        run of samples of the nested sampler.
    model
        The PyAutoFit model object, which includes model components representing the galaxies that are fitted to
        the imaging data.
    search
        The non-linear search used to perform this model-fit.

    Returns
    -------
    ResultImaging
        The result of fitting the model to the imaging dataset, via a non-linear search.
    """

    @property
    def max_log_likelihood_fit(self) -> FitImaging:
        """
        An instance of a `FitImaging` corresponding to the maximum log likelihood model inferred by the non-linear
        search.
        """
        hyper_image_sky = self.analysis.hyper_image_sky_via_instance_from(
            instance=self.instance_copy
        )

        hyper_background_noise = self.analysis.hyper_background_noise_via_instance_from(
            instance=self.instance_copy
        )

        instance = self.analysis.instance_with_associated_hyper_images_from(
            instance=self.instance_copy
        )

        plane = self.analysis.plane_via_instance_from(instance=instance)

        return self.analysis.fit_imaging_via_plane_from(
            plane=plane,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )
