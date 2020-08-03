from autogalaxy.pipeline.phase import abstract


class Result(abstract.result.Result):
    @property
    def max_log_likelihood_fit(self):

        hyper_image_sky = self.analysis.hyper_image_sky_for_instance(
            instance=self.instance
        )

        hyper_background_noise = self.analysis.hyper_background_noise_for_instance(
            instance=self.instance
        )

        return self.analysis.positions_fit_for_tracer(
            plane=self.max_log_likelihood_plane,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

    @property
    def mask(self):
        return self.max_log_likelihood_fit.mask

    @property
    def masked_dataset(self):
        return self.max_log_likelihood_fit.masked_dataset

    @property
    def pixelization(self):
        for galaxy in self.max_log_likelihood_fit.galaxies:
            if galaxy.pixelization is not None:
                return galaxy.pixelization
