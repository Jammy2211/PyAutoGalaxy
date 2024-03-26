from autoconf import cached_property

from autogalaxy.analysis.adapt_images.adapt_images import AdaptImages


class AdaptImageMaker:
    def __init__(self, result, use_model_images: bool = False):
        """
        Class used for making adapt images on-the-fly for efficient computations.

        Adapt images can be slow to compute, especially for large datasets. When an analysis pipeline is resumed
        from already computed results, the adapt images in early fits may not be used. By default, a pipeline would
        still recalculate the adapt images for every fit, slowing down the time it takes to resume a pipeline.

        By using the `AdaptImageMaker`, the adapt images are computed on-the-fly during the fit, when and only when
        the `adapt_images` attribute of the `Analysis` class is called. This means that if the adapt images are not
        used when resuming a pipeline, they are not recalculated, speeding up the time it takes to resume a pipeline.

        Parameters
        ----------
        result
            The result from a previous stage of the pipeline which contains the information necessary to make the
            adapt images.
        use_model_images
            Whether to use the model images from the result to make the adapt images. If `False`, the galaxies
            subtracted images of each dataset are used.
        """

        self.result = result
        self.use_model_images = use_model_images

    @cached_property
    def adapt_images(self):
        """
        Returns the adapt images from the result.

        The adapt images are therefore only computed when this attribute is called, which for pipelines resuming
        from already computed results means this function is often omitted, saving computational time.

        This function is cached, therefore once the adapt images are computed they are stored in memory and not
        recomputed when this attribute is called again, saving computational time.

        Returns
        -------
        The adapt images from the result.
        """
        return self.result.adapt_images_from(use_model_images=self.use_model_images)
