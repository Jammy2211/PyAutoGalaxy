import logging
from typing import Optional, Union

from autoconf.dictable import to_dict, output_to_json

import autofit as af
import autoarray as aa

from autogalaxy.analysis.adapt_images.adapt_image_maker import AdaptImageMaker
from autogalaxy.analysis.adapt_images.adapt_images import AdaptImages
from autogalaxy.cosmology.lensing import LensingCosmology
from autogalaxy.analysis.analysis.analysis import Analysis
from autogalaxy.analysis.result import ResultDataset

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class AnalysisDataset(Analysis):
    def __init__(
        self,
        dataset: Union[aa.Imaging, aa.Interferometer],
        adapt_image_maker: Optional[AdaptImageMaker] = None,
        cosmology: LensingCosmology = None,
        settings_inversion: aa.SettingsInversion = None,
        preloads: aa.Preloads = None,
        title_prefix: str = None,
        **kwargs,
    ):
        """
        Abstract Analysis class for all model-fits which fit galaxies to a dataset, like imaging or interferometer data.

        This class stores the settings used to perform the model-fit for certain components of the model (e.g. a
        pixelization or inversion), the Cosmology used for the analysis and adapt images used for certain model
        classes.

        Parameters
        ----------
        dataset
            The dataset that is the model is fitted too.
        adapt_image_maker
            Makes the adapt-model image and galaxies images of a previous result in a model-fitting pipeline, which are
            used by certain classes for adapting the analysis to the properties of the dataset.
        cosmology
            The Cosmology assumed for this analysis.
        settings_inversion
            Settings controlling how an inversion is fitted during the model-fit, for example which linear algebra
            formalism is used.
        title_prefix
            A string that is added before the title of all figures output by visualization, for example to
            put the name of the dataset and galaxy in the title.
        """
        super().__init__(
            cosmology=cosmology,
            preloads=preloads,
            **kwargs,
        )

        self.dataset = dataset
        self.adapt_image_maker = adapt_image_maker
        self._adapt_images = None

        self.settings_inversion = settings_inversion or aa.SettingsInversion()

        self.title_prefix = title_prefix

    @property
    def adapt_images(self):
        if self._adapt_images is not None:
            return self._adapt_images

        if self.adapt_image_maker is not None:
            return self.adapt_image_maker.adapt_images

    def save_attributes(self, paths: af.DirectoryPaths):
        """
        Before the model-fit via the non-linear search begins, this routine saves attributes of the `Analysis` object
        to the `files` folder such that they can be loaded after the analysis using PyAutoFit's database and aggregator
        tools.

        For this analysis the following are output:

        - The settings associated with the inversion.
        - The settings associated with the pixelization.
        - The Cosmology.

        The following .fits files are also output via the plotter interface:

        - The mask applied to the dataset, in the `PrimaryHDU` of `dataset.fits`.
        - The dataset (data / noise-map / over sampler / etc.).
        - The adapt image's model image and galaxy images, if used.

        It is common for these attributes to be loaded by many of the template aggregator functions given in the
        `aggregator` modules. For example, when using the database tools to reperform a fit, this will by default
        load the dataset, settings and other attributes necessary to perform a fit using the attributes output by
        this function.

        Parameters
        ----------
        paths
            The paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization, and the pickled objects used by the aggregator output by this function.
        """
        paths.save_json(
            name="settings_inversion",
            object_dict=to_dict(self.settings_inversion),
        )
        paths.save_json(
            name="cosmology",
            object_dict=to_dict(self.cosmology),
        )

    def save_results(self, paths: af.DirectoryPaths, result: ResultDataset):
        """
        At the end of a model-fit, this routine saves attributes of the `Analysis` object to the `files`
        folder such that they can be loaded after the analysis using PyAutoFit's database and aggregator tools.

        For this analysis it outputs the following:

        - The maximum log likelihood galaxies of the fit.

        Parameters
        ----------
        paths
            The paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        result
            The result of a model fit, including the non-linear search, samples and maximum likelihood tracer.
        """
        try:
            output_to_json(
                obj=result.max_log_likelihood_galaxies,
                file_path=paths._files_path / "galaxies.json",
            )
        except AttributeError:
            pass

    def adapt_images_via_instance_from(self, instance: af.ModelInstance) -> AdaptImages:
        try:
            return self.adapt_images.updated_via_instance_from(instance=instance)
        except AttributeError:
            pass
