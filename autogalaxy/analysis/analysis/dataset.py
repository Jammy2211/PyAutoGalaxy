import copy
import logging
import numpy as np
from typing import Optional, Union
import os

from autoconf import conf
from autoconf.dictable import to_dict, output_to_json
import autofit as af
import autoarray as aa

from autogalaxy import exc
from autogalaxy.analysis.adapt_images.adapt_image_maker import AdaptImageMaker
from autogalaxy.analysis.adapt_images.adapt_images import AdaptImages
from autogalaxy.analysis.maker import FitMaker
from autogalaxy.analysis.preloads import Preloads
from autogalaxy.cosmology.lensing import LensingCosmology
from autogalaxy.cosmology.wrap import Planck15
from autogalaxy.analysis.analysis.analysis import Analysis
from autogalaxy.analysis.result import ResultDataset

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")

class AnalysisDataset(Analysis):
    def __init__(
        self,
        dataset: Union[aa.Imaging, aa.Interferometer],
        adapt_image_maker: Optional[AdaptImageMaker] = None,
        cosmology: LensingCosmology = Planck15(),
        settings_inversion: aa.SettingsInversion = None,
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
        """
        super().__init__(cosmology=cosmology)

        self.dataset = dataset
        self.adapt_image_maker = adapt_image_maker
        self._adapt_images = None

        self.settings_inversion = settings_inversion or aa.SettingsInversion()

        self.preloads = self.preloads_cls()

    @property
    def preloads_cls(self):
        return Preloads

    @property
    def fit_maker_cls(self):
        return FitMaker

    @property
    def adapt_images(self):

        if self._adapt_images is not None:
            return self._adapt_images

        if self.adapt_image_maker is not None:
            return self.adapt_image_maker.adapt_images

    def set_preloads(self, paths: af.DirectoryPaths, model: af.Collection):
        """
        It is common for the model to have components whose parameters are all fixed, and thus the way that component
        fits the data does not change. For example, if all parameter associated with the light profiles of galaxies
        in the model are fixed, the image generated from these galaxies will not change irrespective of the model
        parameters chosen by the non-linear search.

        Preloading exploits this to speed up the log likelihood function, by inspecting the model and storing in memory
        quantities that do not change. For the example above, the image of all galaxies would be stored in memory and
        to perform every fit in the `log_likelihood_funtion`.

        This function sets up all preload quantities, which are described fully in the `preloads` modules. This
        occurs directly before the non-linear search begins, to ensure the model parameterization is fixed.

        Parameters
        ----------
        paths
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        model
            The PyAutoFit model object, which includes model components representing the galaxies that are fitted to
            the imaging data.
        """

        logger.info(
            "PRELOADS - Setting up preloads, may take a few minutes for fits using an inversion."
        )

        self.preloads = self.preloads_cls()

        settings_inversion_original = copy.copy(self.settings_inversion)

        self.settings_inversion.image_mesh_min_mesh_pixels_per_pixel = None
        self.settings_inversion.image_mesh_adapt_background_percent_threshold = None

        fit_maker = self.fit_maker_cls(model=model, fit_from=self.fit_from)

        fit_0 = fit_maker.fit_via_model_from(unit_value=0.45)
        fit_1 = fit_maker.fit_via_model_from(unit_value=0.55)

        if fit_0 is None or fit_1 is None:
            self.preloads = self.preloads_cls(failed=True)

            self.settings_inversion = settings_inversion_original

        else:
            self.preloads = self.preloads_cls.setup_all_via_fits(
                fit_0=fit_0, fit_1=fit_1
            )

            if conf.instance["general"]["test"]["check_preloads"]:
                self.preloads.check_via_fit(fit=fit_0)

            self.settings_inversion = settings_inversion_original

        if isinstance(paths, af.DatabasePaths):
            return

        os.makedirs(paths.profile_path, exist_ok=True)
        self.preloads.output_info_to_summary(file_path=paths.profile_path)

    def modify_after_fit(
        self, paths: af.DirectoryPaths, model: af.AbstractPriorModel, result: af.Result
    ) -> "AnalysisDataset":
        """
        Call functions that perform tasks after a model-fit is completed, for example ensuring the figure of merit
        has not changed from previous estimates and resetting preloads.

        Parameters
        ----------
        paths
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        model
            The PyAutoFit model object, which includes model components representing the galaxies that are fitted to
            the imaging data.
        result
            The result of the model fit that has just been completed.
        """

        self.output_or_check_figure_of_merit_sanity(paths=paths, result=result)

        return self

    def save_attributes(self, paths: af.DirectoryPaths):
        """
        Before the model-fit via the non-linear search begins, this routine saves attributes of the `Analysis` object
        to the `pickles` folder such that they can be loaded after the analysis using PyAutoFit's database and aggregator
        tools.

        For this analysis the following are output:

        - The dataset (data / noise-map / settings / etc.).
        - The settings associated with the inversion.
        - The settings associated with the pixelization.
        - The Cosmology.
        - The adapt image's model image and galaxy images, if used.

        It is common for these attributes to be loaded by many of the template aggregator functions given in the
        `aggregator` modules. For example, when using the database tools to reperform a fit, this will by default
        load the dataset, settings and other attributes necessary to perform a fit using the attributes output by
        this function.

        Parameters
        ----------
        paths
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored, visualization,
            and the pickled objects used by the aggregator output by this function.
        """
        paths.save_fits(
            name="data",
            hdu=self.dataset.data.hdu_for_output,
            prefix="dataset",
        )
        paths.save_fits(
            name="noise_map",
            hdu=self.dataset.noise_map.hdu_for_output,
            prefix="dataset",
        )
        paths.save_json(
            name="settings",
            object_dict=to_dict(self.dataset.settings),
            prefix="dataset",
        )
        paths.save_json(
            name="settings_inversion",
            object_dict=to_dict(self.settings_inversion),
        )
        paths.save_json(
            name="cosmology",
            object_dict=to_dict(self.cosmology),
        )

        if self.adapt_images is not None:
            paths.save_json(
                name="adapt_images",
                object_dict=to_dict(self.adapt_images),
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
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
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

    def output_or_check_figure_of_merit_sanity(
        self, paths: af.DirectoryPaths, result: af.Result
    ):
        """
        Changes to the PyAutoGalaxy source code may inadvertantly change the numerics of how a log likelihood is
        computed. Equally, one may set off a model-fit that resumes from previous results, but change the settings of
        the pixelization or inversion in a way that changes the log likelihood function.

        This function performs an optional sanity check, which raises an exception if the log likelihood calculation
        changes, to ensure a model-fit is not resumed with a different likelihood calculation to the previous run.

        If the model-fit has not been performed before (e.g. it is not a resume) this function outputs
        the `figure_of_merit` (e.g. the log likelihood) of the maximum log likelihood model at the end of the model-fit.

        If the model-fit is a resume, it loads this `figure_of_merit` and compares it against a new value computed for
        the resumed run (again using the maximum log likelihood model inferred). If the two likelihoods do not agree
        and therefore the log likelihood function has changed, an exception is raised and the code execution terminated.

        Parameters
        ----------
        paths
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization, and pickled objects used by the database and aggregator.
        result
            The result containing the maximum log likelihood fit of the model.
        """

        if os.environ.get("PYAUTOFIT_TEST_MODE") == "1":
            return

        if not conf.instance["general"]["test"]["check_figure_of_merit_sanity"]:
            return

        figure_of_merit = result.max_log_likelihood_fit.figure_of_merit

        try:
            figure_of_merit_sanity = paths.load_json(name="figure_of_merit_sanity")

            if not np.isclose(figure_of_merit, figure_of_merit_sanity):
                raise exc.AnalysisException(
                    "Figure of merit sanity check failed. "
                    ""
                    "This means that the existing results of a model fit used a different "
                    "likelihood function compared to the one implemented now.\n\n"
                    f"Old Figure of Merit = {figure_of_merit_sanity}\n"
                    f"New Figure of Merit = {figure_of_merit}"
                )

        except (FileNotFoundError, KeyError):
            paths.save_json(
                name="figure_of_merit_sanity",
                object_dict=figure_of_merit,
            )
