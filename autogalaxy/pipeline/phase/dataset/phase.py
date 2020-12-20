from astropy import cosmology as cosmo
from autoconf import conf
import autofit as af
from autoarray.inversion import pixelizations as pix
from autoarray.inversion import regularization as reg
from autogalaxy.pipeline.phase import abstract
from autogalaxy.pipeline.phase import extensions
from autogalaxy.pipeline.phase.dataset.result import Result

import copy
import os
import shutil
from distutils.dir_util import copy_tree


class PhaseDataset(abstract.AbstractPhase):
    galaxies = af.PhaseProperty("galaxies")

    Result = Result

    def __init__(
        self,
        settings,
        search,
        galaxies=None,
        cosmology=cosmo.Planck15,
        use_as_hyper_dataset=False,
    ):
        """

        A phase in an lens pipeline. Uses the set non_linear search to try to fit models and hyper_galaxies
        passed to it.

        Parameters
        ----------
        search: class
            The class of a non_linear search
        """

        super().__init__(
            search=search, settings=settings, galaxies=galaxies, cosmology=cosmology
        )

        self.hyper_name = None
        self.use_as_hyper_dataset = use_as_hyper_dataset
        self.is_hyper_phase = False

    def run(
        self,
        dataset,
        mask,
        results=None,
        info=None,
        pickle_files=None,
        log_likelihood_cap=None,
    ):
        """
        Run this phase.

        Parameters
        ----------
        mask: Mask2D
            The default masks passed in by the pipeline
        results: autofit.tools.pipeline.ResultsCollection
            An object describing the results of the last phase or None if no phase has been executed
        dataset: scaled_array.ScaledSquarePixelArray
            An masked_imaging that has been masked

        Returns
        -------
        result: AbstractPhase.Result
            A result object comprising the best fit model and other hyper_galaxies.
        """

        self.model = self.model.populate(results)

        results = results or af.ResultsCollection()

        self.modify_dataset(dataset=dataset, results=results)
        self.modify_settings(dataset=dataset, results=results)
        self.modify_search_paths()

        analysis = self.make_analysis(dataset=dataset, mask=mask, results=results)

        result = self.run_analysis(
            analysis=analysis,
            info=info,
            pickle_files=pickle_files,
            log_likelihood_cap=log_likelihood_cap,
        )

        return self.make_result(result=result, analysis=analysis)

    def make_analysis(self, dataset, mask, results=None):
        """
        Returns an lens object. Also calls the prior passing and masked_imaging modifying functions to allow child
        classes to change the behaviour of the phase.

        Parameters
        ----------
        mask: Mask2D
            The default masks passed in by the pipeline
        dataset: im.Imaging
            An masked_imaging that has been masked
        results: autofit.tools.pipeline.ResultsCollection
            The result from the previous phase

        Returns
        -------
        lens : Analysis
            An lens object that the `NonLinearSearch` calls to determine the fit of a set of values
        """
        raise NotImplementedError()

    def modify_dataset(self, dataset, results):
        pass

    def modify_settings(self, dataset, results):
        pass

    def modify_search_paths(self):
        """
        Modify the output paths of the phase before the non-linear search is run, so that the output path can be
        customized using the tags of the phase.
        """
        if self.hyper_name is None:
            hyper_tag = ""
        else:
            hyper_tag = f"{self.hyper_name}__"

        if not self.has_pixelization:
            self.search.paths.tag = f"{hyper_tag}{self.settings.phase_tag_no_inversion}"
        else:
            self.search.paths.tag = (
                f"{hyper_tag}{self.settings.phase_tag_with_inversion}"
            )

        if self.hyper_name == "hyper":

            rename_hyper_combined = conf.instance["general"]["hyper"][
                "rename_hyper_combined"
            ]

            if rename_hyper_combined:

                output_path_hyper = copy.copy(self.search.paths.output_path)
                output_path_hyper_combined = output_path_hyper.replace(
                    "hyper", "hyper_combined"
                )

                if os.path.exists(output_path_hyper_combined):
                    copy_tree(output_path_hyper_combined, output_path_hyper)
                    if os.path.isfile(f"{output_path_hyper_combined}.zip"):
                        shutil.copyfile(
                            f"{output_path_hyper_combined}.zip",
                            f"{output_path_hyper}.zip",
                        )
                    shutil.rmtree(output_path_hyper_combined)

                if os.path.isfile(f"{output_path_hyper_combined}.zip"):
                    os.remove(f"{output_path_hyper_combined}.zip")

                if os.path.exists(os.path.join(output_path_hyper_combined, "..")):
                    shutil.rmtree(os.path.join(output_path_hyper_combined, ".."))

    def extend_with_hyper_phase(self, setup_hyper, include_hyper_image_sky=True):

        self.use_as_hyper_dataset = True

        if not self.has_pixelization:
            if setup_hyper.hypers_all_off:
                return self
            if setup_hyper.hypers_all_except_image_sky_off:
                if not include_hyper_image_sky:
                    return self

        if self.has_pixelization:
            hyper_search = setup_hyper.hyper_search_with_inversion
        else:
            hyper_search = setup_hyper.hyper_search_no_inversion

        if include_hyper_image_sky:
            hyper_image_sky = setup_hyper.hyper_image_sky
        else:
            hyper_image_sky = None

        return extensions.HyperPhase(
            phase=self,
            hyper_search=hyper_search,
            model_classes=(pix.Pixelization, reg.Regularization),
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=setup_hyper.hyper_background_noise,
            hyper_galaxy_names=setup_hyper.hyper_galaxy_names,
        )
