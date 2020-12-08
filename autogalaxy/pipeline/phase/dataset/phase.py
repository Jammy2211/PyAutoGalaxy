from astropy import cosmology as cosmo

import autofit as af
from autoarray.inversion import pixelizations as pix
from autoarray.inversion import regularization as reg
from autogalaxy.pipeline.phase import abstract
from autogalaxy.pipeline.phase import extensions
from autogalaxy.pipeline.phase.dataset.result import Result


class PhaseDataset(abstract.AbstractPhase):
    galaxies = af.PhaseProperty("galaxies")

    Result = Result

    def __init__(self, settings, search, galaxies=None, cosmology=cosmo.Planck15):
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
        self.use_as_hyper_dataset = False
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

    @property
    def model_classes_for_hyper_phase(self) -> tuple:
        raise NotImplementedError

    def extend_with_hyper_phase(self, setup_hyper):

        if len(self.model_classes_for_hyper_phase) == 0:
            return self

        if self.has_pixelization:
            hyper_search = setup_hyper.hyper_search_with_inversion
        else:
            hyper_search = setup_hyper.hyper_search_no_inversion

        self.use_as_hyper_dataset = True

        hyper_phase = extensions.HyperPhase(
            phase=self,
            hyper_search=hyper_search,
            model_classes=self.model_classes_for_hyper_phase,
        )

        return hyper_phase
