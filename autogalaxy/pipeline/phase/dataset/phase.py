import autofit as af
from autoarray.operators.inversion import pixelizations as pix
from autoarray.operators.inversion import regularization as reg
from autogalaxy.hyper import hyper_data as hd
from astropy import cosmology as cosmo
from autofit.tools.phase import Dataset
from autogalaxy.pipeline.phase import abstract
from autogalaxy.pipeline.phase import extensions
from autogalaxy.pipeline.phase.dataset.result import Result


class PhaseDataset(abstract.AbstractPhase):
    galaxies = af.PhaseProperty("galaxies")

    Result = Result

    @af.convert_paths
    def __init__(
        self, paths, settings, search, galaxies=None, cosmology=cosmo.Planck15
    ):
        """

        A phase in an lens pipeline. Uses the set non_linear search to try to fit models and hyper_galaxies
        passed to it.

        Parameters
        ----------
        search: class
            The class of a non_linear search
        """

        has_inversion = inversion_in_galaxies(galaxies=galaxies)

        if not has_inversion:
            paths.tag = settings.phase_no_inversion_tag
        else:
            paths.tag = settings.phase_with_inversion_tag

        super().__init__(paths, search=search)
        self.settings = settings
        self.galaxies = galaxies or []
        self.cosmology = cosmology
        self.use_as_hyper_dataset = False
        self.is_hyper_phase = False

    def run(self, dataset: Dataset, mask, results=None, info=None):
        """
        Run this phase.

        Parameters
        ----------
        mask: Mask
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
        self.save_metadata(dataset=dataset)
        self.save_dataset(dataset=dataset)
        self.save_mask(mask=mask)
        self.save_meta_dataset(meta_dataset=self.meta_dataset)

        self.model = self.model.populate(results)

        results = results or af.ResultsCollection()

        analysis = self.make_analysis(dataset=dataset, mask=mask, results=results)

        phase_attributes = self.make_phase_attributes(analysis=analysis)
        self.save_phase_attributes(phase_attributes=phase_attributes)

        result = self.run_analysis(analysis=analysis, info=info)

        return self.make_result(result=result, analysis=analysis)

    def make_analysis(self, dataset, mask, results=None):
        """
        Create an lens object. Also calls the prior passing and masked_imaging modifying functions to allow child
        classes to change the behaviour of the phase.

        Parameters
        ----------
        mask: Mask
            The default masks passed in by the pipeline
        dataset: im.Imaging
            An masked_imaging that has been masked
        results: autofit.tools.pipeline.ResultsCollection
            The result from the previous phase

        Returns
        -------
        lens : Analysis
            An lens object that the non-linear search calls to determine the fit of a set of values
        """
        raise NotImplementedError()

    def extend_with_inversion_phase(self, inversion_search):
        return extensions.InversionPhase(phase=self, search=inversion_search)

    def extend_with_multiple_hyper_phases(
        self,
        hyper_combined_search=None,
        inversion_search=None,
        hyper_galaxy_search=None,
        include_background_sky=False,
        include_background_noise=False,
        hyper_galaxy_phase_first=False,
    ):

        self.use_as_hyper_dataset = True

        hyper_phases = []

        if self.meta_dataset.has_pixelization and inversion_search:
            if not include_background_sky and not include_background_noise:
                phase_inversion = extensions.InversionPhase(
                    phase=self,
                    search=inversion_search,
                    model_classes=(pix.Pixelization, reg.Regularization),
                )
            elif include_background_sky and not include_background_noise:
                phase_inversion = extensions.InversionPhase(
                    phase=self,
                    search=inversion_search,
                    model_classes=(
                        pix.Pixelization,
                        reg.Regularization,
                        hd.HyperImageSky,
                    ),
                )
            elif not include_background_sky and include_background_noise:
                phase_inversion = extensions.InversionPhase(
                    phase=self,
                    search=inversion_search,
                    model_classes=(
                        pix.Pixelization,
                        reg.Regularization,
                        hd.HyperBackgroundNoise,
                    ),
                )
            else:
                phase_inversion = extensions.InversionPhase(
                    phase=self,
                    search=inversion_search,
                    model_classes=(
                        pix.Pixelization,
                        reg.Regularization,
                        hd.HyperImageSky,
                        hd.HyperBackgroundNoise,
                    ),
                )

            hyper_phases.append(phase_inversion)

        if hyper_galaxy_search is not None:
            phase_hyper_galaxy = extensions.HyperGalaxyPhase(
                phase=self,
                search=hyper_galaxy_search,
                include_sky_background=include_background_sky,
                include_noise_background=include_background_noise,
            )
            hyper_phases.append(phase_hyper_galaxy)

        if hyper_galaxy_phase_first:
            if inversion_search is not None and hyper_galaxy_search is not None:
                hyper_phases = [phase for phase in reversed(hyper_phases)]

        if len(hyper_phases) == 0:
            return self
        else:
            return extensions.CombinedHyperPhase(
                phase=self, search=hyper_combined_search, hyper_phases=hyper_phases
            )


def inversion_in_galaxies(galaxies):

    if galaxies is not dict:
        return False

    for name, galaxy_model in galaxies.items():
        if galaxy_model.pixelization is not None:
            return True
        return False
