from os import path
from astropy import cosmology as cosmo

import autofit as af
from autogalaxy.dataset import imaging
from autogalaxy.pipeline.phase import dataset
from autoarray.inversion import pixelizations as pix
from autoarray.inversion import regularization as reg
from autogalaxy.pipeline.phase.imaging.analysis import Analysis
from autogalaxy.pipeline.phase.imaging.result import Result
from autogalaxy.pipeline.phase.settings import SettingsPhaseImaging


class PhaseImaging(dataset.PhaseDataset):
    galaxies = af.PhaseProperty("galaxies")
    hyper_image_sky = af.PhaseProperty("hyper_image_sky")
    hyper_background_noise = af.PhaseProperty("hyper_background_noise")

    Analysis = Analysis
    Result = Result

    def __init__(
        self,
        *,
        search,
        galaxies=None,
        hyper_image_sky=None,
        hyper_background_noise=None,
        settings=SettingsPhaseImaging(),
        cosmology=cosmo.Planck15,
        use_as_hyper_dataset=False
    ):
        """

        A phase in an lens pipeline. Uses the set non_linear search to try to fit models and hyper_galaxies
        passed to it.

        Parameters
        ----------
        search: class
            The class of a non_linear search
        sub_size: int
            The side length of the subgrid
        """

        super().__init__(
            search=search,
            galaxies=galaxies,
            settings=settings,
            cosmology=cosmology,
            use_as_hyper_dataset=use_as_hyper_dataset,
        )

        self.hyper_image_sky = hyper_image_sky
        self.hyper_background_noise = hyper_background_noise

        self.is_hyper_phase = False

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

        masked_imaging = imaging.MaskedImaging(
            imaging=dataset, mask=mask, settings=self.settings.settings_masked_imaging
        )

        self.output_phase_info()

        return self.Analysis(
            masked_imaging=masked_imaging,
            settings=self.settings,
            cosmology=self.cosmology,
            results=results,
        )

    def output_phase_info(self):

        file_phase_info = path.join(self.search.paths.output_path, "phase.info")

        with open(file_phase_info, "w") as phase_info:
            phase_info.write("Optimizer = {} \n".format(type(self.search).__name__))
            phase_info.write(
                "Sub-grid size = {} \n".format(
                    self.settings.settings_masked_imaging.sub_size
                )
            )
            phase_info.write(
                "PSF shape = {} \n".format(
                    self.settings.settings_masked_imaging.psf_shape_2d
                )
            )
            phase_info.write("Cosmology = {} \n".format(self.cosmology))

            phase_info.close()


class PhaseAttributes:
    def __init__(self, cosmology, hyper_model_image, hyper_galaxy_image_path_dict):
        self.cosmology = cosmology
        self.hyper_model_image = hyper_model_image
        self.hyper_galaxy_image_path_dict = hyper_galaxy_image_path_dict
