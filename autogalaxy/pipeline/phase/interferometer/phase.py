import autofit as af
from astropy import cosmology as cosmo
from autoarray.operators import transformer
from autogalaxy.pipeline.phase.settings import PhaseSettingsInterferometer
from autogalaxy.pipeline.phase import dataset
from autogalaxy.pipeline.phase.interferometer.analysis import Analysis
from autogalaxy.pipeline.phase.interferometer.meta_interferometer import (
    MetaInterferometer,
)
from autogalaxy.pipeline.phase.interferometer.result import Result


class PhaseInterferometer(dataset.PhaseDataset):
    galaxies = af.PhaseProperty("galaxies")
    hyper_background_noise = af.PhaseProperty("hyper_background_noise")

    Analysis = Analysis
    Result = Result

    @af.convert_paths
    def __init__(
        self,
        paths,
        *,
        search,
        real_space_mask,
        galaxies=None,
        hyper_background_noise=None,
        settings=PhaseSettingsInterferometer(),
        cosmology=cosmo.Planck15,
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

        paths.tag = settings.phase_with_inversion_tag

        super().__init__(
            paths,
            galaxies=galaxies,
            settings=settings,
            search=search,
            cosmology=cosmology,
        )

        self.hyper_background_noise = hyper_background_noise

        self.is_hyper_phase = False

        self.meta_dataset = MetaInterferometer(
            settings=settings,
            model=self.model,
            real_space_mask=real_space_mask,
            is_hyper_phase=False,
        )

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def modify_visibilities(self, visibilities, results):
        """
        Customize an masked_interferometer. e.g. removing lens light.

        Parameters
        ----------
        image: scaled_array.ScaledSquarePixelArray
            An masked_interferometer that has been masked
        results: autofit.tools.pipeline.ResultsCollection
            The result of the previous lens

        Returns
        -------
        masked_interferometer: scaled_array.ScaledSquarePixelArray
            The modified image (not changed by default)
        """
        return visibilities

    def make_analysis(self, dataset, mask, results=None):
        """
        Create an lens object. Also calls the prior passing and masked_interferometer modifying functions to allow child
        classes to change the behaviour of the phase.

        Parameters
        ----------
        mask: Mask
            The default masks passed in by the pipeline
        dataset: im.Interferometer
            An masked_interferometer that has been masked
        results: autofit.tools.pipeline.ResultsCollection
            The result from the previous phase

        Returns
        -------
        lens : Analysis
            An lens object that the non-linear search calls to determine the fit of a set of values
        """
        self.meta_dataset.model = self.model
        modified_visibilities = self.modify_visibilities(
            visibilities=dataset.visibilities, results=results
        )

        masked_interferometer = self.meta_dataset.masked_dataset_from(
            dataset=dataset,
            mask=mask,
            results=results,
            modified_visibilities=modified_visibilities,
        )

        self.output_phase_info()

        analysis = self.Analysis(
            masked_interferometer=masked_interferometer,
            cosmology=self.cosmology,
            image_path=self.search.paths.image_path,
            results=results,
        )

        return analysis

    def make_phase_attributes(self, analysis):
        return PhaseAttributes(
            cosmology=self.cosmology,
            hyper_model_image=analysis.hyper_model_image,
            hyper_galaxy_image_path_dict=analysis.hyper_galaxy_image_path_dict,
        )

    def output_phase_info(self):

        file_phase_info = "{}/{}".format(self.search.paths.output_path, "phase.info")

        with open(file_phase_info, "w") as phase_info:
            phase_info.write("Optimizer = {} \n".format(type(self.search).__name__))
            phase_info.write(
                "Sub-grid size = {} \n".format(self.meta_dataset.settings.sub_size)
            )
            phase_info.write(
                "Primary Beam shape = {} \n".format(
                    self.meta_dataset.settings.primary_beam_shape_2d
                )
            )
            phase_info.write("Cosmology = {} \n".format(self.cosmology))

            phase_info.close()


class PhaseAttributes:
    def __init__(self, cosmology, hyper_model_image, hyper_galaxy_image_path_dict):

        self.cosmology = cosmology
        self.hyper_model_image = hyper_model_image
        self.hyper_galaxy_image_path_dict = hyper_galaxy_image_path_dict
