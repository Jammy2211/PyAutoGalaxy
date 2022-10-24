import numpy as np
from typing import Dict, List, Optional

import autoarray as aa

from autogalaxy.analysis.preloads import Preloads
from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.plane.plane import Plane
from autogalaxy.profiles.light_profiles.light_profiles_linear import LightProfileLinear


class AbstractFitImaging(aa.FitImaging):
    def __init__(
        self,
        dataset: aa.Imaging,
        plane: Plane,
        preloads: aa.Preloads = Preloads(),
        profiling_dict: Optional[Dict] = None,
    ):
        """
        An abstract object which fits an imaging dataset using a `Plane` object.

        The fit performs the following steps:

        1) Compute the sum of all images of galaxy light profiles in the `Plane`.

        2) Blur this with the imaging PSF to created the `blurred_image`.

        3) Subtract this image from the `data` to create the `profile_subtracted_image`.

        4) If the `Plane` has any linear algebra objects (e.g. linear light profiles, a pixelization / regulariation)
        fit the `profile_subtracted_image` with these objects via an inversion.

        5) Compute the `model_data` as the sum of the `blurred_image` and `reconstructed_data` of the inversion (if
        an inversion is not performed the `model_data` is only the `blurred_image`.

        The remaining steps, where quantities like residuals and a chi-squared are computed, depend on the type
        of fit object (which is dictated by the data format) and are documented in their specific fit classes.

        When performing a `model-fit`via an `AnalysisImaging` object the `figure_of_merit` of this `FitImaging` object
        is called and returned in the `log_likelihood_function`.

        Parameters
        -----------
        dataset
            The imaging dataset which is fitted by the galaxies in the plane.
        plane
            The plane of galaxies whose light profile images are used to fit the imaging data.
        preloads
            Contains preloaded calculations (e.g. linear algebra matrices) which can skip certain calculations in
            the fit.
        profiling_dict
            A dictionary which if passed to the fit records how long fucntion calls which have the `profile_func`
            decorator take to run.
        """

        super().__init__(dataset=dataset, profiling_dict=profiling_dict)

        self.plane = plane

        self.preloads = preloads

    @property
    def blurred_image(self) -> aa.Array2D:
        """
        Returns the image of the light profiles of all galaxies in the fit's plane, convolved with the
        imaging dataset's PSF.
        """
        return self.plane.blurred_image_2d_from(
            grid=self.dataset.grid,
            convolver=self.dataset.convolver,
            blurring_grid=self.dataset.blurring_grid,
        )

    @property
    def profile_subtracted_image(self) -> aa.Array2D:
        """
        Returns the dataset's image data with all blurred light profile images in the fit's plane subtracted.
        """
        return self.image - self.blurred_image

    @property
    def model_data(self) -> aa.Array2D:
        """
        Returns the model-image that is used to fit the data.

        If the plane does not have any linear objects and therefore omits an inversion, the model data is the
        sum of all light profile images blurred with the PSF.

        If a inversion is included it is the sum of this image and the inversion's reconstruction of the image.
        """
        return self.blurred_image

    @property
    def galaxies(self) -> List[Galaxy]:
        return self.plane.galaxies

    @property
    def grid(self) -> aa.type.Grid2DLike:
        return self.imaging.grid

    @property
    def galaxy_model_image_dict(self) -> Dict[Galaxy, np.ndarray]:
        """
        A dictionary which associates every galaxy in the plane with its `model_image`.

        This image is the image of the sum of:

        - The images of all ordinary light profiles in that plane summed and convolved with the imaging data's PSF.
        - The images of all linear objects (e.g. linear light profiles / pixelizations), where the images are solved
        for first via the inversion.

        For modeling, this dictionary is used to set up the `hyper_images` that adapt certain pixelizations to the
        data being fitted.
        """

        galaxy_blurred_image_2d_dict = self.plane.galaxy_blurred_image_2d_dict_from(
            grid=self.grid,
            convolver=self.imaging.convolver,
            blurring_grid=self.imaging.blurring_grid,
        )

        return {**galaxy_blurred_image_2d_dict}

    @property
    def model_images_of_galaxies_list(self) -> List:
        """
        A list of the model images of each galaxy in the plane.
        """
        return list(self.galaxy_model_image_dict.values())

    @property
    def subtracted_images_of_galaxies_list(self) -> List[aa.Array2D]:
        """
        A list of the subtracted image of every galaxy.

        A subtracted image of a galaxy is the data where all other galaxy images are subtracted from it, therefore
        showing how a galaxy appears in the data in the absence of all other galaxies.

        This is used to visualize the contribution of each galaxy in the data.
        """

        subtracted_images_of_galaxies_list = []

        model_images_of_galaxies_list = self.model_images_of_galaxies_list

        for galaxy_index in range(len(self.galaxies)):

            other_galaxies_model_images = [
                model_image
                for i, model_image in enumerate(model_images_of_galaxies_list)
                if i != galaxy_index
            ]

            subtracted_image = self.image - sum(other_galaxies_model_images)

            subtracted_images_of_galaxies_list.append(subtracted_image)

        return subtracted_images_of_galaxies_list

    @property
    def unmasked_blurred_image(self) -> aa.Array2D:
        """
        The blurred image of the overall fit that would be evaluated without a mask being used.

        Linear objects are tied to the mask defined to used to perform the fit, therefore their unmasked blurred
        image cannot be computed.
        """
        if self.plane.has(cls=LightProfileLinear):
            exc.raise_linear_light_profile_in_unmasked()

        return self.plane.unmasked_blurred_image_2d_from(
            grid=self.grid, psf=self.imaging.psf
        )

    @property
    def unmasked_blurred_image_of_galaxies_list(self) -> List[aa.Array2D]:
        """
        The blurred image of every galaxy int he plane used in this fit, that would be evaluated without a mask being
        used.

        Linear objects are tied to the mask defined to used to perform the fit, therefore their unmasked blurred
        image cannot be computed.
        """
        if self.plane.has(cls=LightProfileLinear):
            exc.raise_linear_light_profile_in_unmasked()

        return self.plane.unmasked_blurred_image_2d_list_from(
            grid=self.grid, psf=self.imaging.psf
        )
