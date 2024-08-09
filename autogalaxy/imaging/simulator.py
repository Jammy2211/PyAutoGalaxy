import numpy as np
from typing import List

import autoarray as aa

from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.galaxy.galaxies import Galaxies


class SimulatorImaging(aa.SimulatorImaging):
    def via_galaxies_from(
        self, galaxies: List[Galaxy], grid: aa.type.Grid2DLike
    ) -> aa.Imaging:
        """
        Simulate an `Imaging` dataset from an input plane and grid.

        The planbe is used to generate the image of the galaxies which is simulated.

        The steps of the `SimulatorImaging` simulation process (e.g. PSF convolution, noise addition) are
        described in the `SimulatorImaging` `__init__` method docstring.

        Parameters
        ----------
        galaxies
            The galaxies whose light is simulated.
        grid
            The image-plane grid which the image of the strong lens is generated on.
        """

        galaxies = Galaxies(galaxies=galaxies)

        for galaxy in galaxies:
            galaxy.set_snr_of_snr_light_profiles(
                grid=grid,
                exposure_time=self.exposure_time,
                background_sky_level=self.background_sky_level,
                psf=self.psf,
            )

        image = galaxies.padded_image_2d_from(
            grid=grid, psf_shape_2d=self.psf.shape_native
        )

        dataset = self.via_image_from(image=image)

        return dataset.trimmed_after_convolution_from(
            kernel_shape=self.psf.shape_native
        )
