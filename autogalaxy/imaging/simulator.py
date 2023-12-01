import numpy as np

import autoarray as aa

from autogalaxy.plane.plane import Plane


class SimulatorImaging(aa.SimulatorImaging):
    def via_plane_from(self, plane: Plane, grid: aa.type.Grid2DLike) -> aa.Imaging:
        """
        Simulate an `Imaging` dataset from an input plane and grid.

        The planbe is used to generate the image of the galaxies which is simulated.

        The steps of the `SimulatorImaging` simulation process (e.g. PSF convolution, noise addition) are
        described in the `SimulatorImaging` `__init__` method docstring.

        Parameters
        ----------
        plane
            The plane, which contains all galaxies whose light is simulated.
        grid
            The image-plane grid which the image of the strong lens is generated on.
        """

        plane.set_snr_of_snr_light_profiles(
            grid=grid,
            exposure_time=self.exposure_time,
            background_sky_level=self.background_sky_level,
            psf=self.psf,
        )

        image = plane.padded_image_2d_from(
            grid=grid, psf_shape_2d=self.psf.shape_native
        )

        dataset = self.via_image_from(image=image.binned)

        return dataset.trimmed_after_convolution_from(
            kernel_shape=self.psf.shape_native
        )

    def via_galaxies_from(self, galaxies, grid):
        """
        Simulate an `Imaging` dataset from an input list of galaxies and grid.

        The galaxies are used to create a plane, which generates the image which is simulated.

        The steps of the `SimulatorImaging` simulation process (e.g. PSF convolution, noise addition) are
        described in the `SimulatorImaging` `__init__` method docstring.

        Parameters
        ----------
        galaxies
            The galaxies used to create the tracer, which describes the galaxy images used to simulate the imaging
            dataset.
        grid
            The image-plane grid which the image of the strong lens is generated on.
        """

        plane = Plane(
            redshift=float(np.mean([galaxy.redshift for galaxy in galaxies])),
            galaxies=galaxies,
        )

        return self.via_plane_from(plane=plane, grid=grid)
