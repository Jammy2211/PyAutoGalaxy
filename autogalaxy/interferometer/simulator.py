import numpy as np
import autoarray as aa

from autogalaxy.plane.plane import Plane


class SimulatorInterferometer(aa.SimulatorInterferometer):
    def via_plane_from(self, plane, grid):
        """
        Returns a realistic simulated image by applying effects to a plain simulated image.

        Parameters
        ----------
        image
            The image before simulating (e.g. the lens and source galaxies before optics blurring and Imaging read-out).
        pixel_scales
            The scale of each pixel in arc seconds
        exposure_time_map
            An arrays representing the effective exposure time of each pixel.
        psf: PSF
            An arrays describing the PSF the simulated image is blurred with.
        add_poisson_noise: Bool
            If `True` poisson noise_maps is simulated and added to the image, based on the total counts in each image
            pixel
        noise_seed: int
            A seed for random noise_maps generation
        """

        image = plane.image_2d_from(grid=grid)

        return self.via_image_from(image=image.binned)

    def via_galaxies_from(self, galaxies, grid):
        """Simulate imaging data for this data, as follows:

        1)  Setup the image-plane grid of the Imaging arrays, which defines the coordinates used for the ray-tracing.

        2) Use this grid and the lens and source galaxies to setup a plane, which generates the image of \
           the simulated imaging data.

        3) Simulate the imaging data, using a special image which ensures edge-effects don't
           degrade simulator of the telescope optics (e.g. the PSF convolution).

        4) Plot the image using Matplotlib, if the plot_imaging bool is True.

        5) Output the dataset to .fits format if a dataset_path and data_name are specified. Otherwise, return the simulated \
           imaging data instance."""

        plane = Plane(
            redshift=float(np.mean([galaxy.redshift for galaxy in galaxies])),
            galaxies=galaxies,
        )

        return self.via_plane_from(plane=plane, grid=grid)
