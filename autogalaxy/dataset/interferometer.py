import numpy as np
from autoarray.structures import grids
from autoarray.dataset import interferometer
from autoarray.operators import transformer
from autogalaxy import exc
from autogalaxy.plane import plane as pl


class MaskedInterferometer(interferometer.MaskedInterferometer):
    def __init__(
        self,
        interferometer,
        visibilities_mask,
        real_space_mask,
        grid_class=grids.Grid,
        grid_inversion_class=grids.Grid,
        fractional_accuracy=0.9999,
        sub_steps=None,
        transformer_class=transformer.TransformerNUFFT,
        primary_beam_shape_2d=None,
        inversion_pixel_limit=None,
        renormalize_primary_beam=True,
    ):
        """
        The lens dataset is the collection of data (image, noise map, primary_beam), a mask, grid, convolver \
        and other utilities that are used for modeling and fitting an image of a strong lens.

        Whilst the image, noise map, etc. are loaded in 2D, the lens dataset creates reduced 1D arrays of each \
        for lens calculations.

        Parameters
        ----------
        imaging: im.Imaging
            The imaging data all in 2D (the image, noise map, primary_beam, etc.)
        real_space_mask: msk.Mask
            The 2D mask that is applied to the image.
        sub_size : int
            The size of the sub-grid used for each lens SubGrid. E.g. a value of 2 grid each image-pixel on a 2x2 \
            sub-grid.
        primary_beam_shape_2d : (int, int)
            The shape of the primary_beam used for convolving model image generated using analytic light profiles. A smaller \
            shape will trim the primary_beam relative to the input image primary_beam, giving a faster analysis run-time.
        positions : [[]]
            Lists of image-pixel coordinates (arc-seconds) that mappers close to one another in the source-plane(s), \
            used to speed up the non-linear sampling.
        pixel_scales_interp : float
            If *True*, expensive to compute mass profile deflection angles will be computed on a sparse grid and \
            interpolated to the grid, sub and blurring grids.
        inversion_pixel_limit : int or None
            The maximum number of pixels that can be used by an inversion, with the limit placed primarily to speed \
            up run.
        """

        super(MaskedInterferometer, self).__init__(
            interferometer=interferometer,
            visibilities_mask=visibilities_mask,
            real_space_mask=real_space_mask,
            grid_class=grid_class,
            grid_inversion_class=grid_inversion_class,
            fractional_accuracy=fractional_accuracy,
            sub_steps=sub_steps,
            transformer_class=transformer_class,
            primary_beam_shape_2d=primary_beam_shape_2d,
            inversion_pixel_limit=inversion_pixel_limit,
            renormalize_primary_beam=renormalize_primary_beam,
        )

    def check_inversion_pixels_are_below_limit_via_plane(self, plane):

        if self.inversion_pixel_limit is not None:
            if plane.has_pixelization:
                if plane.pixelization.pixels > self.inversion_pixel_limit:
                    raise exc.PixelizationException


class SimulatorInterferometer(interferometer.SimulatorInterferometer):
    def __init__(
        self,
        uv_wavelengths,
        exposure_time_map,
        background_sky_map,
        transformer_class=transformer.TransformerDFT,
        primary_beam=None,
        renormalize_primary_beam=True,
        noise_sigma=0.1,
        noise_if_add_noise_false=0.1,
        noise_seed=-1,
    ):
        """A class representing a Imaging observation, using the shape of the image, the pixel scale,
        psf, exposure time, etc.

        Parameters
        ----------
        shape_2d : (int, int)
            The shape of the observation. Note that we do not simulator a full Imaging frame (e.g. 2000 x 2000 pixels for \
            Hubble imaging), but instead just a cut-out around the strong lens.
        pixel_scales : float
            The size of each pixel in arc seconds.
        psf : PSF
            An arrays describing the PSF kernel of the image.
        exposure_time_map : float
            The exposure time of an observation using this data.
        background_sky_map : float
            The level of the background sky of an observationg using this data.
        """

        super(SimulatorInterferometer, self).__init__(
            uv_wavelengths=uv_wavelengths,
            exposure_time_map=exposure_time_map,
            background_sky_map=background_sky_map,
            transformer_class=transformer_class,
            primary_beam=primary_beam,
            renormalize_primary_beam=renormalize_primary_beam,
            noise_sigma=noise_sigma,
            noise_if_add_noise_false=noise_if_add_noise_false,
            noise_seed=noise_seed,
        )

    def from_plane_and_grid(self, plane, grid, name=None):
        """
        Create a realistic simulated image by applying effects to a plain simulated image.

        Parameters
        ----------
        name
        image : ndarray
            The image before simulating (e.g. the lens and source galaxies before optics blurring and Imaging read-out).
        pixel_scales: float
            The scale of each pixel in arc seconds
        exposure_time_map : ndarray
            An arrays representing the effective exposure time of each pixel.
        psf: PSF
            An arrays describing the PSF the simulated image is blurred with.
        background_sky_map : ndarray
            The value of background sky in every image pixel (electrons per second).
        add_noise: Bool
            If True poisson noise_maps is simulated and added to the image, based on the total counts in each image
            pixel
        noise_seed: int
            A seed for random noise_maps generation
        """

        image = plane.image_from_grid(grid=grid)

        return self.from_image(image=image.in_1d_binned, name=name)

    def from_galaxies_and_grid(self, galaxies, grid, name=None):
        """Simulate imaging data for this data, as follows:

        1)  Setup the image-plane grid of the Imaging arrays, which defines the coordinates used for the ray-tracing.

        2) Use this grid and the lens and source galaxies to setup a plane, which generates the image of \
           the simulated imaging data.

        3) Simulate the imaging data, using a special image which ensures edge-effects don't
           degrade simulator of the telescope optics (e.g. the PSF convolution).

        4) Plot the image using Matplotlib, if the plot_imaging bool is True.

        5) Output the dataset to .fits format if a dataset_path and data_name are specified. Otherwise, return the simulated \
           imaging data instance."""

        plane = pl.Plane(
            redshift=float(np.mean([galaxy.redshift for galaxy in galaxies])),
            galaxies=galaxies,
        )

        return self.from_plane_and_grid(plane=plane, grid=grid, name=name)
