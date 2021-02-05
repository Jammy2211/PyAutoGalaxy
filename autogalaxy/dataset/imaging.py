import copy

import numpy as np
from autoarray.structures import arrays
from autoarray.structures import grids
from autoarray.structures import kernel
from autoarray.dataset import imaging
from autogalaxy.plane import plane as pl


class SettingsMaskedImaging(imaging.SettingsMaskedImaging):
    def __init__(
        self,
        grid_class=grids.Grid2D,
        grid_inversion_class=grids.Grid2D,
        sub_size=2,
        fractional_accuracy=0.9999,
        sub_steps=None,
        pixel_scales_interp=None,
        bin_up_factor=None,
        signal_to_noise_limit=None,
        psf_shape_2d=None,
        renormalize_psf=True,
    ):
        """
        The lens dataset is the collection of data_type (image, noise-map, PSF), a mask, grid, convolver \
        and other utilities that are used for modeling and fitting an image of a strong lens.

        Whilst the image, noise-map, etc. are loaded in 2D, the lens dataset creates reduced 1D arrays of each \
        for lens calculations.

        Parameters
        ----------
        grid_class : ag.Grid2D
            The type of grid used to create the image from the `Galaxy` and `Plane`. The options are `Grid2D`,
            `Grid2DIterate` and `Grid2DInterpolate` (see the `Grid2D` documentation for a description of these options).
        grid_inversion_class : ag.Grid2D
            The type of grid used to create the grid that maps the `Inversion` source pixels to the data's image-pixels.
            The options are `Grid2D`, `Grid2DIterate` and `Grid2DInterpolate` (see the `Grid2D` documentation for a
            description of these options).
        sub_size : int
            If the grid and / or grid_inversion use a `Grid2D`, this sets the sub-size used by the `Grid2D`.
        fractional_accuracy : float
            If the grid and / or grid_inversion use a `Grid2DIterate`, this sets the fractional accuracy it
            uses when evaluating functions.
        sub_steps : [int]
            If the grid and / or grid_inversion use a `Grid2DIterate`, this sets the steps the sub-size is increased by
            to meet the fractional accuracy when evaluating functions.
        pixel_scales_interp : float or (float, float)
            If the grid and / or grid_inversion use a `Grid2DInterpolate`, this sets the resolution of the interpolation
            grid.
        signal_to_noise_limit : float
            If input, the dataset's noise-map is rescaled such that no pixel has a signal-to-noise above the
            signa to noise limit.
        """

        super().__init__(
            grid_class=grid_class,
            grid_inversion_class=grid_inversion_class,
            sub_size=sub_size,
            fractional_accuracy=fractional_accuracy,
            sub_steps=sub_steps,
            pixel_scales_interp=pixel_scales_interp,
            bin_up_factor=bin_up_factor,
            signal_to_noise_limit=signal_to_noise_limit,
            psf_shape_2d=psf_shape_2d,
            renormalize_psf=renormalize_psf,
        )


class MaskedImaging(imaging.MaskedImaging):
    def __init__(self, imaging, mask, settings=SettingsMaskedImaging()):
        """
        The lens dataset is the collection of data (image, noise-map, PSF), a mask, grid, convolver \
        and other utilities that are used for modeling and fitting an image of a strong lens.

        Whilst the image, noise-map, etc. are loaded in 2D, the lens dataset creates reduced 1D arrays of each \
        for lens calculations.

        Parameters
        ----------
        imaging: im.Imaging
            The imaging data all in 2D (the image, noise-map, PSF, etc.)
        mask: msk.Mask2D
            The 2D mask that is applied to the image.
        sub_size : int
            The size of the sub-grid used for each lens SubGrid. E.g. a value of 2 grid each image-pixel on a 2x2 \
            sub-grid.
        psf_shape_2d : (int, int)
            The shape of the PSF used for convolving model image generated using analytic light profiles. A smaller \
            shape will trim the PSF relative to the input image PSF, giving a faster analysis run-time.
        positions : [[]]
            Lists of image-pixel coordinates (arc-seconds) that mappers close to one another in the source-plane(s), \
            used to speed up the non-linear sampling.
        pixel_scales_interp : float
            If `True`, expensive to compute mass profile deflection angles will be computed on a sparse grid and \
            interpolated to the grid, sub and blurring grids.
        """

        super(MaskedImaging, self).__init__(
            imaging=imaging, mask=mask, settings=settings
        )


class SimulatorImaging(imaging.SimulatorImaging):
    def __init__(
        self,
        exposure_time: float,
        background_sky_level: float = 0.0,
        psf: kernel.Kernel2D = None,
        renormalize_psf: bool = True,
        read_noise: float = None,
        add_poisson_noise: bool = True,
        noise_if_add_noise_false: float = 0.1,
        noise_seed: int = -1,
    ):
        """A class representing a Imaging observation, using the shape of the image, the pixel scale,
        psf, exposure time, etc.

        Parameters
        ----------
        psf : Kernel2D
            An arrays describing the PSF kernel of the image.
        exposure_time : float
            The exposure time of the simulated imaging.
        background_sky_level : float
            The level of the background sky of the simulated imaging.
        renormalize_psf : bool
            If `True`, the PSF kernel is renormalized so all values sum to 1.0.
        read_noise : float
            The level of read-noise added to the simulated imaging by drawing from a Gaussian distribution with
            sigma equal to the value `read_noise`.
        add_poisson_noise : bool
            Whether Poisson noise corresponding to photon count statistics on the imaging observation is added.
        noise_if_add_noise_false : float
            If noise is not added to the simulated dataset a `noise_map` must still be returned. This value gives
            the value of noise assigned to every pixel in the noise-map.
        noise_seed : int
            The random seed used to add random noise, where -1 corresponds to a random seed every run.

        """

        super(SimulatorImaging, self).__init__(
            psf=psf,
            exposure_time=exposure_time,
            background_sky_level=background_sky_level,
            renormalize_psf=renormalize_psf,
            read_noise=read_noise,
            add_poisson_noise=add_poisson_noise,
            noise_if_add_noise_false=noise_if_add_noise_false,
            noise_seed=noise_seed,
        )

    def from_plane_and_grid(self, plane, grid, name=None):
        """
        Returns a realistic simulated image by applying effects to a plain simulated image.

        Parameters
        ----------
        """

        image = plane.padded_image_from_grid_and_psf_shape(
            grid=grid, psf_shape_2d=self.psf.shape_native
        )

        imaging = self.from_image(image=image.slim_binned, name=name)

        return imaging.trimmed_after_convolution_from(
            kernel_shape=self.psf.shape_native
        )

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
