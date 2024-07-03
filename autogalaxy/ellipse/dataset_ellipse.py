import numpy as np
from pathlib import Path
from scipy import interpolate
from typing import Tuple, Union

from autoconf import cached_property

import autoarray as aa


class DatasetEllipse:
    def __init__(self, data: aa.Array2D, noise_map: aa.Array2D):
        """
        An ellipse dataset, containing the image data, noise-map and associated quantities ellipse fitting
        calculations.

        This object is the input to the `FitEllipse` object, which fits the dataset with ellipses and quantifies
        the goodness-of-fit via a residual map, likelihood, chi-squared and other quantities.

        The following quantities of the ellipse data are available and used for the following tasks:

        - `data`: The image data, which shows the signal that is analysed and fitted with ellipses.

        - `noise_map`: The RMS standard deviation error in every pixel, which is used to compute the chi-squared value
        and likelihood of a fit.

        The `data` and `noise_map` are typically the same images of a galaxy used to perform standard light-profile
        fitting.

        Parameters
        ----------
        data
            The imaging data in 2D, on which elliptical isophotes are fitted.
        noise_map
            The noise-map of the imaging data, which describes the RMS standard deviation in every pixel.
        radii_min
            The minimum circular radius where isophotes are fitted to the data.
        radii_max
            The maximum circular radius where isophotes are fitted to the data.
        radii_bins
            The number of bins between radii_min and radii_max where isophotes are fitted to the data.
        """
        self.data = data
        self.noise_map = noise_map
        self.mask = data.mask

    @cached_property
    def points_interp(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        The points on which the interpolation from the 2D grid of data is performed.
        """

        y = np.arange(self.data.shape_native[0])
        x = np.arange(self.data.shape_native[1])

        return (x, y)

    @cached_property
    def data_interp(self) -> interpolate.RegularGridInterpolator:
        """
        Returns a 2D interpolation of the data, which is used to evaluate the data at any point in 2D space.
        """

        y = np.arange(self.data.shape_native[0])
        x = np.arange(self.data.shape_native[1])

        return interpolate.RegularGridInterpolator(
            points=(x, y), values=self.data.native, bounds_error=False, fill_value=0.0
        )

    @cached_property
    def noise_map_interp(self) -> interpolate.RegularGridInterpolator:
        """
        Returns a 2D interpolation of the noise-map, which is used to evaluate the noise-map at any point in 2D space.
        """

        y = np.arange(self.noise_map.shape_native[0])
        x = np.arange(self.noise_map.shape_native[1])

        return interpolate.RegularGridInterpolator(
            points=(x, y),
            values=self.noise_map.native,
            bounds_error=False,
            fill_value=0.0,
        )

    @classmethod
    def from_fits(
        cls,
        pixel_scales: aa.type.PixelScales,
        data_path: Union[Path, str],
        noise_map_path: Union[Path, str],
        data_hdu: int = 0,
        noise_map_hdu: int = 0,
    ) -> "DatasetEllipse":
        """
        Load an imaging dataset from multiple .fits file.

        For each attribute of the imaging data (e.g. `data`, `noise_map`, `pre_cti_data`) the path to
        the .fits and the `hdu` containing the data can be specified.

        The `noise_map` assumes the noise value in each `data` value are independent, where these values are the
        the RMS standard deviation error in each pixel.

        A `noise_covariance_matrix` can be input instead, which represents the covariance between noise values in
        the data and can be used to fit the data accounting for correlations (the `noise_map` is the diagonal values
        of this matrix).

        If the dataset has a mask associated with it (e.g. in a `mask.fits` file) the file must be loaded separately
        via the `Mask2D` object and applied to the imaging after loading via fits using the `from_fits` method.

        Parameters
        ----------
        pixel_scales
            The (y,x) arcsecond-to-pixel units conversion factor of every pixel. If this is input as a `float`,
            it is converted to a (float, float).
        data_path
            The path to the data .fits file containing the image data (e.g. '/path/to/image.fits').
        data_hdu
            The hdu the image data is contained in the .fits file specified by `data_path`.
        psf_path
            The path to the psf .fits file containing the psf (e.g. '/path/to/psf.fits').
        psf_hdu
            The hdu the psf is contained in the .fits file specified by `psf_path`.
        noise_map_path
            The path to the noise_map .fits file containing the noise_map (e.g. '/path/to/noise_map.fits').
        noise_map_hdu
            The hdu the noise map is contained in the .fits file specified by `noise_map_path`.
        noise_covariance_matrix
            A noise-map covariance matrix representing the covariance between noise in every `data` value.
        over_sampling
            The over sampling schemes which divide the grids into sub grids of smaller pixels within their host image
            pixels when using the grid to evaluate a function (e.g. images) to better approximate the 2D line integral
            This class controls over sampling for all the different grids (e.g. `grid`, `grid_pixelization).
        """

        data = aa.Array2D.from_fits(
            file_path=data_path, hdu=data_hdu, pixel_scales=pixel_scales
        )

        noise_map = aa.Array2D.from_fits(
            file_path=noise_map_path, hdu=noise_map_hdu, pixel_scales=pixel_scales
        )

        return DatasetEllipse(
            data=data,
            noise_map=noise_map,
        )
