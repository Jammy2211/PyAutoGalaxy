import numpy as np
from scipy import interpolate
from typing import Tuple

from autoconf import cached_property

import autoarray as aa

class DatasetEllipse:

    def __init__(self, data : aa.Array2D, noise_map : aa.Array2D, radii_min : float, radii_max : float, radii_bins : int):
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
        self.radii_min = radii_min
        self.radii_max = radii_max
        self.radii_bins = radii_bins
        self.mask = data.mask

    @property
    def radii_array(self) -> np.ndarray:
        """
        Returns the array of radii values that the isophotes are fitted to the data at, which are spaced in log10
        intervals between the minimum and maximum radii values.

        Returns
        -------
        The array of radii values that the isophotes are fitted to the data at.
        """
        return np.logspace(np.log10(self.radii_min), np.log10(self.radii_max), self.radii_bins)

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
                points=(x, y),
                values=self.data.native,
                bounds_error=False,
                fill_value=0.0
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
                fill_value=0.0
            )