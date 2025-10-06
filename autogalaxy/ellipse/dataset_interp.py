import numpy as np
from typing import Tuple

from autoconf import cached_property

import autoarray as aa


class DatasetInterp:
    def __init__(self, dataset: aa.Imaging):
        """
        An ellipse interpolator, which contains a dataset (e.g. the image data and noise-map) and performs interpo.aiton
        calculations used for ellipse fitting.

        This object is used by the input to the `FitEllipse` object, which fits the dataset with ellipses and quantifies
        the goodness-of-fit via a residual map, likelihood, chi-squared and other quantities.

        The following quantities of the ellipse data are interpolated and used for the following tasks:

        - `data`: The image data, which shows the signal that is analysed and fitted with ellipses.

        - `noise_map`: The RMS standard deviation error in every pixel, which is used to compute the chi-squared value
        and likelihood of a fit.

        The `data` and `noise_map` are typically the same images of a galaxy used to perform standard light-profile
        fitting.

        Parameters
        ----------
        dataset
            The imaging data, containing the image data, noise map.
        """
        self.dataset = dataset

    @cached_property
    def points_interp(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        The points on which the interpolation from the 2D grid of data is performed.
        """

        x = self.dataset.mask.derive_grid.all_false.native[0, :, 1]
        y = np.flip(self.dataset.mask.derive_grid.all_false.native[:, 0, 0])

        return (x, y)

    @cached_property
    def mask_interp(self) -> "interpolate.RegularGridInterpolator":
        """
        Returns a 2D interpolation of the mask, which is used to determine whether inteprolated values use a masked
        pixel for the interpolation and thus should not be included in a fit.
        """
        from scipy import interpolate

        return interpolate.RegularGridInterpolator(
            points=self.points_interp,
            values=np.float64(self.dataset.data.mask),
            bounds_error=False,
            fill_value=0.0,
        )

    @cached_property
    def data_interp(self) -> "interpolate.RegularGridInterpolator":
        """
        Returns a 2D interpolation of the data, which is used to evaluate the data at any point in 2D space.
        """
        from scipy import interpolate

        return interpolate.RegularGridInterpolator(
            points=self.points_interp,
            values=np.float64(self.dataset.data.native),
            bounds_error=False,
            fill_value=0.0,
        )

    @cached_property
    def noise_map_interp(self) -> "interpolate.RegularGridInterpolator":
        """
        Returns a 2D interpolation of the noise-map, which is used to evaluate the noise-map at any point in 2D space.
        """
        from scipy import interpolate

        return interpolate.RegularGridInterpolator(
            points=self.points_interp,
            values=np.float64(self.dataset.noise_map.native),
            bounds_error=False,
            fill_value=0.0,
        )
