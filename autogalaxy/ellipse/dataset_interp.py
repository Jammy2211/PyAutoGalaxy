import numpy as np
from pathlib import Path
from scipy import interpolate
from typing import Tuple, Union

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

        y = np.arange(self.dataset.shape_native[0])
        x = np.arange(self.dataset.shape_native[1])

        return (x, y)

    @cached_property
    def data_interp(self) -> interpolate.RegularGridInterpolator:
        """
        Returns a 2D interpolation of the data, which is used to evaluate the data at any point in 2D space.
        """

        y = np.arange(self.dataset.data.shape_native[0])
        x = np.arange(self.dataset.data.shape_native[1])

        return interpolate.RegularGridInterpolator(
            points=(x, y), values=self.dataset.data.native, bounds_error=False, fill_value=0.0
        )

    @cached_property
    def noise_map_interp(self) -> interpolate.RegularGridInterpolator:
        """
        Returns a 2D interpolation of the noise-map, which is used to evaluate the noise-map at any point in 2D space.
        """

        y = np.arange(self.dataset.noise_map.shape_native[0])
        x = np.arange(self.dataset.noise_map.shape_native[1])

        return interpolate.RegularGridInterpolator(
            points=(x, y),
            values=self.dataset.noise_map.native,
            bounds_error=False,
            fill_value=0.0,
        )
