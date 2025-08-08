import numpy as np
from typing import List, Optional

from autoconf import cached_property

import autoarray as aa

from autogalaxy.ellipse.dataset_interp import DatasetInterp
from autogalaxy.ellipse.ellipse.ellipse import Ellipse
from autogalaxy.ellipse.ellipse.ellipse_multipole import EllipseMultipole


class FitEllipse(aa.FitDataset):
    def __init__(
        self,
        dataset: aa.Imaging,
        ellipse: Ellipse,
        multipole_list: Optional[List[EllipseMultipole]] = None,
    ):
        """
        A fit to a `DatasetInterp` dataset, using a model image to represent the observed data and noise-map.

        Parameters
        ----------
        dataset
            The dataset containing the signal and noise-map that is fitted.

        """
        super().__init__(dataset=dataset)

        self.ellipse = ellipse
        self.multipole_list = multipole_list

    @cached_property
    def interp(self) -> DatasetInterp:
        """
        Returns a class which handles the interpolation of values from the image data and noise-map, so that they
        can be mapped to each ellipse for the fit.
        """
        return DatasetInterp(dataset=self.dataset)

    def points_from_major_axis_from(self) -> np.ndarray:
        """
        Returns the (y,x) coordinates on the ellipse that are used to interpolate the data and noise-map values.

        These points are computed by overlaying the ellipse over the 2D data and noise-map and computing the (y,x)
        coordinates on the ellipse that are closest to the data points.

        If multipole components are used, the points are also perturbed by the multipole components.

        Returns
        -------
        The (y,x) coordinates on the ellipse where the interpolation occurs.
        """
        points = self.ellipse.points_from_major_axis_from(
            pixel_scale=self.dataset.pixel_scales[0],
        )

        if self.multipole_list is not None:
            for multipole in self.multipole_list:
                points = multipole.points_perturbed_from(
                    pixel_scale=self.dataset.pixel_scales[0],
                    points=points,
                    ellipse=self.ellipse,
                )

        if self.interp.mask_interp is not None:

            i_total = 300

            total_points_required = points.shape[0]

            for i in range(1, i_total + 1):

                total_points = points.shape[0]
                total_points_masked = np.sum(self.interp.mask_interp(points) > 0)

                if total_points_required == total_points - total_points_masked:
                    continue

                if total_points_required < total_points - total_points_masked:

                    number_of_extra_points = (
                        total_points - total_points_masked - total_points_required
                    )

                    unmasked_indices = np.where(self.interp.mask_interp(points) == 0)[0]
                    unmasked_indices = unmasked_indices[number_of_extra_points:]

                    points = points[unmasked_indices]

                    continue

                points = self.ellipse.points_from_major_axis_from(
                    pixel_scale=self.dataset.pixel_scales[0], n_i=i
                )

                if self.multipole_list is not None:
                    for multipole in self.multipole_list:

                        points = multipole.points_perturbed_from(
                            pixel_scale=self.dataset.pixel_scales[0],
                            points=points,
                            ellipse=self.ellipse,
                            n_i=i,
                        )

                if i == i_total:

                    raise ValueError(
                        """
                        The code has attempted to add over 300 extra points to the ellipse and still not found a set of points that
                        do not hit the mask with the expected number of points. 
        
                        This is likely due to the mask being too large or a strange geometry, and the code is unable to find a
                        set of points that do not hit the mask.
                        """
                    )

            points = points[self.interp.mask_interp(points) == 0]

        return points

    @cached_property
    def _points_from_major_axis(self) -> np.ndarray:
        """
        Returns cached (y,x) coordinates on the ellipse that are used to interpolate the data and noise-map values.

        Returns
        -------
        The (y,x) coordinates on the ellipse where the interpolation occurs.
        """
        return self.points_from_major_axis_from()

    @property
    def data_interp(self) -> aa.ArrayIrregular:
        """
        Returns the data values of the dataset that the ellipse fits, which are computed by overlaying the ellipse over
        the 2D data and performing a 2D interpolation at discrete (y,x) coordinates on the ellipse.

        The (y,x) coordinates on the ellipse where the interpolation occurs are computed in the
        `points_from_major_axis` property of the `Ellipse` class, with the documentation describing how these points
        are computed.

        If the interpolation of an ellipse point uses one or more masked values, this point is not reliable, therefore
        the value is converted to `np.nan` and not used by other fitting quantities.

        Returns
        -------
        The data values of the ellipse fits, computed via a 2D interpolation of where the ellipse
        overlaps the data.
        """
        data = self.interp.data_interp(self._points_from_major_axis)

        return aa.ArrayIrregular(values=data)

    @property
    def noise_map_interp(self) -> aa.ArrayIrregular:
        """
        Returns the noise-map values of the dataset that the ellipse fits, which are computed by overlaying the ellipse
        over the 2D noise-map and performing a 2D interpolation at discrete (y,x) coordinates on the ellipse.

        The (y,x) coordinates on the ellipse where the interpolation occurs are computed in the
        `points_from_major_axis` property of the `Ellipse` class, with the documentation describing how these points
        are computed.

        If the interpolation of an ellipse point uses one or more masked values, this point is not reliable, therefore
        the value is converted to `np.nan` and not used by other fitting quantities.

        Returns
        -------
        The noise-map values of the ellipse fits, computed via a 2D interpolation of where the ellipse
        overlaps the noise-map.
        """
        noise_map = self.interp.noise_map_interp(self._points_from_major_axis)

        return aa.ArrayIrregular(values=noise_map)

    @property
    def signal_to_noise_map_interp(self) -> aa.ArrayIrregular:
        """
        Returns the signal-to-noise-map of the dataset that the ellipse fits, which is computed by overlaying the ellipse
        over the 2D data and noise-map and performing a 2D interpolation at discrete (y,x) coordinates on the ellipse.

        Returns
        -------
        The signal-to-noise-map values of the ellipse fits, computed via a 2D interpolation of where
        the ellipse overlaps the data and noise-map.
        """
        return aa.ArrayIrregular(values=self.data_interp / self.noise_map_interp)

    @property
    def model_data(self) -> aa.ArrayIrregular:
        """
        Returns the model-data, which is the data values where the ellipse overlaps the data minus the mean
        value of these data values.

        By subtracting the mean of the data values from each data value, the model data quantifies how often there
        are large variations in the data values over the ellipse.

        For example, if every data value subtended by the ellipse are close to one another, the difference between
        the data values and the mean will be small.

        Conversely, if some data values are much higher or lower than the mean, the model data will be large.

        Returns
        -------
        The model data values of the ellipse fit, which are the data values minus the mean of the data values.
        """
        return self.data_interp

    @property
    def residual_map(self) -> aa.ArrayIrregular:
        """
        Returns the residual-map of the fit, which is the data minus the model data and therefore the same
        as the model data.

        Returns
        -------
        The residual-map of the fit, which is the data minus the model data and therefore the same as the model data.
        """

        if not self.model_data:
            return aa.ArrayIrregular(values=np.zeros(self.model_data.shape))

        return aa.ArrayIrregular(values=self.model_data - np.nanmean(self.model_data))

    @property
    def normalized_residual_map(self) -> aa.ArrayIrregular:
        """
        Returns the normalized residual-map of the fit, which is the residual-map divided by the noise-map.

        The residual map and noise map are computed by overlaying the ellipse over the 2D data and noise-map and
        performing a 2D interpolation at discrete (y,x) coordinates on the ellipse. See the documentation of the
        `residual_map` and `noise_map` properties for more details.

        Returns
        -------
        The normalized residual-map of the fit, which is the residual-map divided by the noise-map.
        """
        return aa.ArrayIrregular(values=self.residual_map / self.noise_map_interp)

    @property
    def chi_squared_map(self) -> aa.ArrayIrregular:
        """
        Returns the chi-squared-map of the fit, which is the normalized residual-map squared.

        The normalized residual map is computed by overlaying the ellipse over the 2D data and noise-map and
        performing a 2D interpolation at discrete (y,x) coordinates on the ellipse. See the documentation of the
        `normalized_residual_map` property for more details.

        Returns
        -------
        The chi-squared-map of the fit, which is the normalized residual-map squared.
        """
        return aa.ArrayIrregular(values=self.normalized_residual_map**2.0)

    @property
    def chi_squared(self) -> float:
        """
        The sum of the chi-squared-map, which quantifies how well the model data represents the data and noise-map.

        The chi-squared-map is computed by overlaying the ellipse over the 2D data and noise-map and
        performing a 2D interpolation at discrete (y,x) coordinates on the ellipse. See the documentation of the
        `chi_squared_map` property for more details.

        Returns
        -------
        The chi-squared of the fit.
        """
        return float(np.nansum(self.chi_squared_map))

    @property
    def noise_normalization(self):
        """
        The noise normalization term of the log likelihood, which is the sum of the log noise-map values squared.

        Returns
        -------
        The noise normalization term of the log likelihood.
        """
        return np.nansum(np.log(2 * np.pi * self.noise_map_interp**2.0))

    @property
    def log_likelihood(self):
        """
        The log likelihood of the fit, which quantifies how well the model data represents the data and noise-map.

        Returns
        -------
        The log likelihood of the fit.
        """
        return -0.5 * (self.chi_squared)

    @property
    def figure_of_merit(self) -> float:
        """
        The figure of merit of the fit, which is passed by the `Analysis` class to the non-linear search to
        determine the best-fit solution.

        Returns
        -------
        The figure of merit of the fit.
        """
        return -0.5 * (self.chi_squared + self.noise_normalization)
