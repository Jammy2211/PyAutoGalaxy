import numpy as np
from typing import Tuple

from autogalaxy.profiles.geometry_profiles import EllProfile


class Ellipse(EllProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        major_axis: float = 1.0,
    ):
        """
        class representing an ellispe, which is used to perform ellipse fitting to 2D data (e.g. an image).

        The elliptical components (`ell_comps`) of this profile are used to define the `axis_ratio` (q)
        and `angle` (\phi) :

        :math: \phi = (180/\pi) * arctan2(e_y / e_x) / 2
        :math: f = sqrt(e_y^2 + e_x^2)
        :math: q = (1 - f) / (1 + f)

        Where:

        e_y = y elliptical component = `ell_comps[0]`
        e_x = x elliptical component = `ell_comps[1]`
        q = axis_ratio (major_axis / minor_axis)

        This means that given an axis-ratio and angle the elliptical components can be computed as:

        :math: f = (1 - q) / (1 + q)
        :math: e_y = f * sin(2*\phi)
        :math: e_x = f * cos(2*\phi)
        """

        super().__init__(centre=centre, ell_comps=ell_comps)
        self.major_axis = major_axis

    @property
    def circular_radius(self) -> float:
        """
        The radius of the circle that bounds the ellipse, assuming that the `major_axis` is the radius of the circle.
        """
        return 2.0 * np.pi * np.sqrt((2.0 * self.major_axis**2.0) / 2.0)

    @property
    def eccentricity(self) -> float:
        """
        The ellipticity of the ellipse, which is the factor by which the ellipse is offset from a circle.
        """
        return (1.0 - self.axis_ratio) / (1.0 + self.axis_ratio)

    @property
    def minor_axis(self):
        """
        The minor-axis of the ellipse for a given major-axis, computed as:

        :math: b = a * sqrt(1 - e^2)

        Where:

        a = major-axis
        b = minor-axis
        e = eccentricity

        Parameters
        ----------
        major_axis
            The major-axis of the ellipse.

        Returns
        -------
        float
            The minor-axis of the ellipse.
        """

        return self.major_axis * np.sqrt(1.0 - self.eccentricity**2.0)

    @property
    def ellipse_radii_from_major_axis(self):
        return np.divide(
            self.major_axis * self.major_axis,
            np.sqrt(
                np.add(
                    self.major_axis ** 2.0 * np.sin(self.angles_from_x0 - self.angle) ** 2.0,
                    self.minor_axis ** 2.0 * np.cos(self.angles_from_x0 - self.angle) ** 2.0,
                )
            ),
        )

    @property
    def angles_from_x0(self) -> np.ndarray:
        n = np.min([500, int(self.circular_radius)])

        return np.linspace(0.0, 2.0 * np.pi, n)

    @property
    def x_from_major_axis(self):
        return self.ellipse_radii_from_major_axis * np.cos(self.angles_from_x0) + self.centre[1]

    @property
    def y_from_major_axis(self):
        return self.ellipse_radii_from_major_axis * np.sin(self.angles_from_x0) + self.centre[0]

    @property
    def points_from_major_axis(self):

        x = self.x_from_major_axis
        y = self.y_from_major_axis

        idx = np.logical_or(np.isnan(x), np.isnan(y))
        if np.sum(idx) > 0.0:
            raise NotImplementedError()

        return np.stack(arrays=(y, x), axis=-1)
