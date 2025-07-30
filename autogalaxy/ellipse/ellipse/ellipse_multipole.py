import numpy as np
from typing import Tuple
from autogalaxy.convert import multipole_comps_from, multipole_k_m_and_phi_m_from


from autogalaxy.ellipse.ellipse.ellipse import Ellipse


class EllipseMultipole:
    def __init__(
        self,
        m=4,
        multipole_comps: Tuple[float, float] = (0.0, 0.0),
    ):
        """
        class representing the multipole of an ellispe with, which is used to perform ellipse fitting to
        2D data (e.g. an image).

        The multipole is added to the (y,x) coordinates of an ellipse that are already computed via the `Ellipse` class.

        The addition of the multipole is performed as follows:

        :math: r_m = \sum_{i=1}^{m} \left( a_i \cos(i(\theta - \phi)) + b_i \sin(i(\theta - \phi)) \right)
        :math: y_m = r_m \sin(\theta)
        :math: x_m = r_m \cos(\theta)

        Where:

        m = The order of the multipole.
        r = The radial coordinate of the ellipse perturbed by the multipole.
        \phi = The angle of the ellipse.
        a = The amplitude of the cosine term of the multipole.
        b = The amplitude of the sine term of the multipole.
        y = The y-coordinate of the ellipse perturbed by the multipole.
        x = The x-coordinate of the ellipse perturbed by the multipole.
        """

        self.m = m
        self.multipole_comps = multipole_comps

    def points_perturbed_from(
        self, pixel_scale, points, ellipse: Ellipse, n_i: int = 0
    ) -> np.ndarray:
        """
        Returns the (y,x) coordinates of the input points, which are perturbed by the multipole of the ellipse.

        Parameters
        ----------
        pixel_scale
            The pixel scale of the data that the ellipse is fitted to and interpolated over.
        points
            The (y,x) coordinates of the ellipse that are perturbed by the multipole.
        ellipse
            The ellipse that is perturbed by the multipole, which is used to compute the angles of the ellipse.

        Returns
        -------
        The (y,x) coordinates of the input points, which are perturbed by the multipole.
        """

        angles = ellipse.angles_from_x0_from(pixel_scale=pixel_scale, n_i=n_i)

        radial = np.add(
            self.multipole_comps[1] * np.cos(self.m * (angles - ellipse.angle_radians)),
            self.multipole_comps[0] * np.sin(self.m * (angles - ellipse.angle_radians)),
        )

        x = points[:, 1] + (radial * np.cos(angles))
        y = points[:, 0] + (radial * np.sin(angles))

        return np.stack(arrays=(y, x), axis=-1)

class EllipseMultipoleRelative(EllipseMultipole):
    def __init__(
        self,
        m=4,
        input_multipole_comps: Tuple[float, float] = (0.0, 0.0),
        major_axis=1.,
    ):

        k, phi = multipole_k_m_and_phi_m_from(multipole_comps=input_multipole_comps, m=m)
        k_adjusted = k*major_axis

        adjusted_multipole_comps = multipole_comps_from(k_adjusted, phi, m)

        super().__init__(m, adjusted_multipole_comps)

        self.adjusted_multipole_comps = adjusted_multipole_comps
        self.m = m

    def points_perturbed_from(
        self, pixel_scale, points, ellipse: Ellipse, n_i: int = 0
    ) -> np.ndarray:
        """
        Returns the (y,x) coordinates of the input points, which are perturbed by the multipole of the ellipse.

        Parameters
        ----------
        pixel_scale
            The pixel scale of the data that the ellipse is fitted to and interpolated over.
        points
            The (y,x) coordinates of the ellipse that are perturbed by the multipole.
        ellipse
            The ellipse that is perturbed by the multipole, which is used to compute the angles of the ellipse.

        Returns
        -------
        The (y,x) coordinates of the input points, which are perturbed by the multipole.
        """

        angles = ellipse.angles_from_x0_from(pixel_scale=pixel_scale, n_i=n_i)

        radial = np.add(
            self.adjusted_multipole_comps[1] * np.cos(self.m * (angles - ellipse.angle_radians)),
            self.adjusted_multipole_comps[0] * np.sin(self.m * (angles - ellipse.angle_radians)),
        )

        x = points[:, 1] + (radial * np.cos(angles))
        y = points[:, 0] + (radial * np.sin(angles))

        return np.stack(arrays=(y, x), axis=-1)