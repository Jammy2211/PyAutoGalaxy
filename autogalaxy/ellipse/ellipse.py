from typing import Tuple

from autogalaxy.profiles.geometry_profiles import EllProfile

class Ellipse(EllProfile):

    def __init__(self, centre: Tuple[float, float] = (0.0, 0.0), ell_comps : Tuple[float, float] = (0.0, 0.0)):
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

        For an input (y,x) grid of Cartesian coordinates this is used to compute the elliptical coordinates of a
        profile:

        .. math:: \\xi = q^{0.5} * ((y-y_c^2 + x-x_c^2 / q^2)^{0.5}

        Where:

        y_c = profile y centre = `centre[0]`
        x_c = profile x centre = `centre[1]`

        The majority of elliptical profiles use \\xi to compute their image.
        """

        super().__init__(centre=centre, ell_comps=ell_comps)

    @property
    def ellipticity(self) -> float:
        """
        The ellipticity of the ellipse, which is the factor by which the ellipse is offset from a circle.
        """
        return (1.0 - self.axis_ratio) / (1.0 + self.axis_ratio)

