"""
This code a copy of the following lenstronomy submodules, with some functions removed:
- https://github.com/lenstronomy/lenstronomy/blob/main/lenstronomy/LensModel/Profiles/shear.py
- https://github.com/lenstronomy/lenstronomy/blob/main/lenstronomy/Util/py

All credits go to Simon Birrer and lenstronomy contributors.
"""

import autoarray as aa

__author__ = "sibirrer, Lenstronomy contributors"

import numpy as np

__all__ = ["ShearGammaPsi", "Shear"]

from autogalaxy.profiles.mass.abstract.abstract import MassProfile

class ShearEuclid(MassProfile):
    # def __init__(self, gamma_1: float = 0.0, gamma_2: float = 0.0):
    #     """
    #     An `ExternalShear` term, to model the line-of-sight contribution of other galaxies / satellites.
    #
    #     The shear angle is defined in the direction of stretching of the image. Therefore, if an object located \
    #     outside the lens is responsible for the shear, it will be offset 90 degrees from the value of angle.
    #
    #     Parameters
    #     ----------
    #     gamma
    #     """
    #
    #     super().__init__(centre=(0.0, 0.0), ell_comps=(0.0, 0.0))
    #     self.gamma_1 = gamma_1
    #     self.gamma_2 = gamma_2

    def __init__(self, gamma_ext: float = 0.0, phi_ext: float = 0.0):
        """
        An `ExternalShear` term, to model the line-of-sight contribution of other galaxies / satellites.

        The shear angle is defined in the direction of stretching of the image. Therefore, if an object located \
        outside the lens is responsible for the shear, it will be offset 90 degrees from the value of angle.

        Parameters
        ----------
        gamma
        """
        self.gamma_ext = gamma_ext
        self.phi_ext = phi_ext

    @aa.grid_dec.to_array
    def convergence_2d_from(self, grid, **kwargs):
        return np.zeros(shape=grid.shape[0])

    @aa.grid_dec.to_array
    def potential_2d_from(self, grid, **kwargs):
        return np.zeros(shape=grid.shape[0])

    @aa.grid_dec.to_vector_yx
    def deflections_yx_2d_from(self, grid, **kwargs):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """

        profile = ShearGammaPsi()

        x = grid[:, 1]
        y = grid[:, 0]

        deflections_x, deflections_y = profile.derivatives(
            x,
            y,
            self.gamma_ext,
            self.phi_ext,
            ra_0=0.0,
            dec_0=0.0,
        )

        return np.stack((deflections_y, deflections_x), axis=-1)

def cart2polar(x, y, center_x=0, center_y=0):
    """Transforms cartesian coords [x,y] into polar coords [r,phi] in the frame of the
    lens center.

    :param x: set of x-coordinates
    :type x: array of size (n)
    :param y: set of x-coordinates
    :type y: array of size (n)
    :param center_x: rotation point
    :type center_x: float
    :param center_y: rotation point
    :type center_y: float
    :returns: array of same size with coords [r,phi]
    """
    coord_shift_x = x - center_x
    coord_shift_y = y - center_y
    r = np.sqrt(coord_shift_x ** 2 + coord_shift_y ** 2)
    phi = np.arctan2(coord_shift_y, coord_shift_x)
    return r, phi


def shear_polar2cartesian(phi, gamma):
    """

    :param phi: shear angle (radian)
    :param gamma: shear strength
    :return: shear components gamma1, gamma2
    """
    gamma1 = gamma * np.cos(2 * phi)
    gamma2 = gamma * np.sin(2 * phi)
    return gamma1, gamma2


class ShearGammaPsi(object):
    """
    class to model a shear field with shear strength and direction. The translation ot the cartesian shear distortions
    is as follow:

    .. math::
        \\gamma_1 = \\gamma_{ext} \\cos(2 \\phi_{ext})
        \\gamma_2 = \\gamma_{ext} \\sin(2 \\phi_{ext})

    """

    param_names = ["gamma_ext", "psi_ext", "ra_0", "dec_0"]
    lower_limit_default = {
        "gamma_ext": 0,
        "psi_ext": -np.pi,
        "ra_0": -100,
        "dec_0": -100,
    }
    upper_limit_default = {"gamma_ext": 1, "psi_ext": np.pi, "ra_0": 100, "dec_0": 100}

    def __init__(self):
        self._shear_e1e2 = Shear()
        super(ShearGammaPsi, self).__init__()

    @staticmethod
    def function(x, y, gamma_ext, psi_ext, ra_0=0, dec_0=0):
        """

        :param x: x-coordinate (angle)
        :param y: y0-coordinate (angle)
        :param gamma_ext: shear strength
        :param psi_ext: shear angle (radian)
        :param ra_0: x/ra position where shear deflection is 0
        :param dec_0: y/dec position where shear deflection is 0
        :return:
        """
        # change to polar coordinate
        r, phi = cart2polar(x - ra_0, y - dec_0)
        f_ = 1.0 / 2 * gamma_ext * r ** 2 * np.cos(2 * (phi - psi_ext))
        return f_

    def derivatives(self, x, y, gamma_ext, psi_ext, ra_0=0, dec_0=0):
        # rotation angle
        gamma1, gamma2 = shear_polar2cartesian(psi_ext, gamma_ext)
        return self._shear_e1e2.derivatives(x, y, gamma1, gamma2, ra_0, dec_0)

    def hessian(self, x, y, gamma_ext, psi_ext, ra_0=0, dec_0=0):
        gamma1, gamma2 = shear_polar2cartesian(psi_ext, gamma_ext)
        return self._shear_e1e2.hessian(x, y, gamma1, gamma2, ra_0, dec_0)


class Shear(object):
    """Class for external shear gamma1, gamma2 expression."""

    param_names = ["gamma1", "gamma2", "ra_0", "dec_0"]
    lower_limit_default = {"gamma1": -0.5, "gamma2": -0.5, "ra_0": -100, "dec_0": -100}
    upper_limit_default = {"gamma1": 0.5, "gamma2": 0.5, "ra_0": 100, "dec_0": 100}

    def function(self, x, y, gamma1, gamma2, ra_0=0, dec_0=0):
        """

        :param x: x-coordinate (angle)
        :param y: y0-coordinate (angle)
        :param gamma1: shear component
        :param gamma2: shear component
        :param ra_0: x/ra position where shear deflection is 0
        :param dec_0: y/dec position where shear deflection is 0
        :return: lensing potential
        """
        x_ = x - ra_0
        y_ = y - dec_0
        f_ = 1 / 2.0 * (gamma1 * x_ * x_ + 2 * gamma2 * x_ * y_ - gamma1 * y_ * y_)
        return f_

    def derivatives(self, x, y, gamma1, gamma2, ra_0=0, dec_0=0):
        """

        :param x: x-coordinate (angle)
        :param y: y0-coordinate (angle)
        :param gamma1: shear component
        :param gamma2: shear component
        :param ra_0: x/ra position where shear deflection is 0
        :param dec_0: y/dec position where shear deflection is 0
        :return: deflection angles
        """
        x_ = x - ra_0
        y_ = y - dec_0
        f_x = gamma1 * x_ + gamma2 * y_
        f_y = +gamma2 * x_ - gamma1 * y_
        return f_x, f_y

    def hessian(self, x, y, gamma1, gamma2, ra_0=0, dec_0=0):
        """

        :param x: x-coordinate (angle)
        :param y: y0-coordinate (angle)
        :param gamma1: shear component
        :param gamma2: shear component
        :param ra_0: x/ra position where shear deflection is 0
        :param dec_0: y/dec position where shear deflection is 0
        :return: f_xx, f_xy, f_yx, f_yy
        """
        gamma1 = gamma1
        gamma2 = gamma2
        kappa = 0
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_xy, f_xy, f_yy
