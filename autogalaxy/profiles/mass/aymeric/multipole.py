"""
This module contains functions to compute the lensing potential, deflection angles, and Hessian
for a spherical multipole with radial power-law decay.

Note that only the m=4 case is implemented for the deflection angles.

Credits to Leon Ecker (LMU), based on Chu et al. (2013) and Nightingale et al. (2023).
"""

from autogalaxy.profiles.mass.abstract.abstract import MassProfile

__author__ = "Leon Ecker"

import numpy as np
from typing import Tuple

def cart2polar(x, y, center_x=0, center_y=0):
    coord_shift_x = x - center_x
    coord_shift_y = y - center_y
    r = np.sqrt(coord_shift_x ** 2 + coord_shift_y ** 2)
    phi = np.arctan2(coord_shift_y, coord_shift_x)
    return r, phi



class SphericalPowerlawMultipoleEuclid(MassProfile):

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        k_m : float = 1.0,
        phi_m: float = 0.0,
        einstein_radius: float = 1.0,
        gamma: float = 2.0,
    ):

        self.centre = centre
        self.k_m = k_m
        self.phi_m = phi_m
        self.einstein_radius = einstein_radius
        self.gamma = gamma

    def convergence_2d_from(self, grid, **kwargs):
        return np.zeros(shape=grid.shape[0])

    def potential_2d_from(self, grid, **kwargs):
        return np.zeros(shape=grid.shape[0])

    def deflections_yx_2d_from(self, grid, **kwargs):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """

        profile = SphericalPowerlawMultipole()

        x = grid[:, 1]
        y = grid[:, 0]

        deflections_x, deflections_y = profile.deflection_angles_multipol_m4_o24(
                x,
            y,
                self.einstein_radius,
                self.gamma,
                self.k_m,
                self.phi_m,
                center_x=self.centre[1],
                center_y=self.centre[0],
                m=4,
            )

        return np.stack((deflections_y, deflections_x), axis=-1)



__all__ = ["SphericalPowerlawMultipole"]


class SphericalPowerlawMultipole(object):

    def function(self, x, y, theta_E, gamma, k_m, phi_m, center_x=0, center_y=0, m=4):
        """
        Lensing potential of PLmultipole contribution (for 1 component with m>=2).
        The equation for PLMultipole is Eq. (8) from Nightingale et al. (2023) (https://arxiv.org/abs/2209.10566)
        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param k_m : multipole strength and sqrt (a_m^2+b_m^2) of old definition.
        :param phi_m : multipole orientation in radian, arctan(b_m/a_m) of old definition
        :param center_x: profile center
        :param center_y: profile center
        :return: lensing potential
        """
        r, phi = cart2polar(x, y, center_x=center_x, center_y=center_y)
        r = np.maximum(r, 0.000001)
        f_ = theta_E ** (gamma - 1) * k_m / ((3 - gamma) ** 2 - m ** 2) * r ** (3 - gamma) * np.cos(m * (phi - phi_m))
        return f_

    def deflection_angles_multipol_m4_o24(self, x, y, theta_E, gamma, eta_m, phi_m, center_x=0, center_y=0, m=4):
        """This functions is similar to deflection_angles_multipol_m4, but uses the O'Riordan+24 definition of the multipole strength eta_m."""
        k_m = 0.5 * theta_E ** (gamma - 1) * eta_m  # converts from O'Riordan+24 to Nightingale+23 definition.
        return self.deflection_angles_multipol_m4(x, y, theta_E, gamma, k_m, phi_m, center_x=center_x,
                                                  center_y=center_y, m=m)

    def deflection_angles_multipol_m4(self, x, y, theta_E, gamma, k_m, phi_m, center_x=0, center_y=0, m=4):
        """
        Deflection of a multipole contribution (for 1 component with m>=2)

        This uses an extension of the parametrization from Chu et al. (2013) and Nightingale et al. (2023).
        It computes deflection angles alpha_x and alpha_y on a grid.

        Parameters
        ----------
        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param k_m : multipole strength and sqrt (a_m^2+b_m^2) of old definition.
        :param phi_m : multipole orientation in radian, arctan(b_m/a_m) of old definition
        :param center_x: profile center
        :param center_y: profile center
        :return: deflection angles alpha_x, alpha_y
        """
        r, phi = cart2polar(x, y, center_x=center_x, center_y=center_y)
        r = np.maximum(r, 0.000001)
        pre_factor = theta_E ** (gamma - 1) * k_m / ((3 - gamma) ** 2 - m ** 2)
        f_x = pre_factor * (
                np.cos(phi) * r ** (2 - gamma) * np.cos(m * (phi - phi_m)) +
                np.sin(phi) * r ** (1 - gamma) * m * np.sin(m * (phi - phi_m)))
        f_y = pre_factor * (
                np.sin(phi) * r ** (2 - gamma) * np.cos(m * (phi - phi_m)) -
                np.cos(phi) * r ** (1 - gamma) * m * np.sin(m * (phi - phi_m)))
        return f_x, f_y

    def hessian(self, x, y, theta_E, gamma, k_m, phi_m, center_x=0, center_y=0, m=4):
        """
        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param k_m : multipole strength and sqrt (a_m^2+b_m^2) of old definition.
        :param phi_m : multipole orientation in radian, arctan(b_m/a_m) of old definition
        :param center_x: profile center
        :param center_y: profile center
        :return: second derivates of the lensing potenial f_xx, f_xy, f_yx, f_yy
        """
        r, phi = cart2polar(x, y, center_x=center_x, center_y=center_y)
        r = np.maximum(r, 0.000001)
        pre_factor = theta_E ** (gamma - 1) * k_m / ((3 - gamma) ** 2 - m ** 2)

        term1_xx = pre_factor * r ** (1 - gamma) * (2 - gamma) * np.cos(phi) * (
                    np.cos(phi) * (3 - gamma) * np.cos(m * (phi - phi_m)) + np.sin(phi) * m * np.sin(m * (phi - phi_m)))
        term2_xx = pre_factor * r ** (1 - gamma) * (
                    np.sin(phi) ** 2 * np.cos(m * (phi - phi_m)) * ((3 - gamma) - m ** 2) + (2 - gamma) * np.cos(
                phi) * np.sin(phi) * m * np.sin(m * (phi - phi_m)))
        f_xx_multi = term1_xx + term2_xx

        term1_yy = pre_factor * r ** (1 - gamma) * (2 - gamma) * np.sin(phi) * (
                    np.sin(phi) * (3 - gamma) * np.cos(m * (phi - phi_m)) - np.cos(phi) * m * np.sin(m * (phi - phi_m)))
        term2_yy = pre_factor * r ** (1 - gamma) * (
                    np.cos(phi) ** 2 * np.cos(m * (phi - phi_m)) * ((3 - gamma) - m ** 2) - (2 - gamma) * np.cos(
                phi) * np.sin(phi) * m * np.sin(m * (phi - phi_m)))
        f_yy_multi = term1_yy + term2_yy

        term1_xy = pre_factor * r ** (1 - gamma) * (2 - gamma) * np.sin(phi) * (
                    np.cos(phi) * (3 - gamma) * np.cos(m * (phi - phi_m)) + np.sin(phi) * m * np.sin(m * (phi - phi_m)))
        term2_xy = pre_factor * r ** (1 - gamma) * (
                    np.sin(phi) * np.cos(phi) * np.cos(m * (phi - phi_m)) * (-(3 - gamma) + m ** 2) - (
                        2 - gamma) * np.cos(phi) ** 2 * m * np.sin(m * (phi - phi_m)))
        f_xy_multi = term1_xy + term2_xy

        return f_xx_multi, f_xy_multi, f_xy_multi, f_yy_multi
