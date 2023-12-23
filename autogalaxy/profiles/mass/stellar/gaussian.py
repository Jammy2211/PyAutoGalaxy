import copy
import numpy as np
from scipy.special import wofz
from scipy.integrate import quad
from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.mass.abstract.abstract import MassProfile
from autogalaxy.profiles.mass.stellar.abstract import StellarProfile


class Gaussian(MassProfile, StellarProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        sigma: float = 1.0,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The elliptical Gaussian light profile.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        sigma
            The sigma value of the Gaussian.
        """

        super(Gaussian, self).__init__(centre=centre, ell_comps=ell_comps)
        super(MassProfile, self).__init__(centre=centre, ell_comps=ell_comps)
        self.mass_to_light_ratio = mass_to_light_ratio
        self.intensity = intensity
        self.sigma = sigma

    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """
        return self.deflections_2d_via_analytic_from(grid=grid)

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_2d_via_analytic_from(self, grid: aa.type.Grid2DLike):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """

        deflections = (
            self.mass_to_light_ratio
            * self.intensity
            * self.sigma
            * np.sqrt((2 * np.pi) / (1.0 - self.axis_ratio**2.0))
            * self.zeta_from(grid=grid)
        )

        return self.rotated_grid_from_reference_frame_from(
            np.multiply(
                1.0, np.vstack((-1.0 * np.imag(deflections), np.real(deflections))).T
            )
        )

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_2d_via_integral_from(self, grid: aa.type.Grid2DLike):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        Note: sigma is divided by sqrt(q) here.

        """

        def calculate_deflection_component(npow, index):
            deflection_grid = self.axis_ratio * grid[:, index]

            for i in range(grid.shape[0]):
                deflection_grid[i] *= (
                    self.intensity
                    * self.mass_to_light_ratio
                    * quad(
                        self.deflection_func,
                        a=0.0,
                        b=1.0,
                        args=(
                            grid[i, 0],
                            grid[i, 1],
                            npow,
                            self.axis_ratio,
                            self.sigma / np.sqrt(self.axis_ratio),
                        ),
                    )[0]
                )

            return deflection_grid

        deflection_y = calculate_deflection_component(1.0, 0)
        deflection_x = calculate_deflection_component(0.0, 1)

        return self.rotated_grid_from_reference_frame_from(
            np.multiply(1.0, np.vstack((deflection_y, deflection_x)).T)
        )

    @staticmethod
    def deflection_func(u, y, x, npow, axis_ratio, sigma):
        _eta_u = np.sqrt(axis_ratio) * np.sqrt(
            (u * ((x**2) + (y**2 / (1 - (1 - axis_ratio**2) * u))))
        )

        return np.exp(-0.5 * np.square(np.divide(_eta_u, sigma))) / (
            (1 - (1 - axis_ratio**2) * u) ** (npow + 0.5)
        )

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def convergence_2d_from(self, grid: aa.type.Grid2DLike):
        """Calculate the projected convergence at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the convergence is computed on.

        """
        return self.convergence_func(self.eccentric_radii_grid_from(grid))

    def convergence_func(self, grid_radius: float) -> float:
        return self.mass_to_light_ratio * self.image_2d_via_radii_from(grid_radius)

    @aa.grid_dec.grid_2d_to_structure
    def potential_2d_from(self, grid: aa.type.Grid2DLike):
        return np.zeros(shape=grid.shape[0])

    def image_2d_via_radii_from(self, grid_radii: np.ndarray):
        """Calculate the intensity of the Gaussian light profile on a grid of radial coordinates.

        Parameters
        ----------
        grid_radii
            The radial distance from the centre of the profile. for each coordinate on the grid.

        Note: sigma is divided by sqrt(q) here.
        """
        return np.multiply(
            self.intensity,
            np.exp(
                -0.5
                * np.square(
                    np.divide(grid_radii, self.sigma / np.sqrt(self.axis_ratio))
                )
            ),
        )

    @property
    def axis_ratio(self):
        axis_ratio = super().axis_ratio
        return axis_ratio if axis_ratio < 0.9999 else 0.9999

    def zeta_from(self, grid: aa.type.Grid2DLike):
        q2 = self.axis_ratio**2.0
        ind_pos_y = grid[:, 0] >= 0
        shape_grid = np.shape(grid)
        output_grid = np.zeros((shape_grid[0]), dtype=np.complex128)
        scale_factor = self.axis_ratio / (self.sigma * np.sqrt(2.0 * (1.0 - q2)))

        xs_0 = grid[:, 1][ind_pos_y] * scale_factor
        ys_0 = grid[:, 0][ind_pos_y] * scale_factor
        xs_1 = grid[:, 1][~ind_pos_y] * scale_factor
        ys_1 = -grid[:, 0][~ind_pos_y] * scale_factor

        output_grid[ind_pos_y] = -1j * (
            wofz(xs_0 + 1j * ys_0)
            - np.exp(-(xs_0**2.0) * (1.0 - q2) - ys_0 * ys_0 * (1.0 / q2 - 1.0))
            * wofz(self.axis_ratio * xs_0 + 1j * ys_0 / self.axis_ratio)
        )

        output_grid[~ind_pos_y] = np.conj(
            -1j
            * (
                wofz(xs_1 + 1j * ys_1)
                - np.exp(-(xs_1**2.0) * (1.0 - q2) - ys_1 * ys_1 * (1.0 / q2 - 1.0))
                * wofz(self.axis_ratio * xs_1 + 1j * ys_1 / self.axis_ratio)
            )
        )

        return output_grid
