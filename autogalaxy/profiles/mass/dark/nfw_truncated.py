import numpy as np
from typing import Tuple

import autoarray as aa

from autogalaxy.cosmology.lensing import LensingCosmology
from autogalaxy.cosmology.wrap import Planck15
from autogalaxy.profiles.mass.dark.abstract import AbstractgNFW


class NFWTruncatedSph(AbstractgNFW):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        kappa_s: float = 0.05,
        scale_radius: float = 1.0,
        truncation_radius: float = 2.0,
    ):
        super().__init__(
            centre=centre,
            ell_comps=(0.0, 0.0),
            kappa_s=kappa_s,
            inner_slope=1.0,
            scale_radius=scale_radius,
        )

        self.truncation_radius = truncation_radius
        self.tau = self.truncation_radius / self.scale_radius

    @aa.grid_dec.grid_2d_to_vector_yx
    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """

        eta = np.multiply(1.0 / self.scale_radius, self.radial_grid_from(grid=grid))

        deflection_grid = np.multiply(
            (4.0 * self.kappa_s * self.scale_radius / eta),
            self.deflection_func_sph(grid_radius=eta),
        )

        return self._cartesian_grid_via_radial_from(grid, deflection_grid)

    def deflection_func_sph(self, grid_radius):
        grid_radius = grid_radius + 0j
        return np.real(self.coord_func_m(grid_radius=grid_radius))

    def convergence_func(self, grid_radius: float) -> float:
        grid_radius = ((1.0 / self.scale_radius) * grid_radius) + 0j
        return np.real(2.0 * self.kappa_s * self.coord_func_l(grid_radius=grid_radius))

    @aa.grid_dec.grid_2d_to_structure
    def potential_2d_from(self, grid: aa.type.Grid2DLike):
        return np.zeros(shape=grid.shape[0])

    def coord_func_k(self, grid_radius):
        return np.log(
            np.divide(
                grid_radius,
                np.sqrt(np.square(grid_radius) + np.square(self.tau)) + self.tau,
            )
        )

    def coord_func_l(self, grid_radius):
        f_r = self.coord_func_f(grid_radius=grid_radius)
        g_r = self.coord_func_g(grid_radius=grid_radius)
        k_r = self.coord_func_k(grid_radius=grid_radius)

        return np.divide(self.tau**2.0, (self.tau**2.0 + 1.0) ** 2.0) * (
            ((self.tau**2.0 + 1.0) * g_r)
            + (2 * f_r)
            - (np.pi / (np.sqrt(self.tau**2.0 + grid_radius**2.0)))
            + (
                (
                    (self.tau**2.0 - 1.0)
                    / (self.tau * (np.sqrt(self.tau**2.0 + grid_radius**2.0)))
                )
                * k_r
            )
        )

    def coord_func_m(self, grid_radius):
        f_r = self.coord_func_f(grid_radius=grid_radius)
        k_r = self.coord_func_k(grid_radius=grid_radius)

        return (self.tau**2.0 / (self.tau**2.0 + 1.0) ** 2.0) * (
            ((self.tau**2.0 + 2.0 * grid_radius**2.0 - 1.0) * f_r)
            + (np.pi * self.tau)
            + ((self.tau**2.0 - 1.0) * np.log(self.tau))
            + (
                np.sqrt(grid_radius**2.0 + self.tau**2.0)
                * (((self.tau**2.0 - 1.0) / self.tau) * k_r - np.pi)
            )
        )

    def mass_at_truncation_radius_solar_mass(
        self,
        redshift_profile,
        redshift_source,
        redshift_of_cosmic_average_density="profile",
        cosmology: LensingCosmology = Planck15(),
    ):
        mass_at_200 = self.mass_at_200_solar_masses(
            redshift_object=redshift_profile,
            redshift_source=redshift_source,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            cosmology=cosmology,
        )

        return (
            mass_at_200
            * (self.tau**2.0 / (self.tau**2.0 + 1.0) ** 2.0)
            * (
                ((self.tau**2.0 - 1) * np.log(self.tau))
                + (self.tau * np.pi)
                - (self.tau**2.0 + 1)
            )
        )
