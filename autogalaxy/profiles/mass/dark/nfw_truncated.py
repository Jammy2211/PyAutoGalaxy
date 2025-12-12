import numpy as np
from typing import Tuple

import autoarray as aa

from autogalaxy.cosmology.lensing import LensingCosmology
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

    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """

        eta = xp.multiply(
            1.0 / self.scale_radius,
            self.radial_grid_from(grid=grid, xp=xp, **kwargs).array,
        )

        deflection_grid = xp.multiply(
            (4.0 * self.kappa_s * self.scale_radius / eta),
            self.deflection_func_sph(grid_radius=eta),
        )

        return self._cartesian_grid_via_radial_from(
            grid=grid, radius=deflection_grid, xp=xp
        )

    def deflection_func_sph(self, grid_radius, xp=np):
        grid_radius = grid_radius + 0j
        return xp.real(self.coord_func_m(grid_radius=grid_radius, xp=xp))

    def convergence_func(self, grid_radius: float, xp=np) -> float:
        grid_radius = ((1.0 / self.scale_radius) * grid_radius) + 0j
        return xp.real(
            2.0 * self.kappa_s * self.coord_func_l(grid_radius=grid_radius.array, xp=xp)
        )

    @aa.grid_dec.to_array
    def potential_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        return xp.zeros(shape=grid.shape[0])

    def coord_func_k(self, grid_radius, xp=np):
        return xp.log(
            xp.divide(
                grid_radius,
                xp.sqrt(xp.square(grid_radius) + xp.square(self.tau)) + self.tau,
            )
        )

    def coord_func_l(self, grid_radius, xp=np):
        f_r = self.coord_func_f(grid_radius=grid_radius, xp=xp)
        g_r = self.coord_func_g(grid_radius=grid_radius, xp=xp)
        k_r = self.coord_func_k(grid_radius=grid_radius, xp=xp)

        return xp.divide(self.tau**2.0, (self.tau**2.0 + 1.0) ** 2.0) * (
            ((self.tau**2.0 + 1.0) * g_r)
            + (2 * f_r)
            - (xp.pi / (xp.sqrt(self.tau**2.0 + grid_radius**2.0)))
            + (
                (
                    (self.tau**2.0 - 1.0)
                    / (self.tau * (xp.sqrt(self.tau**2.0 + grid_radius**2.0)))
                )
                * k_r
            )
        )

    def coord_func_m(self, grid_radius, xp=np):
        f_r = self.coord_func_f(grid_radius=grid_radius, xp=xp)
        k_r = self.coord_func_k(grid_radius=grid_radius, xp=xp)

        return (self.tau**2.0 / (self.tau**2.0 + 1.0) ** 2.0) * (
            ((self.tau**2.0 + 2.0 * grid_radius**2.0 - 1.0) * f_r)
            + (xp.pi * self.tau)
            + ((self.tau**2.0 - 1.0) * xp.log(self.tau))
            + (
                xp.sqrt(grid_radius**2.0 + self.tau**2.0)
                * (((self.tau**2.0 - 1.0) / self.tau) * k_r - xp.pi)
            )
        )

    def mass_at_truncation_radius_solar_mass(
        self,
        redshift_profile,
        redshift_source,
        redshift_of_cosmic_average_density="profile",
        cosmology: LensingCosmology = None,
        xp=np,
    ):
        from autogalaxy.cosmology.wrap import Planck15

        cosmology = cosmology or Planck15()

        mass_at_200 = self.mass_at_200_solar_masses(
            redshift_object=redshift_profile,
            redshift_source=redshift_source,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            cosmology=cosmology,
            xp=xp,
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
