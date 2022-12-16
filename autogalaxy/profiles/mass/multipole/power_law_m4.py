from astropy import units
import numpy as np
from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.mass.abstract.abstract import MassProfile


def polar_grid_from(grid : aa.type.Grid2DLike, centre : Tuple[float, float] = (0.0, 0.0)) -> Tuple[np.ndarray, np.ndarray]:

    y, x = grid.T

    x_shifted = np.subtract(x, centre[1])
    y_shifted = np.subtract(y, centre[0])

    r = np.sqrt(x_shifted**2 + y_shifted**2)

    phi = np.arctan2(y_shifted, x_shifted)

    return r, phi


def multipole_parameters_from(ell_comps : Tuple[float, float]) -> Tuple[float, float]:
    """
    :param e1: ellipticity component
    :param e2: ellipticity component
    :return:
    """
    phi_m = np.arctan(ell_comps[0] / ell_comps[1]) * units.rad.to(units.deg)
    k_m = np.sqrt(ell_comps[1] ** 2 + ell_comps[0] ** 2)
    if phi_m < 0.0:
        return k_m, phi_m + 90.0
    return k_m, phi_m


class MultipolePowerLawM4(MassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
        slope: float = 2.0,
        ell_comps_multipole: Tuple[float, float] = (0.0, 0.0),
    ):
        """
        phi_m: [rad]
        """
        super().__init__(centre=centre, ell_comps=(0.0, 0.0))

        self.m = 4

        self.einstein_radius = einstein_radius
        self.slope = slope

        self.ell_comps_multipole = ell_comps_multipole
        self.k_m, self.phi_m = multipole_parameters_from(ell_comps=ell_comps_multipole)
        self.phi_m *= units.deg.to(units.rad)

    def jacobian(self, a_r: float, a_phi, phi):
        """
        The Jacobian transformation from polar to cartesian coordinates
        """
        return (
            a_r * np.sin(phi) + a_phi * np.cos(phi),
            a_r * np.cos(phi) - a_phi * np.sin(phi),
        )

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def convergence_2d_from(self, grid : aa.type.Grid1D2DLike) -> np.ndarray:

        r, phi = polar_grid_from(grid=grid)

        return (
            1.0
            / 2.0
            * (self.einstein_radius / r) ** (self.slope - 1)
            * self.k_m
            * np.cos(self.m * (phi - self.phi_m))
        )

    @aa.grid_dec.grid_2d_to_structure
    def potential_2d_from(self, grid: aa.type.Grid2DLike) -> np.ndarray:
        return np.zeros(shape=grid.shape[0])

    @aa.grid_dec.grid_2d_to_vector_yx
    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid : aa.type.Grid1D2DLike) -> np.ndarray:

        r, phi = polar_grid_from(grid=grid)

        a_r = (
            -(
                (3.0 - self.slope)
                * self.einstein_radius ** (self.slope - 1.0)
                * r ** (2.0 - self.slope)
            )
            / (self.m**2.0 - (3.0 - self.slope))
            * self.k_m
            * np.cos(self.m * (phi - self.phi_m))
        )

        a_phi = (
            (
                self.m**2.0
                * self.einstein_radius ** (self.slope - 1.0)
                * r ** (2.0 - self.slope)
            )
            / (self.m**2.0 - (3.0 - self.slope))
            * self.k_m
            * np.sin(self.m * (phi - self.phi_m))
        )

        return np.stack(self.jacobian(a_r=a_r, a_phi=a_phi, phi=phi), axis=-1)
