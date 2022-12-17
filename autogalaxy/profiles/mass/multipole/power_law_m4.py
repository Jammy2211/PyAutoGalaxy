from astropy import units
import numpy as np
from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.mass.abstract.abstract import MassProfile


def radial_and_angle_grid_from(
    grid: aa.type.Grid2DLike, centre: Tuple[float, float] = (0.0, 0.0)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts the input grid of Cartesian (y,x) coordinates to their correspond radial and polar grids.

    Parameters
    ----------
    grid
        The grid of (y,x) arc-second coordinates that are converted to radial and polar values.
    centre
        The centre of the multipole profile.

    Returns
    -------
    The radial and polar coordinate grids of the input (y,x) Cartesian grid.
    """
    y, x = grid.T

    x_shifted = np.subtract(x, centre[1])
    y_shifted = np.subtract(y, centre[0])

    radial_grid = np.sqrt(x_shifted**2 + y_shifted**2)

    angle_grid = np.arctan2(y_shifted, x_shifted)

    return radial_grid, angle_grid


def multipole_parameters_from(
    ell_comps_multipole: Tuple[float, float]
) -> Tuple[float, float]:
    """
    Converts the multipole elliptical components to their normalizartion value `k_m` and angle `phi`,
    which are given by:

    .. math::
        \phi^{\rm mass}_m = \arctan{\frac{\epsilon_{\rm 2}^{\rm mp}}{\epsilon_{\rm 2}^{\rm mp}}}, \, \,
        k^{\rm mass}_m = \sqrt{{\epsilon_{\rm 1}^{\rm mp}}^2 + {\epsilon_{\rm 2}^{\rm mp}}^2} \, .

    Parameters
    ----------
    ell_comps_multipole
        The first and second ellipticity components of the multipole.


    Returns
    -------
    The normalization parameters of the multipole.
    """
    angle_m = np.arctan(ell_comps_multipole[0] / ell_comps_multipole[1]) * units.rad.to(
        units.deg
    )
    k_m = np.sqrt(ell_comps_multipole[1] ** 2 + ell_comps_multipole[0] ** 2)
    if angle_m < 0.0:
        return k_m, angle_m + 90.0
    return k_m, angle_m


class MultipolePowerLawM4(MassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
        slope: float = 2.0,
        ell_comps_multipole: Tuple[float, float] = (0.0, 0.0),
    ):
        r"""
        A multipole extension with multipole order M=4 to the power-law total mass distribution.

        Quantities computed from this profile (e.g. deflections, convergence) are of only the multipole, and not the
        power-law mass distribution itself.

        The typical use case is therefore for the multipoles to be combined with a `PowerLaw` mass profile with the
        same parameters (see example below).

        When combined with a power-law, the functional form of the convergence is:

        .. math::
            \kappa(r, \phi) = \frac{1}{2} \left(\frac{\theta_{\rm E}^{\rm mass}}{r}\right)^{\gamma^{\rm mass} - 1}
            k^{\rm mass}_m \, \cos(m(\phi - \phi^{\rm mass}_m)) \, ,

        Where \\xi are elliptical coordinates calculated according to :class: SphProfile.

        The parameters :math: k^{\rm mass}_m and :math: \phi^{\rm mass}_are parameterized as elliptical components
        :math: (\epsilon_{\rm 1}^{\rm mp}\,\epsilon_{\rm 2}^{\rm mp}), which are given by:

        .. math::
                \phi^{\rm mass}_m = \arctan{\frac{\epsilon_{\rm 2}^{\rm mp}}{\epsilon_{\rm 2}^{\rm mp}}}, \, \,
                k^{\rm mass}_m = \sqrt{{\epsilon_{\rm 1}^{\rm mp}}^2 + {\epsilon_{\rm 2}^{\rm mp}}^2} \, .

        This mass profile is described fully in the following paper: https://arxiv.org/abs/1302.5482

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        einstein_radius
            The arc-second Einstein radius.
        slope
            The density slope of the power-law (lower value -> shallower profile, higher value -> steeper profile).
        ell_comps_multipole
            The first and second ellipticity components of the multipole.

        Examples
        --------

        mass = al.mp.PowerLaw(
            centre=(0.0, 0.0),
            ell_comps=(-0.1, 0.2),
            einstein_radius=1.0,
            slope=2.2
        )

        multipole = al.mp.MultipolePowerLawM4(
            centre=(0.0, 0.0),
            einstein_radius=1.0,
            slope=2.2,
            ell_comps_multipole=(0.3, 0.2)
        )

        galaxy = al.Galaxy(
            redshift=0.5,
            mass=mass,
            multipole=multipole
        )

        grid=al.Grid2D.uniform(shape_native=(10, 10), pixel_scales=0.1)

        deflections = galaxy.deflections_yx_2d_from(
            grid=grid
        )
        """
        super().__init__(centre=centre, ell_comps=(0.0, 0.0))

        self.m = 4

        self.einstein_radius = einstein_radius
        self.slope = slope

        self.ell_comps_multipole = ell_comps_multipole
        self.k_m, self.angle_m = multipole_parameters_from(
            ell_comps_multipole=ell_comps_multipole
        )
        self.angle_m *= units.deg.to(units.rad)

    def jacobian(
        self, a_r: np.ndarray, a_angle: np.ndarray, polar_angle_grid: np.ndarray
    ) -> Tuple[np.ndarray, Tuple]:
        """
        The Jacobian transformation from polar to cartesian coordinates.

        Parameters
        ----------
        a_r
            Ask Aris
        a_angle
            Ask Aris
        polar_angle_grid
            The polar angle coordinates of the input (y,x) Cartesian grid of coordinates.
        """
        return (
            a_r * np.sin(polar_angle_grid) + a_angle * np.cos(polar_angle_grid),
            a_r * np.cos(polar_angle_grid) - a_angle * np.sin(polar_angle_grid),
        )

    @aa.grid_dec.grid_2d_to_vector_yx
    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid: aa.type.Grid1D2DLike) -> np.ndarray:
        """
        Calculate the deflection angles on a grid of (y,x) arc-second coordinates.

        For coordinates (0.0, 0.0) the analytic calculation of the deflection angle gives a NaN. Therefore,
        coordinates at (0.0, 0.0) are shifted slightly to (1.0e-8, 1.0e-8).

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        radial_grid, polar_angle_grid = radial_and_angle_grid_from(grid=grid)

        a_r = (
            -(
                (3.0 - self.slope)
                * self.einstein_radius ** (self.slope - 1.0)
                * radial_grid ** (2.0 - self.slope)
            )
            / (self.m**2.0 - (3.0 - self.slope))
            * self.k_m
            * np.cos(self.m * (polar_angle_grid - self.angle_m))
        )

        a_angle = (
            (
                self.m**2.0
                * self.einstein_radius ** (self.slope - 1.0)
                * radial_grid ** (2.0 - self.slope)
            )
            / (self.m**2.0 - (3.0 - self.slope))
            * self.k_m
            * np.sin(self.m * (polar_angle_grid - self.angle_m))
        )

        return np.stack(
            self.jacobian(a_r=a_r, a_angle=a_angle, polar_angle_grid=polar_angle_grid),
            axis=-1,
        )

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def convergence_2d_from(self, grid: aa.type.Grid1D2DLike) -> np.ndarray:
        """
        Calculate the projected convergence on a grid of (y,x) arc-second coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the convergence is computed on.
        """
        r, angle = radial_and_angle_grid_from(grid=grid)

        return (
            1.0
            / 2.0
            * (self.einstein_radius / r) ** (self.slope - 1)
            * self.k_m
            * np.cos(self.m * (angle - self.angle_m))
        )

    @aa.grid_dec.grid_2d_to_structure
    def potential_2d_from(self, grid: aa.type.Grid2DLike) -> np.ndarray:
        """
        Calculate the potential on a grid of (y,x) arc-second coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        return np.zeros(shape=grid.shape[0])
