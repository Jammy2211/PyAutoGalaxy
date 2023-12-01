import numpy as np
from scipy.integrate import quad
from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.mass.dark.gnfw import gNFW
from autogalaxy.profiles.mass.abstract.cse import MassProfileCSE


class NFW(gNFW, MassProfileCSE):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        kappa_s: float = 0.05,
        scale_radius: float = 1.0,
    ):
        """
        The elliptical NFW profiles, used to fit the dark matter halo of the lens.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        kappa_s
            The overall normalization of the dark matter halo \
            (kappa_s = (rho_s * scale_radius)/lensing_critical_density)
        scale_radius
            The NFW scale radius `r_s`, as an angle on the sky in arcseconds.
        """

        super().__init__(
            centre=centre,
            ell_comps=ell_comps,
            kappa_s=kappa_s,
            inner_slope=1.0,
            scale_radius=scale_radius,
        )
        super(MassProfileCSE, self).__init__()

    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):
        return self.deflections_2d_via_cse_from(grid=grid)

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

        """

        def calculate_deflection_component(npow, index):
            deflection_grid = self.axis_ratio * grid[:, index]

            for i in range(grid.shape[0]):
                deflection_grid[i] *= (
                    self.kappa_s
                    * quad(
                        self.deflection_func,
                        a=0.0,
                        b=1.0,
                        args=(
                            grid[i, 0],
                            grid[i, 1],
                            npow,
                            self.axis_ratio,
                            self.scale_radius,
                        ),
                    )[0]
                )

            return deflection_grid

        deflection_y = calculate_deflection_component(1.0, 0)
        deflection_x = calculate_deflection_component(0.0, 1)

        return self.rotated_grid_from_reference_frame_from(
            np.multiply(1.0, np.vstack((deflection_y, deflection_x)).T)
        )

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_2d_via_cse_from(self, grid: aa.type.Grid2DLike):
        return self._deflections_2d_via_cse_from(grid=grid)

    @staticmethod
    def deflection_func(u, y, x, npow, axis_ratio, scale_radius):
        _eta_u = (1.0 / scale_radius) * np.sqrt(
            (u * ((x**2) + (y**2 / (1 - (1 - axis_ratio**2) * u))))
        )

        if _eta_u > 1:
            _eta_u_2 = (1.0 / np.sqrt(_eta_u**2 - 1)) * np.arctan(
                np.sqrt(_eta_u**2 - 1)
            )
        elif _eta_u < 1:
            _eta_u_2 = (1.0 / np.sqrt(1 - _eta_u**2)) * np.arctanh(
                np.sqrt(1 - _eta_u**2)
            )
        else:
            _eta_u_2 = 1

        return (
            2.0
            * (1 - _eta_u_2)
            / (_eta_u**2 - 1)
            / ((1 - (1 - axis_ratio**2) * u) ** (npow + 0.5))
        )

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def convergence_2d_via_cse_from(self, grid: aa.type.Grid2DLike):
        """
        Calculate the projected 2D convergence from a grid of (y,x) arc second coordinates, by computing and summing
        the convergence of each individual cse used to decompose the mass profile.

        The cored steep elliptical (cse) decomposition of a the elliptical NFW mass
        profile (e.g. `decompose_convergence_via_cse`) is using equation (12) of
        Oguri 2021 (https://arxiv.org/abs/2106.11464).

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the convergence is computed on.
        """

        elliptical_radii = self.elliptical_radii_grid_from(grid)

        return self._convergence_2d_via_cse_from(grid_radii=elliptical_radii)

    def convergence_func(self, grid_radius: float) -> float:
        grid_radius = (1.0 / self.scale_radius) * grid_radius + 0j
        return np.real(2.0 * self.kappa_s * self.coord_func_g(grid_radius=grid_radius))

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def potential_2d_from(self, grid: aa.type.Grid2DLike):
        """
        Calculate the potential at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """

        potential_grid = np.zeros(grid.shape[0])

        for i in range(grid.shape[0]):
            potential_grid[i] = quad(
                self.potential_func,
                a=0.0,
                b=1.0,
                args=(
                    grid[i, 0],
                    grid[i, 1],
                    self.axis_ratio,
                    self.kappa_s,
                    self.scale_radius,
                ),
                epsrel=1.49e-5,
            )[0]

        return potential_grid

    @staticmethod
    def potential_func(u, y, x, axis_ratio, kappa_s, scale_radius):
        _eta_u = (1.0 / scale_radius) * np.sqrt(
            (u * ((x**2) + (y**2 / (1 - (1 - axis_ratio**2) * u))))
        )

        if _eta_u > 1:
            _eta_u_2 = (1.0 / np.sqrt(_eta_u**2 - 1)) * np.arctan(
                np.sqrt(_eta_u**2 - 1)
            )
        elif _eta_u < 1:
            _eta_u_2 = (1.0 / np.sqrt(1 - _eta_u**2)) * np.arctanh(
                np.sqrt(1 - _eta_u**2)
            )
        else:
            _eta_u_2 = 1

        return (
            4.0
            * kappa_s
            * scale_radius
            * (axis_ratio / 2.0)
            * (_eta_u / u)
            * ((np.log(_eta_u / 2.0) + _eta_u_2) / _eta_u)
            / ((1 - (1 - axis_ratio**2) * u) ** 0.5)
        )

    def decompose_convergence_via_cse(
        self, grid_radii: np.ndarray, total_cses=30, sample_points=60
    ):
        """
        Decompose the convergence of the elliptical NFW mass profile into cored steep elliptical (cse) profiles.

        This uses an input function `func` which is specific to the elliptical NFW mass profile, and is defined by
        equation (12) of Oguri 2021 (https://arxiv.org/abs/2106.11464).

        Parameters
        ----------
        func
            The function representing the profile that is decomposed into CSEs.
        radii_min:
            The minimum radius to fit
        radii_max:
            The maximum radius to fit
        total_cses
            The number of CSEs used to approximate the input func.
        sample_points: int (should be larger than 'total_cses')
            The number of data points to fit

        Returns
        -------
        Tuple[List, List]
            A list of amplitudes and core radii of every cored steep elliptical (cse) the mass profile is decomposed
            into.
        """
        radii_min = 0.005
        radii_max = max(7.5, np.max(grid_radii))

        def nfw_2d(r):
            grid_radius = (1.0 / self.scale_radius) * r + 0j
            return np.real(
                2.0 * self.kappa_s * self.coord_func_g(grid_radius=grid_radius)
            )

        return self._decompose_convergence_via_cse_from(
            func=nfw_2d,
            radii_min=radii_min,
            radii_max=radii_max,
            total_cses=total_cses,
            sample_points=sample_points,
        )

    @staticmethod
    def coord_func(r):
        if r > 1:
            return (1.0 / np.sqrt(r**2 - 1)) * np.arctan(np.sqrt(r**2 - 1))
        elif r < 1:
            return (1.0 / np.sqrt(1 - r**2)) * np.arctanh(np.sqrt(1 - r**2))
        elif r == 1:
            return 1


class NFWSph(NFW):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        kappa_s: float = 0.05,
        scale_radius: float = 1.0,
    ):
        """
        The spherical NFW profiles, used to fit the dark matter halo of the lens.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        kappa_s
            The overall normalization of the dark matter halo \
            (kappa_s = (rho_s * scale_radius)/lensing_critical_density)
        scale_radius
            The arc-second radius where the average density within this radius is 200 times the critical density of \
            the Universe..
        """

        super().__init__(
            centre=centre,
            ell_comps=(0.0, 0.0),
            kappa_s=kappa_s,
            scale_radius=scale_radius,
        )

    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):
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

        eta = np.multiply(1.0 / self.scale_radius, self.radial_grid_from(grid=grid))

        deflection_grid = np.multiply(
            (4.0 * self.kappa_s * self.scale_radius / eta),
            self.deflection_func_sph(grid_radius=eta),
        )

        return self._cartesian_grid_via_radial_from(grid, deflection_grid)

    def deflection_func_sph(self, grid_radius):
        grid_radius = grid_radius + 0j
        return np.real(self.coord_func_h(grid_radius=grid_radius))

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def potential_2d_from(self, grid: aa.type.Grid2DLike):
        """
        Calculate the potential at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """
        eta = (1.0 / self.scale_radius) * self.radial_grid_from(grid) + 0j
        return np.real(
            2.0 * self.scale_radius * self.kappa_s * self.potential_func_sph(eta)
        )

    @staticmethod
    def potential_func_sph(eta):
        return ((np.log(eta / 2.0)) ** 2) - (np.arctanh(np.sqrt(1 - eta**2))) ** 2
