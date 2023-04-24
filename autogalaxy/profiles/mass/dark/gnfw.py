import inspect
import numpy as np
from scipy import LowLevelCallable
from scipy import special
from scipy.integrate import quad
from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.mass.dark.abstract import AbstractgNFW


def jit_integrand(integrand_function):
    from numba import cfunc
    from numba.types import intc, CPointer, float64

    jitted_function = aa.util.numba.jit(nopython=True, cache=True)(integrand_function)
    no_args = len(inspect.getfullargspec(integrand_function).args)

    wrapped = None

    if no_args == 4:
        # noinspection PyUnusedLocal
        def wrapped(n, xx):
            return jitted_function(xx[0], xx[1], xx[2], xx[3])

    elif no_args == 5:
        # noinspection PyUnusedLocal
        def wrapped(n, xx):
            return jitted_function(xx[0], xx[1], xx[2], xx[3], xx[4])

    elif no_args == 6:
        # noinspection PyUnusedLocal
        def wrapped(n, xx):
            return jitted_function(xx[0], xx[1], xx[2], xx[3], xx[4], xx[5])

    elif no_args == 7:
        # noinspection PyUnusedLocal
        def wrapped(n, xx):
            return jitted_function(xx[0], xx[1], xx[2], xx[3], xx[4], xx[5], xx[6])

    elif no_args == 8:
        # noinspection PyUnusedLocal
        def wrapped(n, xx):
            return jitted_function(
                xx[0], xx[1], xx[2], xx[3], xx[4], xx[5], xx[6], xx[7]
            )

    elif no_args == 9:
        # noinspection PyUnusedLocal
        def wrapped(n, xx):
            return jitted_function(
                xx[0], xx[1], xx[2], xx[3], xx[4], xx[5], xx[6], xx[7], xx[8]
            )

    elif no_args == 10:
        # noinspection PyUnusedLocal
        def wrapped(n, xx):
            return jitted_function(
                xx[0], xx[1], xx[2], xx[3], xx[4], xx[5], xx[6], xx[7], xx[8], xx[9]
            )

    elif no_args == 11:
        # noinspection PyUnusedLocal
        def wrapped(n, xx):
            return jitted_function(
                xx[0],
                xx[1],
                xx[2],
                xx[3],
                xx[4],
                xx[5],
                xx[6],
                xx[7],
                xx[8],
                xx[9],
                xx[10],
            )

    cf = cfunc(float64(intc, CPointer(float64)))

    return LowLevelCallable(cf(wrapped).ctypes)


class gNFW(AbstractgNFW):
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):
        return self.deflections_2d_via_mge_from(grid=grid)

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_2d_via_mge_from(self, grid: aa.type.Grid2DLike):
        return self._deflections_2d_via_mge_from(
            grid=grid, sigmas_factor=self.axis_ratio
        )

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_2d_via_integral_from(
        self, grid: aa.type.Grid2DLike, tabulate_bins=1000
    ):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        tabulate_bins
            The number of bins to tabulate the inner integral of this profile.

        """

        @jit_integrand
        def surface_density_integrand(x, kappa_radius, scale_radius, inner_slope):
            return (
                (3 - inner_slope)
                * (x + kappa_radius / scale_radius) ** (inner_slope - 4)
                * (1 - np.sqrt(1 - x * x))
            )

        def calculate_deflection_component(npow, yx_index):
            deflection_grid = np.zeros(grid.shape[0])

            for i in range(grid.shape[0]):
                deflection_grid[i] = (
                    2.0
                    * self.kappa_s
                    * self.axis_ratio
                    * grid[i, yx_index]
                    * quad(
                        self.deflection_func,
                        a=0.0,
                        b=1.0,
                        args=(
                            grid[i, 0],
                            grid[i, 1],
                            npow,
                            self.axis_ratio,
                            minimum_log_eta,
                            maximum_log_eta,
                            tabulate_bins,
                            surface_density_integral,
                        ),
                        epsrel=gNFW.epsrel,
                    )[0]
                )

                return deflection_grid

        (
            eta_min,
            eta_max,
            minimum_log_eta,
            maximum_log_eta,
            bin_size,
        ) = self.tabulate_integral(grid, tabulate_bins)

        surface_density_integral = np.zeros((tabulate_bins,))

        for i in range(tabulate_bins):
            eta = 10.0 ** (minimum_log_eta + (i - 1) * bin_size)

            integral = quad(
                surface_density_integrand,
                a=0.0,
                b=1.0,
                args=(eta, self.scale_radius, self.inner_slope),
                epsrel=gNFW.epsrel,
            )[0]

            surface_density_integral[i] = (
                (eta / self.scale_radius) ** (1 - self.inner_slope)
            ) * (((1 + eta / self.scale_radius) ** (self.inner_slope - 3)) + integral)

        deflection_y = calculate_deflection_component(npow=1.0, yx_index=0)
        deflection_x = calculate_deflection_component(npow=0.0, yx_index=1)

        return self.rotated_grid_from_reference_frame_from(
            np.multiply(1.0, np.vstack((deflection_y, deflection_x)).T)
        )

    @staticmethod
    def deflection_func(
        u,
        y,
        x,
        npow,
        axis_ratio,
        minimum_log_eta,
        maximum_log_eta,
        tabulate_bins,
        surface_density_integral,
    ):
        _eta_u = np.sqrt((u * ((x**2) + (y**2 / (1 - (1 - axis_ratio**2) * u)))))
        bin_size = (maximum_log_eta - minimum_log_eta) / (tabulate_bins - 1)
        i = 1 + int((np.log10(_eta_u) - minimum_log_eta) / bin_size)
        r1 = 10.0 ** (minimum_log_eta + (i - 1) * bin_size)
        r2 = r1 * 10.0**bin_size
        kap = surface_density_integral[i] + (
            surface_density_integral[i + 1] - surface_density_integral[i]
        ) * (_eta_u - r1) / (r2 - r1)
        return kap / (1.0 - (1.0 - axis_ratio**2) * u) ** (npow + 0.5)

    def convergence_func(self, grid_radius: float) -> float:
        def integral_y(y, eta):
            return (y + eta) ** (self.inner_slope - 4) * (1 - np.sqrt(1 - y**2))

        grid_radius = (1.0 / self.scale_radius) * grid_radius

        for index in range(grid_radius.shape[0]):
            integral_y_value = quad(
                integral_y,
                a=0.0,
                b=1.0,
                args=grid_radius[index],
                epsrel=gNFW.epsrel,
            )[0]

            grid_radius[index] = (
                2.0
                * self.kappa_s
                * (grid_radius[index] ** (1 - self.inner_slope))
                * (
                    (1 + grid_radius[index]) ** (self.inner_slope - 3)
                    + ((3 - self.inner_slope) * integral_y_value)
                )
            )

        return grid_radius

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def potential_2d_from(self, grid: aa.type.Grid2DLike, tabulate_bins=1000):
        """
        Calculate the potential at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        tabulate_bins
            The number of bins to tabulate the inner integral of this profile.

        """

        @jit_integrand
        def deflection_integrand(x, kappa_radius, scale_radius, inner_slope):
            return (x + kappa_radius / scale_radius) ** (inner_slope - 3) * (
                (1 - np.sqrt(1 - x**2)) / x
            )

        (
            eta_min,
            eta_max,
            minimum_log_eta,
            maximum_log_eta,
            bin_size,
        ) = self.tabulate_integral(grid, tabulate_bins)

        potential_grid = np.zeros(grid.shape[0])

        deflection_integral = np.zeros((tabulate_bins,))

        for i in range(tabulate_bins):
            eta = 10.0 ** (minimum_log_eta + (i - 1) * bin_size)

            integral = quad(
                deflection_integrand,
                a=0.0,
                b=1.0,
                args=(eta, self.scale_radius, self.inner_slope),
                epsrel=gNFW.epsrel,
            )[0]

            deflection_integral[i] = (
                (eta / self.scale_radius) ** (2 - self.inner_slope)
            ) * (
                (1.0 / (3 - self.inner_slope))
                * special.hyp2f1(
                    3 - self.inner_slope,
                    3 - self.inner_slope,
                    4 - self.inner_slope,
                    -(eta / self.scale_radius),
                )
                + integral
            )

        for i in range(grid.shape[0]):
            potential_grid[i] = (2.0 * self.kappa_s * self.axis_ratio) * quad(
                self.potential_func,
                a=0.0,
                b=1.0,
                args=(
                    grid[i, 0],
                    grid[i, 1],
                    self.axis_ratio,
                    minimum_log_eta,
                    maximum_log_eta,
                    tabulate_bins,
                    deflection_integral,
                ),
                epsrel=gNFW.epsrel,
            )[0]

        return potential_grid

    @staticmethod
    def potential_func(
        u,
        y,
        x,
        axis_ratio,
        minimum_log_eta,
        maximum_log_eta,
        tabulate_bins,
        potential_integral,
    ):
        _eta_u = np.sqrt((u * ((x**2) + (y**2 / (1 - (1 - axis_ratio**2) * u)))))
        bin_size = (maximum_log_eta - minimum_log_eta) / (tabulate_bins - 1)
        i = 1 + int((np.log10(_eta_u) - minimum_log_eta) / bin_size)
        r1 = 10.0 ** (minimum_log_eta + (i - 1) * bin_size)
        r2 = r1 * 10.0**bin_size
        angle = potential_integral[i] + (
            potential_integral[i + 1] - potential_integral[i]
        ) * (_eta_u - r1) / (r2 - r1)
        return _eta_u * (angle / u) / (1.0 - (1.0 - axis_ratio**2) * u) ** 0.5


class gNFWSph(gNFW):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        kappa_s: float = 0.05,
        inner_slope: float = 1.0,
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
        inner_slope
            The inner slope of the dark matter halo.
        scale_radius
            The arc-second radius where the average density within this radius is 200 times the critical density of \
            the Universe..
        """

        super().__init__(
            centre=centre,
            ell_comps=(0.0, 0.0),
            kappa_s=kappa_s,
            inner_slope=inner_slope,
            scale_radius=scale_radius,
        )

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_2d_via_integral_from(self, grid: aa.type.Grid2DLike, **kwargs):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """

        eta = np.multiply(1.0 / self.scale_radius, self.radial_grid_from(grid))

        deflection_grid = np.zeros(grid.shape[0])

        for i in range(grid.shape[0]):
            deflection_grid[i] = np.multiply(
                4.0 * self.kappa_s * self.scale_radius, self.deflection_func_sph(eta[i])
            )

        return self._cartesian_grid_via_radial_from(grid, deflection_grid)

    @staticmethod
    def deflection_integrand(y, eta, inner_slope):
        return (y + eta) ** (inner_slope - 3) * ((1 - np.sqrt(1 - y**2)) / y)

    def deflection_func_sph(self, eta):
        integral_y_2 = quad(
            self.deflection_integrand,
            a=0.0,
            b=1.0,
            args=(eta, self.inner_slope),
            epsrel=1.49e-6,
        )[0]
        return eta ** (2 - self.inner_slope) * (
            (1.0 / (3 - self.inner_slope))
            * special.hyp2f1(
                3 - self.inner_slope, 3 - self.inner_slope, 4 - self.inner_slope, -eta
            )
            + integral_y_2
        )
