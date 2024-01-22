import copy
import numpy as np
from scipy.optimize import fsolve
from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.mass.abstract.abstract import MassProfile
from autogalaxy.cosmology.lensing import LensingCosmology
from autogalaxy.cosmology.wrap import Planck15
from autogalaxy.profiles.mass.abstract.mge import (
    MassProfileMGE,
)

from autogalaxy import exc


class DarkProfile:
    pass


class AbstractgNFW(MassProfile, DarkProfile, MassProfileMGE):
    epsrel = 1.49e-5

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        kappa_s: float = 0.05,
        inner_slope: float = 1.0,
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
        inner_slope
            The inner slope of the dark matter halo
        scale_radius
            The NFW scale radius as an angle on the sky in arc-seconds. For a regular NFW halo, \
            the scale radius is roughly where the log-slope of the profile changes from -1 to -3.
        """

        super().__init__(centre=centre, ell_comps=ell_comps)
        super(MassProfileMGE, self).__init__()

        self.kappa_s = kappa_s
        self.scale_radius = scale_radius
        self.inner_slope = inner_slope

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

        grid_eta = self.elliptical_radii_grid_from(grid=grid)

        return self.convergence_func(grid_radius=grid_eta)

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def convergence_2d_via_mge_from(self, grid: aa.type.Grid2DLike):
        """Calculate the projected convergence at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the convergence is computed on.

        """

        elliptical_radii = self.elliptical_radii_grid_from(grid)

        return self._convergence_2d_via_mge_from(grid_radii=elliptical_radii)

    def tabulate_integral(self, grid, tabulate_bins):
        """Tabulate an integral over the convergence of deflection potential of a mass profile. This is used in \
        the GeneralizedNFW profile classes to speed up the integration procedure.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the potential / deflection_stacks are computed on.
        tabulate_bins
            The number of bins to tabulate the inner integral of this profile.
        """
        eta_min = 1.0e-4
        eta_max = 1.05 * np.max(self.elliptical_radii_grid_from(grid))

        minimum_log_eta = np.log10(eta_min)
        maximum_log_eta = np.log10(eta_max)
        bin_size = (maximum_log_eta - minimum_log_eta) / (tabulate_bins - 1)

        return eta_min, eta_max, minimum_log_eta, maximum_log_eta, bin_size

    def decompose_convergence_via_mge(self):
        rho_at_scale_radius = (
            self.kappa_s / self.scale_radius
        )  # density parameter of 3D gNFW

        radii_min = self.scale_radius / 2000.0
        radii_max = self.scale_radius * 30.0

        def gnfw_3d(r):
            x = r / self.scale_radius
            return (
                rho_at_scale_radius
                * x ** (-self.inner_slope)
                * (1.0 + x) ** (self.inner_slope - 3.0)
            )

        amplitude_list, sigma_list = self._decompose_convergence_via_mge(
            func=gnfw_3d, radii_min=radii_min, radii_max=radii_max
        )
        amplitude_list *= np.sqrt(2.0 * np.pi) * sigma_list
        return amplitude_list, sigma_list

    def coord_func_f(self, grid_radius):
        if isinstance(grid_radius, np.ndarray):
            return self.coord_func_f_jit(
                grid_radius=grid_radius,
                f=np.ones(shape=grid_radius.shape[0], dtype="complex64"),
            )
        else:
            return self.coord_func_f_float_jit(grid_radius=grid_radius)

    @staticmethod
    @aa.util.numba.jit()
    def coord_func_f_jit(grid_radius, f):
        for index in range(f.shape[0]):
            if np.real(grid_radius[index]) > 1.0:
                f[index] = (
                    1.0 / np.sqrt(np.square(grid_radius[index]) - 1.0)
                ) * np.arccos(np.divide(1.0, grid_radius[index]))
            elif np.real(grid_radius[index]) < 1.0:
                f[index] = (
                    1.0 / np.sqrt(1.0 - np.square(grid_radius[index]))
                ) * np.arccosh(np.divide(1.0, grid_radius[index]))

        return f

    @staticmethod
    @aa.util.numba.jit()
    def coord_func_f_float_jit(grid_radius):
        if np.real(grid_radius) > 1.0:
            return (1.0 / np.sqrt(np.square(grid_radius) - 1.0)) * np.arccos(
                np.divide(1.0, grid_radius)
            )
        elif np.real(grid_radius) < 1.0:
            return (1.0 / np.sqrt(1.0 - np.square(grid_radius))) * np.arccosh(
                np.divide(1.0, grid_radius)
            )
        else:
            return 1.0

    def coord_func_g(self, grid_radius):
        f_r = self.coord_func_f(grid_radius=grid_radius)

        if isinstance(grid_radius, np.ndarray):
            return self.coord_func_g_jit(
                grid_radius=grid_radius,
                f_r=f_r,
                g=np.zeros(shape=grid_radius.shape[0], dtype="complex64"),
            )
        else:
            return self.coord_func_g_float_jit(grid_radius=grid_radius, f_r=f_r)

    @staticmethod
    @aa.util.numba.jit()
    def coord_func_g_jit(grid_radius, f_r, g):
        for index in range(f_r.shape[0]):
            if np.real(grid_radius[index]) > 1.0:
                g[index] = (1.0 - f_r[index]) / (np.square(grid_radius[index]) - 1.0)
            elif np.real(grid_radius[index]) < 1.0:
                g[index] = (f_r[index] - 1.0) / (1.0 - np.square(grid_radius[index]))
            else:
                g[index] = 1.0 / 3.0

        return g

    @staticmethod
    @aa.util.numba.jit()
    def coord_func_g_float_jit(grid_radius, f_r):
        if np.real(grid_radius) > 1.0:
            return (1.0 - f_r) / (np.square(grid_radius) - 1.0)
        elif np.real(grid_radius) < 1.0:
            return (f_r - 1.0) / (1.0 - np.square(grid_radius))
        else:
            return 1.0 / 3.0

    def coord_func_h(self, grid_radius):
        return np.log(grid_radius / 2.0) + self.coord_func_f(grid_radius=grid_radius)

    def rho_at_scale_radius_solar_mass_per_kpc3(
        self, redshift_object, redshift_source, cosmology: LensingCosmology = Planck15()
    ):
        """
        The Cosmic average density is defined at the redshift of the profile.
        """

        critical_surface_density = cosmology.critical_surface_density_between_redshifts_solar_mass_per_kpc2_from(
            redshift_0=redshift_object, redshift_1=redshift_source
        )

        kpc_per_arcsec = cosmology.kpc_per_arcsec_from(redshift=redshift_object)

        return (
            self.kappa_s
            * critical_surface_density
            / (self.scale_radius * kpc_per_arcsec)
        )

    def delta_concentration(
        self,
        redshift_object,
        redshift_source,
        redshift_of_cosmic_average_density="profile",
        cosmology: LensingCosmology = Planck15(),
    ):
        if redshift_of_cosmic_average_density == "profile":
            redshift_calc = redshift_object
        elif redshift_of_cosmic_average_density == "local":
            redshift_calc = 0.0
        else:
            raise exc.UnitsException(
                "The redshift of the cosmic average density haas been specified as an invalid "
                "string. Must be {local, profile}"
            )

        cosmic_average_density = (
            cosmology.cosmic_average_density_solar_mass_per_kpc3_from(
                redshift=redshift_calc
            )
        )

        rho_scale_radius = self.rho_at_scale_radius_solar_mass_per_kpc3(
            redshift_object=redshift_object,
            redshift_source=redshift_source,
            cosmology=cosmology,
        )

        return rho_scale_radius / cosmic_average_density

    def concentration(
        self,
        redshift_profile,
        redshift_source,
        redshift_of_cosmic_average_density="profile",
        cosmology: LensingCosmology = Planck15(),
    ):
        """
        Computes the NFW halo concentration, `c_{200m}`
        """
        delta_concentration = self.delta_concentration(
            redshift_object=redshift_profile,
            redshift_source=redshift_source,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            cosmology=cosmology,
        )

        return fsolve(
            func=self.concentration_func, x0=10.0, args=(delta_concentration,)
        )[0]

    @staticmethod
    def concentration_func(concentration, delta_concentration):
        return (
            200.0
            / 3.0
            * (
                concentration
                * concentration
                * concentration
                / (np.log(1 + concentration) - concentration / (1 + concentration))
            )
            - delta_concentration
        )

    def radius_at_200(
        self,
        redshift_object,
        redshift_source,
        redshift_of_cosmic_average_density="profile",
        cosmology: LensingCosmology = Planck15(),
    ):
        """
        Returns `r_{200m}` for this halo in **arcseconds**
        """
        concentration = self.concentration(
            redshift_profile=redshift_object,
            redshift_source=redshift_source,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            cosmology=cosmology,
        )

        return concentration * self.scale_radius

    def mass_at_200_solar_masses(
        self,
        redshift_object,
        redshift_source,
        redshift_of_cosmic_average_density="profile",
        cosmology: LensingCosmology = Planck15(),
    ):
        """
        Returns `M_{200m}` of this NFW halo, in solar masses, at the given cosmology.
        """
        if redshift_of_cosmic_average_density == "profile":
            redshift_calc = redshift_object
        elif redshift_of_cosmic_average_density == "local":
            redshift_calc = 0.0
        else:
            raise exc.UnitsException(
                "The redshift of the cosmic average density haas been specified as an invalid "
                "string. Must be {local, profile}"
            )

        cosmic_average_density = (
            cosmology.cosmic_average_density_solar_mass_per_kpc3_from(
                redshift=redshift_calc
            )
        )

        radius_at_200 = self.radius_at_200(
            redshift_object=redshift_object,
            redshift_source=redshift_source,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            cosmology=cosmology,
        )

        kpc_per_arcsec = cosmology.kpc_per_arcsec_from(redshift=redshift_object)

        radius_at_200_kpc = radius_at_200 * kpc_per_arcsec

        return (
            200.0
            * ((4.0 / 3.0) * np.pi)
            * cosmic_average_density
            * (radius_at_200_kpc**3.0)
        )

    @property
    def ellipticity_rescale(self):
        return 1.0 - ((1.0 - self.axis_ratio) / 2.0)
