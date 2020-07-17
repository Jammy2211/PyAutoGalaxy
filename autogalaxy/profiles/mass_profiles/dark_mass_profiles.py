import inspect
import typing

import autofit as af
import numpy as np
from astropy import cosmology as cosmo
from astropy import units
from autoarray import decorator_util
from autoarray.structures import arrays, grids
from autofit.text import formatter
from autogalaxy import dimensions as dim
from autogalaxy import exc
from autogalaxy.profiles import mass_profiles as mp
from autogalaxy.util import cosmology_util
from colossus.cosmology import cosmology as col_cosmology
from colossus.halo.concentration import concentration as col_concentration
from numba import cfunc
from numba.types import intc, CPointer, float64
from pyquad import quad_grid
from scipy import LowLevelCallable
from scipy import special
from scipy.integrate import quad
from scipy.optimize import fsolve


def jit_integrand(integrand_function):

    jitted_function = decorator_util.jit(nopython=True, cache=True)(integrand_function)
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


class DarkProfile:

    pass


# noinspection PyAbstractClass
class AbstractEllipticalGeneralizedNFW(
    mp.EllipticalMassProfile, mp.MassProfile, DarkProfile
):
    epsrel = 1.49e-5

    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0),
        kappa_s: float = 0.05,
        inner_slope: float = 1.0,
        scale_radius: dim.Length = 1.0,
    ):
        """
        The elliptical NFW profiles, used to fit the dark matter halo of the lens.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps : (float, float)
            The first and second ellipticity components of the elliptical coordinate system, where
            fac = (1 - axis_ratio) / (1 + axis_ratio), ellip_y = fac * sin(2*phi) and ellip_x = fac * cos(2*phi).
        kappa_s : float
            The overall normalization of the dark matter halo \
            (kappa_s = (rho_s * scale_radius)/lensing_critical_density)
        inner_slope : float
            The inner slope of the dark matter halo
        scale_radius : float
            The arc-second radius where the average density within this radius is 200 times the critical density of \
            the Universe..
        """

        super(AbstractEllipticalGeneralizedNFW, self).__init__(
            centre=centre, elliptical_comps=elliptical_comps
        )
        super(mp.MassProfile, self).__init__()
        self.kappa_s = kappa_s
        self.scale_radius = scale_radius
        self.inner_slope = inner_slope

    def tabulate_integral(self, grid, tabulate_bins):
        """Tabulate an integral over the convergence of deflection potential of a mass profile. This is used in \
        the GeneralizedNFW profile classes to speed up the integration procedure.

        Parameters
        -----------
        grid : aa.Grid
            The grid of (y,x) arc-second coordinates the potential / deflection_stacks are computed on.
        tabulate_bins : int
            The number of bins to tabulate the inner integral of this profile.
        """
        eta_min = 1.0e-4
        eta_max = 1.05 * np.max(self.grid_to_elliptical_radii(grid))

        minimum_log_eta = np.log10(eta_min)
        maximum_log_eta = np.log10(eta_max)
        bin_size = (maximum_log_eta - minimum_log_eta) / (tabulate_bins - 1)

        return eta_min, eta_max, minimum_log_eta, maximum_log_eta, bin_size

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def convergence_from_grid(self, grid):
        """ Calculate the projected convergence at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : aa.Grid
            The grid of (y,x) arc-second coordinates the convergence is computed on.

        """

        grid_eta = self.grid_to_elliptical_radii(grid=grid)

        return self.convergence_func(grid_radius=grid_eta)

    @property
    def ellipticity_rescale(self):
        return 1.0 - ((1.0 - self.axis_ratio) / 2.0)

    def coord_func_f(self, grid_radius):
        if isinstance(grid_radius, np.ndarray):
            return self.coord_func_f_jit(
                grid_radius=grid_radius,
                f=np.ones(shape=grid_radius.shape[0], dtype="complex64"),
            )
        else:
            return self.coord_func_f_float_jit(grid_radius=grid_radius)

    @staticmethod
    @decorator_util.jit()
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
    @decorator_util.jit()
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
    @decorator_util.jit()
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
    @decorator_util.jit()
    def coord_func_g_float_jit(grid_radius, f_r):

        if np.real(grid_radius) > 1.0:
            return (1.0 - f_r) / (np.square(grid_radius) - 1.0)
        elif np.real(grid_radius) < 1.0:
            return (f_r - 1.0) / (1.0 - np.square(grid_radius))
        else:
            return 1.0 / 3.0

    def coord_func_h(self, grid_radius):
        return np.log(grid_radius / 2.0) + self.coord_func_f(grid_radius=grid_radius)

    def rho_at_scale_radius_for_units(
        self,
        redshift_object,
        redshift_source,
        unit_length="arcsec",
        unit_mass="solMass",
        cosmology=cosmo.Planck15,
    ):
        """The Cosmic average density is defined at the redshift of the profile."""

        kpc_per_arcsec = cosmology_util.kpc_per_arcsec_from(
            redshift=redshift_object, cosmology=cosmology
        )

        critical_surface_density = cosmology_util.critical_surface_density_between_redshifts_from(
            redshift_0=redshift_object,
            redshift_1=redshift_source,
            cosmology=cosmology,
            unit_length=self.unit_length,
            unit_mass=unit_mass,
        )

        rho_at_scale_radius = (
            self.kappa_s * critical_surface_density / self.scale_radius
        )

        rho_at_scale_radius = dim.MassOverLength3(
            value=rho_at_scale_radius, unit_length=self.unit_length, unit_mass=unit_mass
        )
        return rho_at_scale_radius.convert(
            unit_length=unit_length,
            unit_mass=unit_mass,
            kpc_per_arcsec=kpc_per_arcsec,
            critical_surface_density=critical_surface_density,
        )

    def delta_concentration_for_units(
        self,
        redshift_object,
        redshift_source,
        unit_length="arcsec",
        unit_mass="solMass",
        redshift_of_cosmic_average_density="profile",
        cosmology=cosmo.Planck15,
    ):

        if redshift_of_cosmic_average_density is "profile":
            redshift_calc = redshift_object
        elif redshift_of_cosmic_average_density is "local":
            redshift_calc = 0.0
        else:
            raise exc.UnitsException(
                "The redshift of the cosmic average density haas been specified as an invalid "
                "string. Must be {local, profile}"
            )

        cosmic_average_density = cosmology_util.cosmic_average_density_from(
            redshift=redshift_calc,
            cosmology=cosmology,
            unit_length=unit_length,
            unit_mass=unit_mass,
        )

        rho_scale_radius = self.rho_at_scale_radius_for_units(
            unit_length=unit_length,
            unit_mass=unit_mass,
            redshift_object=redshift_object,
            redshift_source=redshift_source,
            cosmology=cosmology,
        )

        return rho_scale_radius / cosmic_average_density

    def concentration_for_units(
        self,
        redshift_profile,
        redshift_source,
        unit_length="arcsec",
        unit_mass="solMass",
        redshift_of_cosmic_average_density="profile",
        cosmology=cosmo.Planck15,
    ):
        delta_concentration = self.delta_concentration_for_units(
            redshift_object=redshift_profile,
            redshift_source=redshift_source,
            unit_length=unit_length,
            unit_mass=unit_mass,
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

    def radius_at_200_for_units(
        self,
        redshift_object,
        redshift_source,
        unit_length="arcsec",
        unit_mass="solMass",
        redshift_of_cosmic_average_density="profile",
        cosmology=cosmo.Planck15,
    ):

        kpc_per_arcsec = cosmology_util.kpc_per_arcsec_from(
            redshift=redshift_object, cosmology=cosmology
        )

        concentration = self.concentration_for_units(
            redshift_profile=redshift_object,
            redshift_source=redshift_source,
            unit_length=unit_length,
            unit_mass=unit_mass,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            cosmology=cosmology,
        )

        radius_at_200 = dim.Length(
            value=concentration * self.scale_radius, unit_length=self.unit_length
        )

        return radius_at_200.convert(
            unit_length=unit_length, kpc_per_arcsec=kpc_per_arcsec
        )

    def mass_at_200_for_units(
        self,
        redshift_object,
        redshift_source,
        unit_length="arcsec",
        unit_mass="solMass",
        redshift_of_cosmic_average_density="profile",
        cosmology=cosmo.Planck15,
    ):

        if redshift_of_cosmic_average_density is "profile":
            redshift_calc = redshift_object
        elif redshift_of_cosmic_average_density is "local":
            redshift_calc = 0.0
        else:
            raise exc.UnitsException(
                "The redshift of the cosmic average density haas been specified as an invalid "
                "string. Must be {local, profile}"
            )

        cosmic_average_density = cosmology_util.cosmic_average_density_from(
            redshift=redshift_calc,
            cosmology=cosmology,
            unit_length=unit_length,
            unit_mass=unit_mass,
        )

        critical_surface_density = cosmology_util.critical_surface_density_between_redshifts_from(
            redshift_0=redshift_object,
            redshift_1=redshift_source,
            cosmology=cosmology,
            unit_length=self.unit_length,
            unit_mass=unit_mass,
        )

        radius_at_200 = self.radius_at_200_for_units(
            redshift_object=redshift_object,
            redshift_source=redshift_source,
            unit_length=unit_length,
            unit_mass=unit_mass,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            cosmology=cosmology,
        )

        mass_at_200 = dim.Mass(
            200.0
            * ((4.0 / 3.0) * np.pi)
            * cosmic_average_density
            * (radius_at_200 ** 3.0),
            unit_mass=unit_mass,
        )

        return mass_at_200.convert(
            unit_mass=unit_mass, critical_surface_density=critical_surface_density
        )

    def summarize_in_units(
        self,
        radii,
        prefix="",
        whitespace=80,
        unit_length="arcsec",
        unit_mass="solMass",
        redshift_profile=None,
        redshift_source=None,
        redshift_of_cosmic_average_density="profile",
        cosmology=cosmo.Planck15,
    ):
        summary = super().summarize_in_units(
            radii=radii,
            prefix=prefix,
            unit_length=unit_length,
            unit_mass=unit_mass,
            redshift_profile=redshift_profile,
            redshift_source=redshift_source,
            cosmology=cosmology,
            whitespace=whitespace,
        )

        rho_at_scale_radius = self.rho_at_scale_radius_for_units(
            unit_length=unit_length,
            unit_mass=unit_mass,
            redshift_object=redshift_profile,
            redshift_source=redshift_source,
            cosmology=cosmology,
        )

        summary += [
            formatter.label_value_and_unit_string(
                label=prefix + "rho_at_scale_radius",
                value=rho_at_scale_radius,
                unit=unit_mass + "/" + unit_length + "3",
                whitespace=whitespace,
            )
        ]

        delta_concentration = self.delta_concentration_for_units(
            unit_length=unit_length,
            unit_mass=unit_mass,
            redshift_object=redshift_profile,
            redshift_source=redshift_source,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            cosmology=cosmology,
        )

        summary += [
            formatter.label_and_value_string(
                label=prefix + "delta_concentration",
                value=delta_concentration,
                whitespace=whitespace,
            )
        ]

        concentration = self.concentration_for_units(
            unit_length=unit_length,
            unit_mass=unit_mass,
            redshift_profile=redshift_profile,
            redshift_source=redshift_source,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            cosmology=cosmology,
        )

        summary += [
            formatter.label_and_value_string(
                label=prefix + "concentration",
                value=concentration,
                whitespace=whitespace,
            )
        ]

        radius_at_200 = self.radius_at_200_for_units(
            unit_length=unit_length,
            unit_mass=unit_mass,
            redshift_object=redshift_profile,
            redshift_source=redshift_source,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            cosmology=cosmology,
        )

        summary += [
            formatter.label_value_and_unit_string(
                label=prefix + "radius_at_200x_cosmic_density",
                value=radius_at_200,
                unit=unit_length,
                whitespace=whitespace,
            )
        ]

        mass_at_200 = self.mass_at_200_for_units(
            unit_length=unit_length,
            unit_mass=unit_mass,
            redshift_object=redshift_profile,
            redshift_source=redshift_source,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            cosmology=cosmology,
        )

        summary += [
            formatter.label_value_and_unit_string(
                label=prefix + "mass_at_200x_cosmic_density",
                value=mass_at_200,
                unit=unit_mass,
                whitespace=whitespace,
            )
        ]

        return summary

    @property
    def unit_mass(self):
        return "angular"


class EllipticalGeneralizedNFW(AbstractEllipticalGeneralizedNFW):
    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def potential_from_grid(self, grid, tabulate_bins=1000):
        """
        Calculate the potential at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : aa.Grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        tabulate_bins : int
            The number of bins to tabulate the inner integral of this profile.

        """

        @jit_integrand
        def deflection_integrand(x, kappa_radius, scale_radius, inner_slope):
            return (x + kappa_radius / scale_radius) ** (inner_slope - 3) * (
                (1 - np.sqrt(1 - x ** 2)) / x
            )

        eta_min, eta_max, minimum_log_eta, maximum_log_eta, bin_size = self.tabulate_integral(
            grid, tabulate_bins
        )

        potential_grid = np.zeros(grid.shape[0])

        deflection_integral = np.zeros((tabulate_bins,))

        for i in range(tabulate_bins):
            eta = 10.0 ** (minimum_log_eta + (i - 1) * bin_size)

            integral = quad(
                deflection_integrand,
                a=0.0,
                b=1.0,
                args=(eta, self.scale_radius, self.inner_slope),
                epsrel=EllipticalGeneralizedNFW.epsrel,
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
                epsrel=EllipticalGeneralizedNFW.epsrel,
            )[0]

        return potential_grid

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def deflections_from_grid(self, grid, tabulate_bins=1000):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : aa.Grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        tabulate_bins : int
            The number of bins to tabulate the inner integral of this profile.

        """

        @jit_integrand
        def surface_density_integrand(x, kappa_radius, scale_radius, inner_slope):
            return (
                (3 - inner_slope)
                * (x + kappa_radius / scale_radius) ** (inner_slope - 4)
                * (1 - np.sqrt(1 - x * x))
            )

        def calculate_deflection_component(npow, index):
            deflection_grid = 2.0 * self.kappa_s * self.axis_ratio * grid[:, index]
            deflection_grid *= quad_grid(
                self.deflection_func,
                0.0,
                1.0,
                grid,
                args=(
                    npow,
                    self.axis_ratio,
                    minimum_log_eta,
                    maximum_log_eta,
                    tabulate_bins,
                    surface_density_integral,
                ),
                epsrel=EllipticalGeneralizedNFW.epsrel,
            )[0]

            return deflection_grid

        eta_min, eta_max, minimum_log_eta, maximum_log_eta, bin_size = self.tabulate_integral(
            grid, tabulate_bins
        )

        surface_density_integral = np.zeros((tabulate_bins,))

        for i in range(tabulate_bins):
            eta = 10.0 ** (minimum_log_eta + (i - 1) * bin_size)

            integral = quad(
                surface_density_integrand,
                a=0.0,
                b=1.0,
                args=(eta, self.scale_radius, self.inner_slope),
                epsrel=EllipticalGeneralizedNFW.epsrel,
            )[0]

            surface_density_integral[i] = (
                (eta / self.scale_radius) ** (1 - self.inner_slope)
            ) * (((1 + eta / self.scale_radius) ** (self.inner_slope - 3)) + integral)

        deflection_y = calculate_deflection_component(1.0, 0)
        deflection_x = calculate_deflection_component(0.0, 1)

        return self.rotate_grid_from_profile(
            np.multiply(1.0, np.vstack((deflection_y, deflection_x)).T)
        )

    def convergence_func(self, grid_radius):
        def integral_y(y, eta):
            return (y + eta) ** (self.inner_slope - 4) * (1 - np.sqrt(1 - y ** 2))

        grid_radius = (1.0 / self.scale_radius) * grid_radius

        for index in range(grid_radius.shape[0]):

            integral_y_value = quad(
                integral_y,
                a=0.0,
                b=1.0,
                args=grid_radius[index],
                epsrel=EllipticalGeneralizedNFW.epsrel,
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

    @staticmethod
    # TODO : Decorator needs to know that potential_integral is 1D arrays
    #    @jit_integrand
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
        eta_u = np.sqrt((u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u)))))
        bin_size = (maximum_log_eta - minimum_log_eta) / (tabulate_bins - 1)
        i = 1 + int((np.log10(eta_u) - minimum_log_eta) / bin_size)
        r1 = 10.0 ** (minimum_log_eta + (i - 1) * bin_size)
        r2 = r1 * 10.0 ** bin_size
        phi = potential_integral[i] + (
            potential_integral[i + 1] - potential_integral[i]
        ) * (eta_u - r1) / (r2 - r1)
        return eta_u * (phi / u) / (1.0 - (1.0 - axis_ratio ** 2) * u) ** 0.5

    @staticmethod
    # TODO : Decorator needs to know that surface_density_integral is 1D arrays
    #    @jit_integrand
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

        eta_u = np.sqrt((u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u)))))
        bin_size = (maximum_log_eta - minimum_log_eta) / (tabulate_bins - 1)
        i = 1 + int((np.log10(eta_u) - minimum_log_eta) / bin_size)
        r1 = 10.0 ** (minimum_log_eta + (i - 1) * bin_size)
        r2 = r1 * 10.0 ** bin_size
        kap = surface_density_integral[i] + (
            surface_density_integral[i + 1] - surface_density_integral[i]
        ) * (eta_u - r1) / (r2 - r1)
        return kap / (1.0 - (1.0 - axis_ratio ** 2) * u) ** (npow + 0.5)


class SphericalGeneralizedNFW(EllipticalGeneralizedNFW):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        kappa_s: float = 0.05,
        inner_slope: float = 1.0,
        scale_radius: dim.Length = 1.0,
    ):
        """
        The spherical NFW profiles, used to fit the dark matter halo of the lens.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        kappa_s : float
            The overall normalization of the dark matter halo \
            (kappa_s = (rho_s * scale_radius)/lensing_critical_density)
        inner_slope : float
            The inner slope of the dark matter halo.
        scale_radius : float
            The arc-second radius where the average density within this radius is 200 times the critical density of \
            the Universe..
        """

        super(SphericalGeneralizedNFW, self).__init__(
            centre=centre,
            elliptical_comps=(0.0, 0.0),
            kappa_s=kappa_s,
            inner_slope=inner_slope,
            scale_radius=scale_radius,
        )

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def deflections_from_grid(self, grid, **kwargs):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : aa.Grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """

        eta = np.multiply(1.0 / self.scale_radius, self.grid_to_grid_radii(grid))

        deflection_grid = np.zeros(grid.shape[0])

        for i in range(grid.shape[0]):
            deflection_grid[i] = np.multiply(
                4.0 * self.kappa_s * self.scale_radius, self.deflection_func_sph(eta[i])
            )

        return self.grid_to_grid_cartesian(grid, deflection_grid)

    @staticmethod
    def deflection_integrand(y, eta, inner_slope):
        return (y + eta) ** (inner_slope - 3) * ((1 - np.sqrt(1 - y ** 2)) / y)

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


class SphericalTruncatedNFW(AbstractEllipticalGeneralizedNFW):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        kappa_s: float = 0.05,
        scale_radius: dim.Length = 1.0,
        truncation_radius: dim.Length = 2.0,
    ):
        super(SphericalTruncatedNFW, self).__init__(
            centre=centre,
            elliptical_comps=(0.0, 0.0),
            kappa_s=kappa_s,
            inner_slope=1.0,
            scale_radius=scale_radius,
        )

        self.truncation_radius = truncation_radius
        self.tau = self.truncation_radius / self.scale_radius

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

        return np.divide(self.tau ** 2.0, (self.tau ** 2.0 + 1.0) ** 2.0) * (
            ((self.tau ** 2.0 + 1.0) * g_r)
            + (2 * f_r)
            - (np.pi / (np.sqrt(self.tau ** 2.0 + grid_radius ** 2.0)))
            + (
                (
                    (self.tau ** 2.0 - 1.0)
                    / (self.tau * (np.sqrt(self.tau ** 2.0 + grid_radius ** 2.0)))
                )
                * k_r
            )
        )

    def coord_func_m(self, grid_radius):
        f_r = self.coord_func_f(grid_radius=grid_radius)
        k_r = self.coord_func_k(grid_radius=grid_radius)

        return (self.tau ** 2.0 / (self.tau ** 2.0 + 1.0) ** 2.0) * (
            ((self.tau ** 2.0 + 2.0 * grid_radius ** 2.0 - 1.0) * f_r)
            + (np.pi * self.tau)
            + ((self.tau ** 2.0 - 1.0) * np.log(self.tau))
            + (
                np.sqrt(grid_radius ** 2.0 + self.tau ** 2.0)
                * (((self.tau ** 2.0 - 1.0) / self.tau) * k_r - np.pi)
            )
        )

    def convergence_func(self, grid_radius):
        grid_radius = ((1.0 / self.scale_radius) * grid_radius) + 0j
        return np.real(2.0 * self.kappa_s * self.coord_func_l(grid_radius=grid_radius))

    def deflection_func_sph(self, grid_radius):
        grid_radius = grid_radius + 0j
        return np.real(self.coord_func_m(grid_radius=grid_radius))

    def potential_from_grid(self, grid):
        return arrays.Array.manual_1d(
            array=np.zeros(shape=grid.shape[0]), shape_2d=grid.sub_shape_2d
        )

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def deflections_from_grid(self, grid, **kwargs):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : aa.Grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """

        eta = np.multiply(1.0 / self.scale_radius, self.grid_to_grid_radii(grid=grid))

        deflection_grid = np.multiply(
            (4.0 * self.kappa_s * self.scale_radius / eta),
            self.deflection_func_sph(grid_radius=eta),
        )

        return self.grid_to_grid_cartesian(grid, deflection_grid)

    def mass_at_truncation_radius(
        self,
        redshift_profile,
        redshift_source,
        unit_length="arcsec",
        unit_mass="solMass",
        redshift_of_cosmic_average_density="profile",
        cosmology=cosmo.Planck15,
    ):
        mass_at_200 = self.mass_at_200_for_units(
            redshift_object=redshift_profile,
            redshift_source=redshift_source,
            unit_length=unit_length,
            unit_mass=unit_mass,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            cosmology=cosmology,
        )

        return (
            mass_at_200
            * (self.tau ** 2.0 / (self.tau ** 2.0 + 1.0) ** 2.0)
            * (
                ((self.tau ** 2.0 - 1) * np.log(self.tau))
                + (self.tau * np.pi)
                - (self.tau ** 2.0 + 1)
            )
        )

    def summarize_in_units(
        self,
        radii,
        prefix="",
        whitespace=80,
        unit_length="arcsec",
        unit_mass="solMass",
        redshift_profile=None,
        redshift_source=None,
        redshift_of_cosmic_average_density="profile",
        cosmology=cosmo.Planck15,
    ):
        summary = super().summarize_in_units(
            radii=radii,
            prefix=prefix,
            whitespace=whitespace,
            unit_length=unit_length,
            unit_mass=unit_mass,
            redshift_profile=redshift_profile,
            redshift_source=redshift_source,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            cosmology=cosmology,
        )

        mass_at_truncation_radius = self.mass_at_truncation_radius(
            unit_length=unit_length,
            unit_mass=unit_mass,
            redshift_profile=redshift_profile,
            redshift_source=redshift_source,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            cosmology=cosmology,
        )

        summary += [
            formatter.label_value_and_unit_string(
                label=prefix + "mass_at_truncation_radius",
                value=mass_at_truncation_radius,
                unit=unit_mass,
                whitespace=whitespace,
            )
        ]

        return summary


class SphericalTruncatedNFWMCRDuffy(SphericalTruncatedNFW):
    """
    This function only applies for the lens configuration as follows:
    Cosmology: FlatLamdaCDM
    H0 = 70.0 km/sec/Mpc
    Omega_Lambda = 0.7
    Omega_m = 0.3
    Redshfit of Main Lens: 0.6
    Redshift of Source: 2.5
    A truncated NFW halo at z = 0.6 with tau = 2.0
    """

    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        mass_at_200: float = 1e9,
        redshift_object: float = 0.5,
        redshift_source: float = 1.0,
    ):
        """
        Input m200: The m200 of the NFW part of the corresponding tNFW part. Unit: M_sun.
        """

        self.mass_at_200_input = mass_at_200

        kappa_s, scale_radius, radius_at_200 = kappa_s_and_scale_radius_for_duffy(
            mass_at_200=mass_at_200,
            redshift_object=redshift_object,
            redshift_source=redshift_source,
        )

        super(SphericalTruncatedNFWMCRDuffy, self).__init__(
            centre=centre,
            kappa_s=kappa_s,
            scale_radius=scale_radius,
            truncation_radius=2.0 * radius_at_200,
        )


class SphericalTruncatedNFWMCRLudlow(SphericalTruncatedNFW):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        mass_at_200: float = 1e9,
        redshift_object: float = 0.5,
        redshift_source: float = 1.0,
    ):
        """
        Input m200: The m200 of the NFW part of the corresponding tNFW part. Unit: M_sun.
        """

        self.mass_at_200_input = mass_at_200

        kappa_s, scale_radius, radius_at_200 = kappa_s_and_scale_radius_for_ludlow(
            mass_at_200=mass_at_200,
            redshift_object=redshift_object,
            redshift_source=redshift_source,
        )

        super(SphericalTruncatedNFWMCRLudlow, self).__init__(
            centre=centre,
            kappa_s=kappa_s,
            scale_radius=scale_radius,
            truncation_radius=2.0 * radius_at_200,
        )


class SphericalTruncatedNFWChallenge(SphericalTruncatedNFW):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        kappa_s: float = 0.05,
        scale_radius: dim.Length = 1.0,
    ):
        def solve_c(c, de_c):
            """
            Equation need for solving concentration c for a given delta_c
            """
            return 200.0 / 3.0 * (c * c * c / (np.log(1 + c) - c / (1 + c))) - de_c

        kpc_per_arcsec = 6.68549148608755
        scale_radius_kpc = scale_radius * kpc_per_arcsec
        cosmic_average_density = 262.30319684750657
        critical_surface_density = 1940654909.413325
        rho_s = kappa_s * critical_surface_density / scale_radius_kpc
        de_c = rho_s / cosmic_average_density  # delta_c
        concentration = fsolve(solve_c, 10.0, args=(de_c,))[0]
        r200 = concentration * scale_radius_kpc / kpc_per_arcsec  # R_200

        super(SphericalTruncatedNFWChallenge, self).__init__(
            centre=centre,
            kappa_s=kappa_s,
            scale_radius=scale_radius,
            truncation_radius=2.0 * r200,
        )


class SphericalTruncatedNFWMCRChallenge(SphericalTruncatedNFW):
    """
    This function only applies for the lens configuration as follows:
    Cosmology: FlatLamdaCDM
    H0 = 70.0 km/sec/Mpc
    Omega_Lambda = 0.7
    Omega_m = 0.3
    Redshfit of Main Lens: 0.6
    Redshift of Source: 2.5
    A truncated NFW halo at z = 0.6 with tau = 2.0
    """

    @af.map_types
    def __init__(self, centre: dim.Position = (0.0, 0.0), mass_at_200: float = 1e9):
        """
        Input m200: The m200 of the NFW part of the corresponding tNFW part. Unit: M_sun.
        """

        cosmic_average_density = (
            262.30319684751
        )  # Critical Density at z=0.6 M_sun/kpc^3
        critical_surface_density = (
            1940654909.413325
        )  # Lensing Critical Surface Density for lens at z=0.6, source at z=2.5. M_sun/kpc^2
        kpc_per_arcsec = 6.685491486088  # 1 arcsec = 6.685491486 kpc at z=0.6

        concentration = 11.5 * (mass_at_200 / 1e10 + (mass_at_200 / 1e10) ** 2.0) ** (
            -0.05
        )  # This is the challenge setting.
        radius_at_200 = (
            mass_at_200 / (200.0 * cosmic_average_density * (4.0 * np.pi / 3.0))
        ) ** (1.0 / 3.0)
        de_c = (
            200.0
            / 3.0
            * (
                concentration
                * concentration
                * concentration
                / (np.log(1.0 + concentration) - concentration / (1.0 + concentration))
            )
        )
        scale_radius_kpc = radius_at_200 / concentration
        rho_s = cosmic_average_density * de_c
        kappa_s = rho_s * scale_radius_kpc / critical_surface_density
        scale_radius = scale_radius_kpc / kpc_per_arcsec

        super(SphericalTruncatedNFWMCRChallenge, self).__init__(
            centre=centre,
            kappa_s=kappa_s,
            scale_radius=scale_radius,
            truncation_radius=2.0 * radius_at_200,
        )


class EllipticalNFW(AbstractEllipticalGeneralizedNFW):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0),
        kappa_s: float = 0.05,
        scale_radius: dim.Length = 1.0,
    ):
        """
        The elliptical NFW profiles, used to fit the dark matter halo of the lens.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps : (float, float)
            The first and second ellipticity components of the elliptical coordinate system, where
            fac = (1 - axis_ratio) / (1 + axis_ratio), ellip_y = fac * sin(2*phi) and ellip_x = fac * cos(2*phi).
        kappa_s : float
            The overall normalization of the dark matter halo \
            (kappa_s = (rho_s * scale_radius)/lensing_critical_density)
        scale_radius : float
            The arc-second radius where the average density within this radius is 200 times the critical density of \
            the Universe..
        """

        super(EllipticalNFW, self).__init__(
            centre=centre,
            elliptical_comps=elliptical_comps,
            kappa_s=kappa_s,
            inner_slope=1.0,
            scale_radius=scale_radius,
        )

    @staticmethod
    def coord_func(r):
        if r > 1:
            return (1.0 / np.sqrt(r ** 2 - 1)) * np.arctan(np.sqrt(r ** 2 - 1))
        elif r < 1:
            return (1.0 / np.sqrt(1 - r ** 2)) * np.arctanh(np.sqrt(1 - r ** 2))
        elif r == 1:
            return 1

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def potential_from_grid(self, grid):
        """
        Calculate the potential at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : aa.Grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """
        potential_grid = quad_grid(
            self.potential_func,
            0.0,
            1.0,
            grid,
            args=(self.axis_ratio, self.kappa_s, self.scale_radius),
            epsrel=1.49e-5,
        )[0]

        return potential_grid

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def deflections_from_grid(self, grid):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : aa.Grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """

        def calculate_deflection_component(npow, index):
            deflection_grid = self.axis_ratio * grid[:, index]
            deflection_grid *= (
                self.kappa_s
                * quad_grid(
                    self.deflection_func,
                    0.0,
                    1.0,
                    grid,
                    args=(npow, self.axis_ratio, self.scale_radius),
                )[0]
            )

            return deflection_grid

        deflection_y = calculate_deflection_component(1.0, 0)
        deflection_x = calculate_deflection_component(0.0, 1)

        return self.rotate_grid_from_profile(
            np.multiply(1.0, np.vstack((deflection_y, deflection_x)).T)
        )

    def convergence_func(self, grid_radius):
        grid_radius = (1.0 / self.scale_radius) * grid_radius + 0j
        return np.real(2.0 * self.kappa_s * self.coord_func_g(grid_radius=grid_radius))

    @staticmethod
    def potential_func(u, y, x, axis_ratio, kappa_s, scale_radius):
        eta_u = (1.0 / scale_radius) * np.sqrt(
            (u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u))))
        )

        if eta_u > 1:
            eta_u_2 = (1.0 / np.sqrt(eta_u ** 2 - 1)) * np.arctan(
                np.sqrt(eta_u ** 2 - 1)
            )
        elif eta_u < 1:
            eta_u_2 = (1.0 / np.sqrt(1 - eta_u ** 2)) * np.arctanh(
                np.sqrt(1 - eta_u ** 2)
            )
        else:
            eta_u_2 = 1

        return (
            4.0
            * kappa_s
            * scale_radius
            * (axis_ratio / 2.0)
            * (eta_u / u)
            * ((np.log(eta_u / 2.0) + eta_u_2) / eta_u)
            / ((1 - (1 - axis_ratio ** 2) * u) ** 0.5)
        )

    @staticmethod
    def deflection_func(u, y, x, npow, axis_ratio, scale_radius):
        eta_u = (1.0 / scale_radius) * np.sqrt(
            (u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u))))
        )

        if eta_u > 1:
            eta_u_2 = (1.0 / np.sqrt(eta_u ** 2 - 1)) * np.arctan(
                np.sqrt(eta_u ** 2 - 1)
            )
        elif eta_u < 1:
            eta_u_2 = (1.0 / np.sqrt(1 - eta_u ** 2)) * np.arctanh(
                np.sqrt(1 - eta_u ** 2)
            )
        else:
            eta_u_2 = 1

        return (
            2.0
            * (1 - eta_u_2)
            / (eta_u ** 2 - 1)
            / ((1 - (1 - axis_ratio ** 2) * u) ** (npow + 0.5))
        )


class SphericalNFW(EllipticalNFW):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        kappa_s: float = 0.05,
        scale_radius: dim.Length = 1.0,
    ):
        """
        The spherical NFW profiles, used to fit the dark matter halo of the lens.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        kappa_s : float
            The overall normalization of the dark matter halo \
            (kappa_s = (rho_s * scale_radius)/lensing_critical_density)
        scale_radius : float
            The arc-second radius where the average density within this radius is 200 times the critical density of \
            the Universe..
        """

        super(SphericalNFW, self).__init__(
            centre=centre,
            elliptical_comps=(0.0, 0.0),
            kappa_s=kappa_s,
            scale_radius=scale_radius,
        )

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def potential_from_grid(self, grid):
        """
        Calculate the potential at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : aa.Grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """
        eta = (1.0 / self.scale_radius) * self.grid_to_grid_radii(grid) + 0j
        return np.real(
            2.0 * self.scale_radius * self.kappa_s * self.potential_func_sph(eta)
        )

    def deflection_func_sph(self, grid_radius):
        grid_radius = grid_radius + 0j
        return np.real(self.coord_func_h(grid_radius=grid_radius))

    @staticmethod
    def potential_func_sph(eta):
        return ((np.log(eta / 2.0)) ** 2) - (np.arctanh(np.sqrt(1 - eta ** 2))) ** 2

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def deflections_from_grid(self, grid, **kwargs):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : aa.Grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """

        eta = np.multiply(1.0 / self.scale_radius, self.grid_to_grid_radii(grid=grid))

        deflection_grid = np.multiply(
            (4.0 * self.kappa_s * self.scale_radius / eta),
            self.deflection_func_sph(grid_radius=eta),
        )

        return self.grid_to_grid_cartesian(grid, deflection_grid)


class SphericalNFWMCRDuffy(SphericalNFW):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        mass_at_200: float = 1e9,
        redshift_object: float = 0.5,
        redshift_source: float = 1.0,
    ):

        self.mass_at_200_input = mass_at_200

        kappa_s, scale_radius, radius_at_200 = kappa_s_and_scale_radius_for_duffy(
            mass_at_200=mass_at_200,
            redshift_object=redshift_object,
            redshift_source=redshift_source,
        )

        super(SphericalNFWMCRDuffy, self).__init__(
            centre=centre, kappa_s=kappa_s, scale_radius=scale_radius
        )


class SphericalNFWMCRLudlow(SphericalNFW):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        mass_at_200: float = 1e9,
        redshift_object: float = 0.5,
        redshift_source: float = 1.0,
    ):

        self.mass_at_200_input = mass_at_200

        kappa_s, scale_radius, radius_at_200 = kappa_s_and_scale_radius_for_ludlow(
            mass_at_200=mass_at_200,
            redshift_object=redshift_object,
            redshift_source=redshift_source,
        )

        super(SphericalNFWMCRLudlow, self).__init__(
            centre=centre, kappa_s=kappa_s, scale_radius=scale_radius
        )


def kappa_s_and_scale_radius_for_duffy(mass_at_200, redshift_object, redshift_source):

    cosmology = cosmo.Planck15

    cosmic_average_density = (
        cosmology.critical_density(redshift_object).to(units.solMass / units.kpc ** 3)
    ).value

    critical_surface_density = cosmology_util.critical_surface_density_between_redshifts_from(
        redshift_0=redshift_object,
        redshift_1=redshift_source,
        cosmology=cosmology,
        unit_length="kpc",
        unit_mass="solMass",
    )

    kpc_per_arcsec = cosmology_util.kpc_per_arcsec_from(
        redshift=redshift_object, cosmology=cosmology
    )

    radius_at_200 = (
        mass_at_200 / (200.0 * cosmic_average_density * (4.0 * np.pi / 3.0))
    ) ** (
        1.0 / 3.0
    )  # r200
    coefficient = 5.71 * (1.0 + redshift_object) ** (
        -0.47
    )  # The coefficient of Duffy mass-concentration (Duffy+2008)
    concentration = coefficient * (mass_at_200 / 2.952465309e12) ** (
        -0.084
    )  # mass-concentration relation. (Duffy+2008)
    de_c = (
        200.0
        / 3.0
        * (
            concentration ** 3
            / (np.log(1.0 + concentration) - concentration / (1.0 + concentration))
        )
    )  # rho_c

    scale_radius_kpc = radius_at_200 / concentration  # scale radius in kpc
    rho_s = cosmic_average_density * de_c  # rho_s
    kappa_s = rho_s * scale_radius_kpc / critical_surface_density  # kappa_s
    scale_radius = scale_radius_kpc / kpc_per_arcsec  # scale radius in arcsec

    return kappa_s, scale_radius, radius_at_200


def kappa_s_and_scale_radius_for_ludlow(mass_at_200, redshift_object, redshift_source):

    cosmology = cosmo.Planck15

    col_cosmo = col_cosmology.setCosmology("planck15")
    m_input = mass_at_200 * col_cosmo.h
    concentration = col_concentration(
        m_input, "200c", redshift_object, model="ludlow16"
    )

    cosmic_average_density = (
        cosmology.critical_density(redshift_object).to(units.solMass / units.kpc ** 3)
    ).value

    critical_surface_density = cosmology_util.critical_surface_density_between_redshifts_from(
        redshift_0=redshift_object,
        redshift_1=redshift_source,
        cosmology=cosmology,
        unit_length="kpc",
        unit_mass="solMass",
    )

    kpc_per_arcsec = cosmology_util.kpc_per_arcsec_from(
        redshift=redshift_object, cosmology=cosmology
    )

    radius_at_200 = (
        mass_at_200 / (200.0 * cosmic_average_density * (4.0 * np.pi / 3.0))
    ) ** (
        1.0 / 3.0
    )  # r200

    de_c = (
        200.0
        / 3.0
        * (
            concentration ** 3
            / (np.log(1.0 + concentration) - concentration / (1.0 + concentration))
        )
    )  # rho_c

    scale_radius_kpc = radius_at_200 / concentration  # scale radius in kpc
    rho_s = cosmic_average_density * de_c  # rho_s
    kappa_s = rho_s * scale_radius_kpc / critical_surface_density  # kappa_s
    scale_radius = scale_radius_kpc / kpc_per_arcsec  # scale radius in arcsec

    return kappa_s, scale_radius, radius_at_200
