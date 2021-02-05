import typing

import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.special import wofz, comb
import copy

from autoarray.structures import grids
from autogalaxy import lensing
from autogalaxy.profiles import geometry_profiles
from autogalaxy import exc


class MassProfile(lensing.LensingObject):
    @property
    def mass_profiles(self):
        return [self]

    @property
    def has_mass_profile(self):
        return True

    @property
    def is_point_mass(self):
        return False

    @property
    def ellipticity_rescale(self):
        return NotImplementedError()

    @property
    def is_mass_sheet(self):
        return False

    def with_new_normalization(self, normalization):
        raise NotImplementedError()


# noinspection PyAbstractClass
class EllipticalMassProfile(geometry_profiles.EllipticalProfile, MassProfile):
    def __init__(
        self,
        centre: typing.Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0),
    ):
        """
        Abstract class for elliptical mass profiles.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps : (float, float)
            The first and second ellipticity components of the elliptical coordinate system, where
            fac = (1 - axis_ratio) / (1 + axis_ratio), ellip_y = fac * sin(2*phi) and ellip_x = fac * cos(2*phi).
        """
        super(EllipticalMassProfile, self).__init__(
            centre=centre, elliptical_comps=elliptical_comps
        )

    @property
    def mass_profile_centres(self):
        if not self.is_mass_sheet:
            return grids.Grid2DIrregularGrouped([self.centre])
        else:
            return []

    def mass_angular_within_circle(self, radius: float):
        """ Integrate the mass profiles's convergence profile to compute the total mass within a circle of \
        specified radius. This is centred on the mass profile.

        The following unit_label for mass can be specified and output:

        - Dimensionless angular unit_label (default) - 'angular'.
        - Solar masses - 'angular' (multiplies the angular mass by the critical surface mass density).

        Parameters
        ----------
        radius : dim.Length
            The radius of the circle to compute the dimensionless mass within.
        unit_mass : str
            The unit_label the mass is returned in {angular, angular}.
        critical_surface_density : float or None
            The critical surface mass density of the strong lens configuration, which converts mass from angulalr \
            unit_label to phsical unit_label (e.g. solar masses).
        """

        return quad(self.mass_integral, a=0.0, b=radius)[0]

    def density_between_circular_annuli(
        self, inner_annuli_radius: float, outer_annuli_radius: float
    ):
        """Calculate the mass between two circular annuli and compute the density by dividing by the annuli surface
        area.

        The value returned by the mass integral is dimensionless, therefore the density between annuli is returned in \
        unit_label of inverse radius squared. A conversion factor can be specified to convert this to a physical value \
        (e.g. the critical surface mass density).

        Parameters
        -----------
        inner_annuli_radius : float
            The radius of the inner annulus outside of which the density are estimated.
        outer_annuli_radius : float
            The radius of the outer annulus inside of which the density is estimated.
        """
        annuli_area = (np.pi * outer_annuli_radius ** 2.0) - (
            np.pi * inner_annuli_radius ** 2.0
        )

        outer_mass = self.mass_angular_within_circle(radius=outer_annuli_radius)

        inner_mass = self.mass_angular_within_circle(radius=inner_annuli_radius)

        return (outer_mass - inner_mass) / annuli_area

    @property
    def average_convergence_of_1_radius(self):
        """The radius a critical curve forms for this mass profile, e.g. where the mean convergence is equal to 1.0.

         In case of ellipitical mass profiles, the 'average' critical curve is used, whereby the convergence is \
         rescaled into a circle using the axis ratio.

         This radius corresponds to the Einstein radius of the mass profile, and is a property of a number of \
         mass profiles below.
         """

        def func(radius):

            return (
                self.mass_angular_within_circle(radius=radius) - np.pi * radius ** 2.0
            )

        return self.ellipticity_rescale * root_scalar(func, bracket=[1e-4, 1e4]).root

    def mass_angular_from_normalization_and_radius(self, normalization, radius):

        mass_profile = self.with_new_normalization(normalization=normalization)

        return mass_profile.mass_angular_within_circle(radius=radius)

    def normalization_from_mass_angular_and_radius(
        self,
        mass_angular,
        radius,
        normalization_min=1e-15,
        normalization_max=1e15,
        bins=200,
    ):

        normalizations = np.logspace(
            np.log10(normalization_min), np.log10(normalization_max), bins
        )

        mass_angulars = [
            self.mass_angular_from_normalization_and_radius(
                normalization=normalization, radius=radius
            )
            for normalization in normalizations
        ]

        normalizations = [
            normalization
            for normalization, mass in zip(normalizations, mass_angulars)
            if mass is not None
        ]
        mass_angulars = list(filter(None, mass_angulars))

        if (
            (len(mass_angulars) < 2)
            or (mass_angulars[0] > mass_angular)
            or (mass_angulars[-1] < mass_angular)
        ):
            raise exc.ProfileException(
                "The normalization could not be computed from the Einstein Radius via the average of the convergence. "
                ""
                "The input einstein_radius may be too small or large to feasibly be computed by integrating the "
                "convergence. Alternative the normalization range or number of bins may need to be changed to "
                "capture the true einstein_radius value."
            )

        def func(normalization, mass_angular_root, radius):

            mass_angular = self.mass_angular_from_normalization_and_radius(
                normalization=normalization, radius=radius
            )

            return mass_angular - mass_angular_root

        return root_scalar(
            func,
            bracket=[normalizations[0], normalizations[-1]],
            args=(mass_angular, radius),
        ).root

    def einstein_radius_from_normalization(self, normalization):

        mass_profile = self.with_new_normalization(normalization=normalization)

        try:
            return mass_profile.average_convergence_of_1_radius
        except ValueError:
            return None

    def normalization_from_einstein_radius(
        self, einstein_radius, normalization_min=1e-9, normalization_max=1e9, bins=100
    ):

        normalizations = np.logspace(
            np.log10(normalization_min), np.log10(normalization_max), bins
        )

        einstein_radii = [
            self.einstein_radius_from_normalization(normalization=normalization)
            for normalization in normalizations
        ]

        normalizations = [
            normalization
            for normalization, radii in zip(normalizations, einstein_radii)
            if radii is not None
        ]
        einstein_radii = list(filter(None, einstein_radii))

        if (
            (len(einstein_radii) < 2)
            or (einstein_radii[0] > einstein_radius)
            or (einstein_radii[-1] < einstein_radius)
        ):
            raise exc.ProfileException(
                "The normalization could not be computed from the Einstein Radius via the average of the convergence. "
                ""
                "The input einstein_radius may be too small or large to feasibly be computed by integrating the "
                "convergence. Alternative the normalization range or number of bins may need to be changed to "
                "capture the true einstein_radius value."
            )

        def func(normalization, einstein_radius_root):

            einstein_radius = self.einstein_radius_from_normalization(
                normalization=normalization
            )

            return einstein_radius - einstein_radius_root

        return root_scalar(
            func,
            bracket=[normalizations[0], normalizations[-1]],
            args=(einstein_radius,),
        ).root


class MassProfileMGE:
    def __init__(self):

        self.count = 0
        self.sigma_calc = 0
        self.z = 0
        self.zq = 0
        self.expv = 0

    @staticmethod
    #  @decorator_util.jit()
    def zeta_from_grid(grid, amps, sigmas, axis_ratio):

        """
        The key part to compute the deflection angle of each Gaussian.
        Because of my optimization, there are blocks looking weird and indirect. What I'm doing here
        is trying to avoid big matrix operation to save time.
        I think there are still spaces we can optimize.

        It seems when using w_f_approx, it gives some errors if y < 0. So when computing for places
        where y < 0, we first compute the value at - y, and then change its sign.
        """

        output_grid_final = np.zeros(grid.shape[0], dtype="complex128")

        q2 = axis_ratio ** 2.0

        scale_factor = axis_ratio / (sigmas[0] * np.sqrt(2.0 * (1.0 - q2)))

        xs = (grid[:, 1] * scale_factor).copy()
        ys = (grid[:, 0] * scale_factor).copy()

        ys_minus = ys < 0.0
        ys[ys_minus] *= -1
        z = xs + 1j * ys
        zq = axis_ratio * xs + 1j * ys / axis_ratio

        expv = -(xs ** 2.0) * (1.0 - q2) - ys ** 2.0 * (1.0 / q2 - 1.0)

        for i in range(len(sigmas)):

            if i > 0:
                z /= sigmas[i] / sigmas[i - 1]
                zq /= sigmas[i] / sigmas[i - 1]
                expv /= (sigmas[i] / sigmas[i - 1]) ** 2.0

            output_grid = -1j * (w_f_approx(z) - np.exp(expv) * w_f_approx(zq))

            output_grid[ys_minus] = np.conj(output_grid[ys_minus])

            output_grid_final += (amps[i] * sigmas[i]) * output_grid

        return output_grid_final

    @staticmethod
    def kesi(p):
        """
        see Eq.(6) of 1906.08263
        """
        n_list = np.arange(0, 2 * p + 1, 1)
        return (2.0 * p * np.log(10) / 3.0 + 2.0 * np.pi * n_list * 1j) ** (0.5)

    @staticmethod
    def eta(p):
        """
        see Eq.(6) of 1906.00263
        """
        eta_list = np.zeros(int(2 * p + 1))
        kesi_list = np.zeros(int(2 * p + 1))
        kesi_list[0] = 0.5
        kesi_list[1 : p + 1] = 1.0
        kesi_list[int(2 * p)] = 1.0 / 2.0 ** p

        for i in np.arange(1, p, 1):
            kesi_list[2 * p - i] = kesi_list[2 * p - i + 1] + 2 ** (-p) * comb(p, i)

        for i in np.arange(0, 2 * p + 1, 1):
            eta_list[i] = (
                (-1) ** i * 2.0 * np.sqrt(2.0 * np.pi) * 10 ** (p / 3.0) * kesi_list[i]
            )

        return eta_list

    def decompose_convergence_into_gaussians(self):
        raise NotImplementedError()

    def _decompose_convergence_into_gaussians(
        self, func, radii_min, radii_max, func_terms=28, func_gaussians=20
    ):
        """

        Parameters
        ----------
        func : func
            The function representing the profile that is decomposed into Gaussians.
        normalization : float
            A normalization factor tyh
        func_terms : int
            The number of terms used to approximate the input func.
        func_gaussians : int
            The number of Gaussians used to represent the input func.

        Returns
        -------
        """

        kesis = self.kesi(func_terms)  # kesi in Eq.(6) of 1906.08263
        etas = self.eta(func_terms)  # eta in Eqr.(6) of 1906.08263

        def f(sigma):
            """Eq.(5) of 1906.08263"""
            return np.sum(etas * np.real(target_function(sigma * kesis)))

        # sigma is sampled from logspace between these radii.

        log_sigmas = np.linspace(np.log(radii_min), np.log(radii_max), func_gaussians)
        d_log_sigma = log_sigmas[1] - log_sigmas[0]
        sigmas = np.exp(log_sigmas)

        amps = np.zeros(func_gaussians)

        for i in range(func_gaussians):
            f_sigma = np.sum(etas * np.real(func(sigmas[i] * kesis)))
            if (i == -1) or (i == (func_gaussians - 1)):
                amps[i] = 0.5 * f_sigma * d_log_sigma / np.sqrt(2.0 * np.pi)
            else:
                amps[i] = f_sigma * d_log_sigma / np.sqrt(2.0 * np.pi)

        return amps, sigmas

    def convergence_from_grid_via_gaussians(self, grid_radii):
        raise NotImplementedError()

    def _convergence_from_grid_via_gaussians(self, grid_radii):
        """Calculate the projected convergence at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : aa.Grid2D
            The grid of (y,x) arc-second coordinates the convergence is computed on.

        """

        self.count = 0
        self.sigma_calc = 0
        self.z = 0
        self.zq = 0
        self.expv = 0

        amps, sigmas = self.decompose_convergence_into_gaussians()

        if self.axis_ratio > 0.9999:
            self.axis_ratio = 0.9999

        convergence = 0.0

        for i in range(len(sigmas)):
            convergence += self.convergence_func_gaussian(
                grid_radii=grid_radii, sigma=sigmas[i], intensity=amps[i]
            )
        return convergence

    def convergence_func_gaussian(self, grid_radii, sigma, intensity):
        return np.multiply(
            intensity, np.exp(-0.5 * np.square(np.divide(grid_radii, sigma)))
        )

    def _deflections_from_grid_via_gaussians(self, grid, sigmas_factor=1.0):

        axis_ratio = self.axis_ratio

        if self.axis_ratio > 0.9999:
            axis_ratio = 0.9999

        amps, sigmas = self.decompose_convergence_into_gaussians()
        sigmas *= sigmas_factor

        angle = self.zeta_from_grid(
            grid=grid, amps=amps, sigmas=sigmas, axis_ratio=axis_ratio
        )

        angle *= np.sqrt((2.0 * np.pi) / (1.0 - axis_ratio ** 2.0))

        return self.rotate_grid_from_profile(np.vstack((-angle.imag, angle.real)).T)


def w_f_approx(z):
    """
    Compute the Faddeeva function :math:`w_{\mathrm F}(z)` using the
    approximation given in Zaghloul (2017).
    :param z: complex number
    :type z: ``complex`` or ``numpy.array(dtype=complex)``
    :return: :math:`w_\mathrm{F}(z)`
    :rtype: ``complex``

    # This function is copied from
    # "https://github.com/sibirrer/lenstronomy/tree/master/lenstronomy/LensModel/Profiles"
    # written by Anowar J. Shajib (see 1906.08263)
    """

    reg_minus_imag = z.imag < 0.0
    z[reg_minus_imag] = np.conj(z[reg_minus_imag])

    sqrt_pi = 1 / np.sqrt(np.pi)
    i_sqrt_pi = 1j * sqrt_pi

    wz = np.empty_like(z)

    z_imag2 = z.imag ** 2
    abs_z2 = z.real ** 2 + z_imag2

    reg1 = abs_z2 >= 38000.0
    if np.any(reg1):
        wz[reg1] = i_sqrt_pi / z[reg1]

    reg2 = (256.0 <= abs_z2) & (abs_z2 < 38000.0)
    if np.any(reg2):
        t = z[reg2]
        wz[reg2] = i_sqrt_pi * t / (t * t - 0.5)

    reg3 = (62.0 <= abs_z2) & (abs_z2 < 256.0)
    if np.any(reg3):
        t = z[reg3]
        wz[reg3] = (i_sqrt_pi / t) * (1 + 0.5 / (t * t - 1.5))

    reg4 = (30.0 <= abs_z2) & (abs_z2 < 62.0) & (z_imag2 >= 1e-13)
    if np.any(reg4):
        t = z[reg4]
        tt = t * t
        wz[reg4] = (i_sqrt_pi * t) * (tt - 2.5) / (tt * (tt - 3.0) + 0.75)

    reg5 = (62.0 > abs_z2) & np.logical_not(reg4) & (abs_z2 > 2.5) & (z_imag2 < 0.072)
    if np.any(reg5):
        t = z[reg5]
        u = -t * t
        f1 = sqrt_pi
        f2 = 1
        s1 = [1.320522, 35.7668, 219.031, 1540.787, 3321.99, 36183.31]
        s2 = [1.841439, 61.57037, 364.2191, 2186.181, 9022.228, 24322.84, 32066.6]

        for s in s1:
            f1 = s - f1 * u
        for s in s2:
            f2 = s - f2 * u

        wz[reg5] = np.exp(u) + 1j * t * f1 / f2

    reg6 = (30.0 > abs_z2) & np.logical_not(reg5)
    if np.any(reg6):
        t3 = -1j * z[reg6]

        f1 = sqrt_pi
        f2 = 1
        s1 = [5.9126262, 30.180142, 93.15558, 181.92853, 214.38239, 122.60793]
        s2 = [
            10.479857,
            53.992907,
            170.35400,
            348.70392,
            457.33448,
            352.73063,
            122.60793,
        ]

        for s in s1:
            f1 = f1 * t3 + s
        for s in s2:
            f2 = f2 * t3 + s

        wz[reg6] = f1 / f2

    # wz[reg_minus_imag] = np.conj(wz[reg_minus_imag])

    return wz


def psi_from(grid, axis_ratio, core_radius):
    """
    Returns the $\Psi$ term in expressions for the calculation of the deflection of an elliptical isothermal mass
    distribution. This is used in the `Isothermal` and `Chameleon` `MassProfile`'s.

    The expression for Psi is:

    $\Psi = \sqrt(q^2(s^2 + x^2) + y^2)$

    Parameters
    ----------
    grid : grid_like
        The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
    axis_ratio : float
        Ratio of profiles ellipse's minor and major axes (b/a)
    core_radius : float
        The radius of the inner core

    Returns
    -------
    float
        The value of the Psi term.

    """
    return np.sqrt(
        np.add(
            np.multiply(
                axis_ratio ** 2.0, np.add(np.square(grid[:, 1]), core_radius ** 2.0)
            ),
            np.square(grid[:, 0]),
        )
    )
