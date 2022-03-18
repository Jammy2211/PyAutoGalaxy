from abc import ABC, abstractmethod
import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.special import wofz, comb
from scipy.linalg import lstsq
from typing import Callable, List, Optional, Tuple

import autoarray as aa

from autogalaxy.profiles.geometry_profiles import EllProfile
from autogalaxy.operate.deflections import OperateDeflections

from autogalaxy import exc


# noinspection PyAbstractClass
class MassProfile(EllProfile, OperateDeflections):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
    ):
        """
        Abstract class for elliptical mass profiles.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps
            The first and second ellipticity components of the elliptical coordinate system, (see the module
            `autogalaxy -> convert.py` for the convention).
        """
        super().__init__(centre=centre, elliptical_comps=elliptical_comps)

    def deflections_yx_2d_from(self, grid):
        raise NotImplementedError

    def deflections_2d_via_potential_2d_from(self, grid):

        potential = self.potential_2d_from(grid=grid)

        deflections_y_2d = np.gradient(potential.native, grid.native[:, 0, 0], axis=0)
        deflections_x_2d = np.gradient(potential.native, grid.native[0, :, 1], axis=1)

        return aa.Grid2D.manual_mask(
            grid=np.stack((deflections_y_2d, deflections_x_2d), axis=-1), mask=grid.mask
        )

    def convergence_2d_from(self, grid):
        raise NotImplementedError

    def convergence_func(self, grid_radius: float) -> float:
        raise NotImplementedError

    @aa.grid_dec.grid_1d_to_structure
    def convergence_1d_from(
        self, grid: aa.type.Grid1D2DLike, radial_grid_shape_slim: Optional[int] = None
    ) -> aa.type.Grid1D2DLike:
        return self.convergence_2d_from(grid=grid)

    def potential_2d_from(self, grid):
        raise NotImplementedError

    @aa.grid_dec.grid_1d_to_structure
    def potential_1d_from(
        self, grid: aa.type.Grid1D2DLike, radial_grid_shape_slim: Optional[int] = None
    ) -> aa.type.Grid1D2DLike:
        return self.potential_2d_from(grid=grid)

    def potential_func(self, u, y, x):
        raise NotImplementedError

    @property
    def has_mass_profile(self):
        return True

    def mass_integral(self, x):
        return 2 * np.pi * x * self.convergence_func(grid_radius=x)

    @property
    def ellipticity_rescale(self):
        return NotImplementedError()

    def mass_angular_within_circle_from(self, radius: float):
        """
        Integrate the mass profiles's convergence profile to compute the total mass within a circle of
        specified radius. This is centred on the mass profile.

        Parameters
        ----------
        radius : dim.Length
            The radius of the circle to compute the dimensionless mass within.
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
        inner_annuli_radius
            The radius of the inner annulus outside of which the density are estimated.
        outer_annuli_radius
            The radius of the outer annulus inside of which the density is estimated.
        """
        annuli_area = (np.pi * outer_annuli_radius ** 2.0) - (
            np.pi * inner_annuli_radius ** 2.0
        )

        outer_mass = self.mass_angular_within_circle_from(radius=outer_annuli_radius)

        inner_mass = self.mass_angular_within_circle_from(radius=inner_annuli_radius)

        return (outer_mass - inner_mass) / annuli_area

    @property
    def average_convergence_of_1_radius(self):
        """
        The radius a critical curve forms for this mass profile, e.g. where the mean convergence is equal to 1.0.

        In case of ellipitical mass profiles, the 'average' critical curve is used, whereby the convergence is \
        rescaled into a circle using the axis ratio.

        This radius corresponds to the Einstein radius of the mass profile, and is a property of a number of \
        mass profiles below.
        """

        def func(radius):

            return (
                self.mass_angular_within_circle_from(radius=radius)
                - np.pi * radius ** 2.0
            )

        return self.ellipticity_rescale * root_scalar(func, bracket=[1e-4, 1e4]).root

    def mass_angular_via_normalization_from(self, normalization, radius):

        mass_profile = self.with_new_normalization(normalization=normalization)

        return mass_profile.mass_angular_within_circle_from(radius=radius)

    def normalization_via_mass_angular_from(
        self,
        mass_angular,
        radius,
        normalization_min=1e-15,
        normalization_max=1e15,
        bins=200,
    ):

        normalization_list = np.logspace(
            np.log10(normalization_min), np.log10(normalization_max), bins
        )

        mass_angulars = [
            self.mass_angular_via_normalization_from(
                normalization=normalization, radius=radius
            )
            for normalization in normalization_list
        ]

        normalization_list = [
            normalization
            for normalization, mass in zip(normalization_list, mass_angulars)
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

            mass_angular = self.mass_angular_via_normalization_from(
                normalization=normalization, radius=radius
            )

            return mass_angular - mass_angular_root

        return root_scalar(
            func,
            bracket=[normalization_list[0], normalization_list[-1]],
            args=(mass_angular, radius),
        ).root

    def with_new_normalization(self, normalization):
        raise NotImplementedError()

    def einstein_radius_via_normalization_from(self, normalization):

        mass_profile = self.with_new_normalization(normalization=normalization)

        try:
            return mass_profile.average_convergence_of_1_radius
        except ValueError:
            return None

    def normalization_via_einstein_radius_from(
        self, einstein_radius, normalization_min=1e-9, normalization_max=1e9, bins=100
    ):

        normalization_list = np.logspace(
            np.log10(normalization_min), np.log10(normalization_max), bins
        )

        einstein_radii = [
            self.einstein_radius_via_normalization_from(normalization=normalization)
            for normalization in normalization_list
        ]

        normalization_list = [
            normalization
            for normalization, radii in zip(normalization_list, einstein_radii)
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

            einstein_radius = self.einstein_radius_via_normalization_from(
                normalization=normalization
            )

            return einstein_radius - einstein_radius_root

        return root_scalar(
            func,
            bracket=[normalization_list[0], normalization_list[-1]],
            args=(einstein_radius,),
        ).root

    def extract_attribute(self, cls, attr_name):
        """
        Returns an attribute of a class and its children profiles in the the galaxy as a `ValueIrregular`
        or `Grid2DIrregular` object.

        For example, if a galaxy has two light profiles and we want the `LightProfile` axis-ratios, the following:

        `galaxy.extract_attribute(cls=LightProfile, name="axis_ratio"`

        would return:

        ValuesIrregular(values=[axis_ratio_0, axis_ratio_1])

        If a galaxy has three mass profiles and we want the `MassProfile` centres, the following:

        `galaxy.extract_attribute(cls=MassProfile, name="centres"`

         would return:

        GridIrregular2D(grid=[(centre_y_0, centre_x_0), (centre_y_1, centre_x_1), (centre_y_2, centre_x_2)])

        This is used for visualization, for example plotting the centres of all light profiles colored by their profile.
        """

        if isinstance(self, cls):
            if hasattr(self, attr_name):

                attribute = getattr(self, attr_name)

                if isinstance(attribute, float):
                    return aa.ValuesIrregular(values=[attribute])
                if isinstance(attribute, tuple):
                    return aa.Grid2DIrregular(grid=[attribute])


class MassProfileMGE:
    """
    This class speeds up deflection angle calculations of certain mass profiles by decompositing them into many
    Gaussians.

    This follows the method of Shajib 2019 - https://academic.oup.com/mnras/article/488/1/1387/5526256
    """

    def __init__(self):

        self.count = 0
        self.sigma_calc = 0
        self.z = 0
        self.zq = 0
        self.expv = 0

    @staticmethod
    #  @aa.util.numba.jit()
    def zeta_from(grid, amps, sigmas, axis_ratio):

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

    def decompose_convergence_via_mge(self):
        raise NotImplementedError()

    def _decompose_convergence_via_mge(
        self, func, radii_min, radii_max, func_terms=28, func_gaussians=20
    ):
        """

        Parameters
        ----------
        func : func
            The function representing the profile that is decomposed into Gaussians.
        normalization
            A normalization factor tyh
        func_terms
            The number of terms used to approximate the input func.
        func_gaussians
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
        sigma_list = np.exp(log_sigmas)

        amplitude_list = np.zeros(func_gaussians)

        for i in range(func_gaussians):
            f_sigma = np.sum(etas * np.real(func(sigma_list[i] * kesis)))
            if (i == -1) or (i == (func_gaussians - 1)):
                amplitude_list[i] = 0.5 * f_sigma * d_log_sigma / np.sqrt(2.0 * np.pi)
            else:
                amplitude_list[i] = f_sigma * d_log_sigma / np.sqrt(2.0 * np.pi)

        return amplitude_list, sigma_list

    def convergence_2d_via_mge_from(self, grid_radii):
        raise NotImplementedError()

    def _convergence_2d_via_mge_from(self, grid_radii):
        """Calculate the projected convergence at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the convergence is computed on.

        """

        self.count = 0
        self.sigma_calc = 0
        self.z = 0
        self.zq = 0
        self.expv = 0

        amps, sigmas = self.decompose_convergence_via_mge()

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

    def _deflections_2d_via_mge_from(self, grid, sigmas_factor=1.0):

        axis_ratio = self.axis_ratio

        if axis_ratio > 0.9999:
            axis_ratio = 0.9999

        amps, sigmas = self.decompose_convergence_via_mge()
        sigmas *= sigmas_factor

        angle = self.zeta_from(
            grid=grid, amps=amps, sigmas=sigmas, axis_ratio=axis_ratio
        )

        angle *= np.sqrt((2.0 * np.pi) / (1.0 - axis_ratio ** 2.0))

        return self.rotate_grid_from_reference_frame(
            np.vstack((-angle.imag, angle.real)).T
        )


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
    grid
        The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
    axis_ratio
            Ratio of profiles ellipse's minor and major axes (b/a)
    core_radius
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


class MassProfileCSE(ABC):
    @staticmethod
    def convergence_cse_1d_from(
        grid_radii: np.ndarray, core_radius: float
    ) -> np.ndarray:
        """
        One dimensional function which is solved to decompose a convergence profile in cored steep ellipsoids, given by
        equation (14) of Oguri 2021 (https://arxiv.org/abs/2106.11464).

        Parameters
        ----------
        grid_radii
            The 1D radial coordinates the decomposition is performed for.
        core_radius
            The core radius of the cored steep ellisoid used for this decomposition.

        """
        return 1.0 / (2.0 * (core_radius ** 2.0 + grid_radii ** 2.0) ** (1.5))

    @staticmethod
    def deflections_via_cse_from(
        term1: float,
        term2: float,
        term3: float,
        term4: float,
        axis_ratio_squared: float,
        core_radius: float,
    ) -> np.ndarray:
        """
        Returns the deflection angles of a 1d cored steep ellisoid (CSE) profile, given by equation (19) and (20) of
        Oguri 2021 (https://arxiv.org/abs/2106.11464).

        To accelerate the deflection angle computation terms are computed separated, defined as term1, 2, 3, 4.

        Parameters
        ----------
        """
        # phi = np.sqrt(q**2.0 * (s**2.0 + gridx2) + gridy2)
        phi = np.sqrt(axis_ratio_squared * core_radius ** 2.0 + term1)
        # Psi = (phi + s)**2.0 + (1 - q * q) * gridx2
        Psi = (phi + core_radius) ** 2.0 + term2
        bottom = core_radius * phi * Psi
        defl_x = (term3 * (phi + axis_ratio_squared * core_radius)) / bottom
        defl_y = (term4 * (phi + core_radius)) / bottom
        return np.vstack((defl_y, defl_x))

    @abstractmethod
    def decompose_convergence_via_cse(self):
        pass

    def _decompose_convergence_via_cse_from(
        self,
        func: Callable,
        radii_min: float,
        radii_max: float,
        total_cses: int = 25,
        sample_points: int = 100,
    ) -> Tuple[List, List]:
        """
        Decompose the convergence of a mass profile into cored steep elliptical (cse) profiles.

        This uses an input function `func` which is specific to the inherited mass profile, and defines the function
        which is solved for in order to decompose its convergence into cses.

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
        error_sigma = 0.1  # error spread. Could be any value.

        r_samples = np.logspace(np.log10(radii_min), np.log10(radii_max), sample_points)
        y_samples = np.ones_like(r_samples) / error_sigma
        y_samples_func = func(r_samples)

        core_radius_list = np.logspace(
            np.log10(radii_min), np.log10(radii_max), total_cses
        )

        # Different from Masamune's (2106.11464) method, I set S to a series fixed values. So that
        # the decomposition can be solved linearly.

        coefficient_matrix = np.zeros((sample_points, total_cses))

        for j in range(total_cses):
            coefficient_matrix[:, j] = self.convergence_cse_1d_from(
                r_samples, core_radius_list[j]
            )

        for k in range(sample_points):
            coefficient_matrix[k] /= y_samples_func[k] * error_sigma

        results = lstsq(coefficient_matrix, y_samples.T)

        amplitude_list = results[0]

        return amplitude_list, core_radius_list

    def convergence_2d_via_cse_from(self, grid_radii: np.ndarray) -> np.ndarray:
        pass

    def _convergence_2d_via_cse_from(self, grid_radii: np.ndarray) -> np.ndarray:
        """
        Calculate the projected 2D convergence from a grid of radial coordinates, by computing and summing the
        convergence of each individual cse used to decompose the mass profile.

        The cored steep elliptical (cse) decomposition of a given mass profile (e.g. `convergence_cse_1d_from`) is
        defined for every mass profile and defines how it is efficiently decomposed its cses.

        Parameters
        ----------
        grid_radii
            The grid of 1D radial arc-second coordinates the convergence is computed on.
        """

        amplitude_list, core_radius_list = self.decompose_convergence_via_cse()

        return sum(
            amplitude
            * self.convergence_cse_1d_from(
                grid_radii=grid_radii, core_radius=core_radius
            )
            for amplitude, core_radius in zip(amplitude_list, core_radius_list)
        )

    def _deflections_2d_via_cse_from(self, grid: np.ndarray) -> np.ndarray:
        """
        Calculate the projected 2D deflection angles from a grid of radial coordinates, by computing and summing the
        deflections of each individual cse used to decompose the mass profile.

        The cored steep elliptical (cse) decomposition of a given mass profile (e.g. `convergence_cse_1d_from`) is
        defined for every mass profile and defines how it is efficiently decomposed its cses.

        Parameters
        ----------
        grid_radii
            The grid of 1D radial arc-second coordinates the convergence is computed on.
        """

        amplitude_list, core_radius_list = self.decompose_convergence_via_cse()
        q = self.axis_ratio
        q2 = q ** 2.0
        grid_y = grid[:, 0]
        grid_x = grid[:, 1]
        gridx2 = grid_x ** 2.0
        gridy2 = grid_y ** 2.0
        term1 = q2 * gridx2 + gridy2
        term2 = (1.0 - q2) * gridx2
        term3 = q * grid_x
        term4 = q * grid_y

        # To accelarate deflection angle computation, I define term1, term2, term3, term4 to avoid
        # repeated matrix operation. There might be still space for optimization.

        deflections_2d = sum(
            amplitude
            * self.deflections_via_cse_from(
                axis_ratio_squared=q2,
                core_radius=core_radius,
                term1=term1,
                term2=term2,
                term3=term3,
                term4=term4,
            )
            for amplitude, core_radius in zip(amplitude_list, core_radius_list)
        )

        return self.rotate_grid_from_reference_frame(deflections_2d.T)
