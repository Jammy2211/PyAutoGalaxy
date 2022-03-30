import copy
import numpy as np
from scipy.integrate import quad
from scipy import special
from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.mass_profiles import MassProfile

from autogalaxy.profiles.mass_profiles.mass_profiles import psi_from


class PointMass(MassProfile):
    def __init__(
        self, centre: Tuple[float, float] = (0.0, 0.0), einstein_radius: float = 1.0
    ):
        """
        Represents a point-mass.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        einstein_radius
            The arc-second Einstein radius of the point-mass.
        """
        super().__init__(centre=centre, elliptical_comps=(0.0, 0.0))
        self.einstein_radius = einstein_radius

    def convergence_2d_from(self, grid: aa.type.Grid2DLike):

        squared_distances = np.square(grid[:, 0] - self.centre[0]) + np.square(
            grid[:, 1] - self.centre[1]
        )
        central_pixel = np.argmin(squared_distances)

        convergence = np.zeros(shape=grid.shape[0])
        #    convergence[central_pixel] = np.pi * self.einstein_radius ** 2.0
        return convergence

    @aa.grid_dec.grid_2d_to_structure
    def potential_2d_from(self, grid: aa.type.Grid2DLike):
        return np.zeros(shape=grid.shape[0])

    @aa.grid_dec.grid_2d_to_vector_yx
    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):
        grid_radii = self.grid_to_grid_radii(grid=grid)
        return self.grid_to_grid_cartesian(
            grid=grid, radius=self.einstein_radius ** 2 / grid_radii
        )

    @property
    def is_point_mass(self):
        return True

    def with_new_normalization(self, normalization):

        mass_profile = copy.copy(self)
        mass_profile.einstein_radius = normalization
        return mass_profile


class EllPowerLawBroken(MassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
        inner_slope: float = 1.5,
        outer_slope: float = 2.5,
        break_radius: float = 0.01,
    ):
        """
        Ell, homoeoidal mass model with an inner_slope
        and outer_slope, continuous in density across break_radius.
        Position angle is defined to be zero on x-axis and
        +ve angle rotates the lens anticlockwise

        The grid variable is a tuple of (theta_1, theta_2), where
        each theta_1, theta_2 is itself a 2D array of the x and y
        coordinates respectively.~
        """

        super().__init__(centre=centre, elliptical_comps=elliptical_comps)

        self.einstein_radius = einstein_radius
        self.einstein_radius_elliptical = np.sqrt(self.axis_ratio) * einstein_radius
        self.break_radius = break_radius
        self.inner_slope = inner_slope
        self.outer_slope = outer_slope

        # Parameters defined in the notes
        self.nu = break_radius / self.einstein_radius_elliptical
        self.dt = (2 - self.inner_slope) / (2 - self.outer_slope)

        # Normalisation (eq. 5)
        if self.nu < 1:
            self.kB = (2 - self.inner_slope) / (
                (2 * self.nu ** 2)
                * (1 + self.dt * (self.nu ** (self.outer_slope - 2) - 1))
            )
        else:
            self.kB = (2 - self.inner_slope) / (2 * self.nu ** 2)

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def convergence_2d_from(self, grid: aa.type.Grid2DLike):
        """
        Returns the dimensionless density kappa=Sigma/Sigma_c (eq. 1)
        """

        # Ell radius
        radius = np.hypot(grid[:, 1] * self.axis_ratio, grid[:, 0])

        # Inside break radius
        kappa_inner = self.kB * (self.break_radius / radius) ** self.inner_slope

        # Outside break radius
        kappa_outer = self.kB * (self.break_radius / radius) ** self.outer_slope

        return kappa_inner * (radius <= self.break_radius) + kappa_outer * (
            radius > self.break_radius
        )

    @aa.grid_dec.grid_2d_to_structure
    def potential_2d_from(self, grid: aa.type.Grid2DLike):
        return np.zeros(shape=grid.shape[0])

    @aa.grid_dec.grid_2d_to_vector_yx
    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid, max_terms=20):
        """
        Returns the complex deflection angle from eq. 18 and 19
        """
        # Rotate coordinates
        z = grid[:, 1] + 1j * grid[:, 0]

        # Ell radius
        R = np.hypot(z.real * self.axis_ratio, z.imag)

        # Factors common to eq. 18 and 19
        factors = (
            2
            * self.kB
            * (self.break_radius ** 2)
            / (self.axis_ratio * z * (2 - self.inner_slope))
        )

        # Hypergeometric functions
        # (in order of appearance in eq. 18 and 19)
        # These can also be computed with scipy.special.hyp2f1(), it's
        # much slower can be a useful test
        F1 = self.hyp2f1_series(
            self.inner_slope, self.axis_ratio, R, z, max_terms=max_terms
        )
        F2 = self.hyp2f1_series(
            self.inner_slope, self.axis_ratio, self.break_radius, z, max_terms=max_terms
        )
        F3 = self.hyp2f1_series(
            self.outer_slope, self.axis_ratio, R, z, max_terms=max_terms
        )
        F4 = self.hyp2f1_series(
            self.outer_slope, self.axis_ratio, self.break_radius, z, max_terms=max_terms
        )

        # theta < break radius (eq. 18)
        inner_part = factors * F1 * (self.break_radius / R) ** (self.inner_slope - 2)

        # theta > break radius (eq. 19)
        outer_part = factors * (
            F2
            + self.dt * (((self.break_radius / R) ** (self.outer_slope - 2)) * F3 - F4)
        )

        # Combine and take the conjugate
        deflections = (
            inner_part * (R <= self.break_radius) + outer_part * (R > self.break_radius)
        ).conjugate()

        return self.rotate_grid_from_reference_frame(
            grid=np.multiply(
                1.0, np.vstack((np.imag(deflections), np.real(deflections))).T
            )
        )

    @staticmethod
    def hyp2f1_series(t, q, r, z, max_terms=20):
        """
        Computes eq. 26 for a radius r, slope t,
        axis ratio q, and coordinates z.
        """

        # u from eq. 25
        q_ = (1 - q ** 2) / (q ** 2)
        u = 0.5 * (1 - np.sqrt(1 - q_ * (r / z) ** 2))

        # First coefficient
        a_n = 1.0

        # Storage for sum
        F = np.zeros_like(z, dtype="complex64")

        for n in range(max_terms):
            F += a_n * (u ** n)
            a_n *= ((2 * n) + 4 - (2 * t)) / ((2 * n) + 4 - t)

        return F

    def with_new_normalization(self, normalization):

        mass_profile = copy.copy(self)
        mass_profile.einstein_radius_elliptical = normalization
        return mass_profile


class SphPowerLawBroken(EllPowerLawBroken):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
        inner_slope: float = 1.5,
        outer_slope: float = 2.5,
        break_radius: float = 0.01,
    ):
        """
        Ell, homoeoidal mass model with an inner_slope
        and outer_slope, continuous in density across break_radius.
        Position angle is defined to be zero on x-axis and
        +ve angle rotates the lens anticlockwise

        The grid variable is a tuple of (theta_1, theta_2), where
        each theta_1, theta_2 is itself a 2D array of the x and y
        coordinates respectively.~
        """

        super().__init__(
            centre=centre,
            elliptical_comps=(0.0, 0.0),
            einstein_radius=einstein_radius,
            inner_slope=inner_slope,
            outer_slope=outer_slope,
            break_radius=break_radius,
        )


class EllPowerLawCored(MassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
        slope: float = 2.0,
        core_radius: float = 0.01,
    ):
        """
        Represents a cored elliptical power-law density distribution

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps
            The first and second ellipticity components of the elliptical coordinate system, (see the module
            `autogalaxy -> convert.py` for the convention).
        einstein_radius
            The arc-second Einstein radius.
        slope
            The density slope of the power-law (lower value -> shallower profile, higher value -> steeper profile).
        core_radius
            The arc-second radius of the inner core.
        """
        super().__init__(centre=centre, elliptical_comps=elliptical_comps)

        self.einstein_radius = einstein_radius
        self.slope = slope
        self.core_radius = core_radius

    @property
    def einstein_radius_rescaled(self):
        """Rescale the einstein radius by slope and axis_ratio, to reduce its degeneracy with other mass-profiles
        parameters"""
        return ((3 - self.slope) / (1 + self.axis_ratio)) * self.einstein_radius ** (
            self.slope - 1
        )

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def convergence_2d_from(self, grid: aa.type.Grid2DLike):
        """
        Calculate the projected convergence on a grid of (y,x) arc-second coordinates.

        The `grid_2d_to_structure` decorator reshapes the ndarrays the convergence is outputted on. See \
        *aa.grid_2d_to_structure* for a description of the output.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the convergence is computed on.

        """

        covnergence_grid = np.zeros(grid.shape[0])

        grid_eta = self.grid_to_elliptical_radii(grid)

        for i in range(grid.shape[0]):
            covnergence_grid[i] = self.convergence_func(grid_eta[i])

        return covnergence_grid

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def potential_2d_from(self, grid: aa.type.Grid2DLike):
        """
        Calculate the potential on a grid of (y,x) arc-second coordinates.

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
                    self.slope,
                    self.core_radius,
                ),
            )[0]

        return self.einstein_radius_rescaled * self.axis_ratio * potential_grid

    @aa.grid_dec.grid_2d_to_vector_yx
    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):
        """
        Calculate the deflection angles on a grid of (y,x) arc-second coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """

        def calculate_deflection_component(npow, index):
            einstein_radius_rescaled = self.einstein_radius_rescaled

            deflection_grid = self.axis_ratio * grid[:, index]

            for i in range(grid.shape[0]):

                deflection_grid[i] *= (
                    einstein_radius_rescaled
                    * quad(
                        self.deflection_func,
                        a=0.0,
                        b=1.0,
                        args=(
                            grid[i, 0],
                            grid[i, 1],
                            npow,
                            self.axis_ratio,
                            self.slope,
                            self.core_radius,
                        ),
                    )[0]
                )

            return deflection_grid

        deflection_y = calculate_deflection_component(1.0, 0)
        deflection_x = calculate_deflection_component(0.0, 1)

        return self.rotate_grid_from_reference_frame(
            grid=np.multiply(1.0, np.vstack((deflection_y, deflection_x)).T)
        )

    def convergence_func(self, grid_radius: float) -> float:
        return self.einstein_radius_rescaled * (
            self.core_radius ** 2 + grid_radius ** 2
        ) ** (-(self.slope - 1) / 2.0)

    @staticmethod
    def potential_func(u, y, x, axis_ratio, slope, core_radius):
        eta = np.sqrt((u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u)))))
        return (
            (eta / u)
            * ((3.0 - slope) * eta) ** -1.0
            * (
                (core_radius ** 2.0 + eta ** 2.0) ** ((3.0 - slope) / 2.0)
                - core_radius ** (3 - slope)
            )
            / ((1 - (1 - axis_ratio ** 2) * u) ** 0.5)
        )

    @staticmethod
    def deflection_func(u, y, x, npow, axis_ratio, slope, core_radius):
        eta_u = np.sqrt((u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u)))))
        return (core_radius ** 2 + eta_u ** 2) ** (-(slope - 1) / 2.0) / (
            (1 - (1 - axis_ratio ** 2) * u) ** (npow + 0.5)
        )

    @property
    def ellipticity_rescale(self):
        return (1.0 + self.axis_ratio) / 2.0

    @property
    def unit_mass(self):
        return "angular"

    def with_new_normalization(self, normalization):

        mass_profile = copy.copy(self)
        mass_profile.einstein_radius = normalization
        return mass_profile


class SphPowerLawCored(EllPowerLawCored):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
        slope: float = 2.0,
        core_radius: float = 0.01,
    ):
        """
        Represents a cored spherical power-law density distribution

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        einstein_radius
            The arc-second Einstein radius.
        slope
            The density slope of the power-law (lower value -> shallower profile, higher value -> steeper profile).
        core_radius
            The arc-second radius of the inner core.
        """
        super().__init__(
            centre=centre,
            elliptical_comps=(0.0, 0.0),
            einstein_radius=einstein_radius,
            slope=slope,
            core_radius=core_radius,
        )

    @aa.grid_dec.grid_2d_to_vector_yx
    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):
        """
        Calculate the deflection angles on a grid of (y,x) arc-second coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """
        eta = self.grid_to_grid_radii(grid=grid)
        deflection = np.multiply(
            2.0 * self.einstein_radius_rescaled,
            np.divide(
                np.add(
                    np.power(
                        np.add(self.core_radius ** 2, np.square(eta)),
                        (3.0 - self.slope) / 2.0,
                    ),
                    -self.core_radius ** (3 - self.slope),
                ),
                np.multiply((3.0 - self.slope), eta),
            ),
        )
        return self.grid_to_grid_cartesian(grid=grid, radius=deflection)


class EllPowerLaw(EllPowerLawCored):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
        slope: float = 2.0,
    ):
        """
        Represents an elliptical power-law density distribution.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps
            The first and second ellipticity components of the elliptical coordinate system, (see the module
            `autogalaxy -> convert.py` for the convention).
        einstein_radius
            The arc-second Einstein radius.
        slope
            The density slope of the power-law (lower value -> shallower profile, higher value -> steeper profile).
        """

        super().__init__(
            centre=centre,
            elliptical_comps=elliptical_comps,
            einstein_radius=einstein_radius,
            slope=slope,
            core_radius=0.0,
        )

    @aa.grid_dec.grid_2d_to_vector_yx
    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):
        """
        Calculate the deflection angles on a grid of (y,x) arc-second coordinates.

        For coordinates (0.0, 0.0) the analytic calculation of the deflection angle gives a NaN. Therefore, \
        coordinates at (0.0, 0.0) are shifted slightly to (1.0e-8, 1.0e-8).

        This code is an adaption of Tessore & Metcalf 2015:
        https://arxiv.org/abs/1507.01819

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """

        slope = self.slope - 1.0
        einstein_radius = (
            2.0 / (self.axis_ratio ** -0.5 + self.axis_ratio ** 0.5)
        ) * self.einstein_radius

        factor = np.divide(1.0 - self.axis_ratio, 1.0 + self.axis_ratio)
        b = np.multiply(einstein_radius, np.sqrt(self.axis_ratio))
        angle = np.arctan2(
            grid[:, 0], np.multiply(self.axis_ratio, grid[:, 1])
        )  # Note, this angle is not the position angle
        R = np.sqrt(
            np.add(np.multiply(self.axis_ratio ** 2, grid[:, 1] ** 2), grid[:, 0] ** 2)
        )
        z = np.add(
            np.multiply(np.cos(angle), 1 + 0j), np.multiply(np.sin(angle), 0 + 1j)
        )

        complex_angle = (
            2.0
            * b
            / (1.0 + self.axis_ratio)
            * (b / R) ** (slope - 1.0)
            * z
            * special.hyp2f1(1.0, 0.5 * slope, 2.0 - 0.5 * slope, -factor * z ** 2)
        )

        deflection_y = complex_angle.imag
        deflection_x = complex_angle.real

        rescale_factor = (self.ellipticity_rescale) ** (slope - 1)

        deflection_y *= rescale_factor
        deflection_x *= rescale_factor

        return self.rotate_grid_from_reference_frame(
            grid=np.vstack((deflection_y, deflection_x)).T
        )

    def convergence_func(self, grid_radius: float) -> float:
        if grid_radius > 0.0:
            return self.einstein_radius_rescaled * grid_radius ** (-(self.slope - 1))
        return np.inf

    @staticmethod
    def potential_func(u, y, x, axis_ratio, slope, core_radius):
        eta_u = np.sqrt((u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u)))))
        return (
            (eta_u / u)
            * ((3.0 - slope) * eta_u) ** -1.0
            * eta_u ** (3.0 - slope)
            / ((1 - (1 - axis_ratio ** 2) * u) ** 0.5)
        )


class SphPowerLaw(EllPowerLaw):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
        slope: float = 2.0,
    ):
        """
        Represents a spherical power-law density distribution.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        einstein_radius
            The arc-second Einstein radius.
        slope
            The density slope of the power-law (lower value -> shallower profile, higher value -> steeper profile).
        """

        super().__init__(
            centre=centre,
            elliptical_comps=(0.0, 0.0),
            einstein_radius=einstein_radius,
            slope=slope,
        )

    @aa.grid_dec.grid_2d_to_vector_yx
    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):

        eta = self.grid_to_grid_radii(grid)
        deflection_r = (
            2.0
            * self.einstein_radius_rescaled
            * np.divide(
                np.power(eta, (3.0 - self.slope)), np.multiply((3.0 - self.slope), eta)
            )
        )

        return self.grid_to_grid_cartesian(grid, deflection_r)


class EllIsothermalCored(EllPowerLawCored):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
        core_radius: float = 0.01,
    ):
        """
        Represents a cored elliptical isothermal density distribution, which is equivalent to the elliptical power-law
        density distribution for the value slope: float = 2.0

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps
            The first and second ellipticity components of the elliptical coordinate system, (see the module
            `autogalaxy -> convert.py` for the convention).
        einstein_radius
            The arc-second Einstein radius.
        core_radius
            The arc-second radius of the inner core.
        """
        super().__init__(
            centre=centre,
            elliptical_comps=elliptical_comps,
            einstein_radius=einstein_radius,
            slope=2.0,
            core_radius=core_radius,
        )


class SphIsothermalCored(SphPowerLawCored):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
        core_radius: float = 0.01,
    ):
        """
        Represents a cored spherical isothermal density distribution, which is equivalent to the elliptical power-law
        density distribution for the value slope: float = 2.0

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        einstein_radius
            The arc-second Einstein radius.
        core_radius
            The arc-second radius of the inner core.
        """
        super().__init__(
            centre=centre,
            einstein_radius=einstein_radius,
            slope=2.0,
            core_radius=core_radius,
        )


class EllIsothermal(EllPowerLaw):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
    ):
        """
        Represents an elliptical isothermal density distribution, which is equivalent to the elliptical power-law
        density distribution for the value slope: float = 2.0

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps
            The first and second ellipticity components of the elliptical coordinate system, (see the module
            `autogalaxy -> convert.py` for the convention).
        einstein_radius
            The arc-second Einstein radius.
        """

        super().__init__(
            centre=centre,
            elliptical_comps=elliptical_comps,
            einstein_radius=einstein_radius,
            slope=2.0,
        )

    @property
    def axis_ratio(self):
        axis_ratio = super().axis_ratio
        return min(axis_ratio, 0.99999)

    @aa.grid_dec.grid_2d_to_vector_yx
    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):
        """
        Calculate the deflection angles on a grid of (y,x) arc-second coordinates.

        For coordinates (0.0, 0.0) the analytic calculation of the deflection angle gives a NaN. Therefore, \
        coordinates at (0.0, 0.0) are shifted slightly to (1.0e-8, 1.0e-8).

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """

        factor = (
            2.0
            * self.einstein_radius_rescaled
            * self.axis_ratio
            / np.sqrt(1 - self.axis_ratio ** 2)
        )

        psi = psi_from(grid=grid, axis_ratio=self.axis_ratio, core_radius=0.0)

        deflection_y = np.arctanh(
            np.divide(np.multiply(np.sqrt(1 - self.axis_ratio ** 2), grid[:, 0]), psi)
        )
        deflection_x = np.arctan(
            np.divide(np.multiply(np.sqrt(1 - self.axis_ratio ** 2), grid[:, 1]), psi)
        )
        return self.rotate_grid_from_reference_frame(
            grid=np.multiply(factor, np.vstack((deflection_y, deflection_x)).T)
        )

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def shear_2d_from(self, grid: aa.type.Grid2DLike):
        """
        Calculate the (gamma_y, gamma_x) shear vector field on a grid of (y,x) arc-second coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """

        convergence = self.convergence_2d_from(grid=grid)

        shear_y = (
            -2
            * convergence
            * np.divide(grid[:, 1] * grid[:, 0], grid[:, 1] ** 2 + grid[:, 0] ** 2)
        )
        shear_x = -convergence * np.divide(
            grid[:, 1] ** 2 - grid[:, 0] ** 2, grid[:, 1] ** 2 + grid[:, 0] ** 2
        )

        shear_field = self.rotate_grid_from_reference_frame(
            grid=np.vstack((shear_y, shear_x)).T
        )

        return aa.VectorYX2DIrregular(vectors=shear_field, grid=grid)


class EllIsothermalInitialize(EllIsothermal):

    pass


class SphIsothermal(EllIsothermal):
    def __init__(
        self, centre: Tuple[float, float] = (0.0, 0.0), einstein_radius: float = 1.0
    ):
        """
        Represents a spherical isothermal density distribution, which is equivalent to the spherical power-law
        density distribution for the value slope: float = 2.0

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        einstein_radius
            The arc-second Einstein radius.
        """
        super().__init__(
            centre=centre, elliptical_comps=(0.0, 0.0), einstein_radius=einstein_radius
        )

    @property
    def axis_ratio(self):
        return 1.0

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def potential_2d_from(self, grid: aa.type.Grid2DLike):
        """
        Calculate the potential on a grid of (y,x) arc-second coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        eta = self.grid_to_elliptical_radii(grid)
        return 2.0 * self.einstein_radius_rescaled * eta

    @aa.grid_dec.grid_2d_to_vector_yx
    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):
        """
        Calculate the deflection angles on a grid of (y,x) arc-second coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        return self.grid_to_grid_cartesian(
            grid=grid,
            radius=np.full(grid.shape[0], 2.0 * self.einstein_radius_rescaled),
        )
