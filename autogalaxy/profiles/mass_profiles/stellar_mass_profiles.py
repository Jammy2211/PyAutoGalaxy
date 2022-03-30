import copy
import numpy as np
from scipy.special import wofz
from scipy.integrate import quad
from typing import List, Tuple

import autoarray as aa

from autogalaxy.profiles.mass_profiles import MassProfile
from autogalaxy.profiles.mass_profiles.mass_profiles import (
    MassProfileMGE,
    MassProfileCSE,
)

from autogalaxy.profiles.mass_profiles.mass_profiles import psi_from


class StellarProfile:

    pass


class EllGaussian(MassProfile, StellarProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        sigma: float = 0.01,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The elliptical Gaussian light profile.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps
            The first and second ellipticity components of the elliptical coordinate system, (see the module
            `autogalaxy -> convert.py` for the convention).
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        sigma
            The sigma value of the Gaussian.
        """

        super(EllGaussian, self).__init__(
            centre=centre, elliptical_comps=elliptical_comps
        )
        super(MassProfile, self).__init__(
            centre=centre, elliptical_comps=elliptical_comps
        )
        self.mass_to_light_ratio = mass_to_light_ratio
        self.intensity = intensity
        self.sigma = sigma

    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """
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

        deflections = (
            self.mass_to_light_ratio
            * self.intensity
            * self.sigma
            * np.sqrt((2 * np.pi) / (1.0 - self.axis_ratio ** 2.0))
            * self.zeta_from(grid=grid)
        )

        return self.rotate_grid_from_reference_frame(
            np.multiply(
                1.0, np.vstack((-1.0 * np.imag(deflections), np.real(deflections))).T
            )
        )

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

        Note: sigma is divided by sqrt(q) here.

        """

        def calculate_deflection_component(npow, index):

            deflection_grid = self.axis_ratio * grid[:, index]

            for i in range(grid.shape[0]):

                deflection_grid[i] *= (
                    self.intensity
                    * self.mass_to_light_ratio
                    * quad(
                        self.deflection_func,
                        a=0.0,
                        b=1.0,
                        args=(
                            grid[i, 0],
                            grid[i, 1],
                            npow,
                            self.axis_ratio,
                            self.sigma / np.sqrt(self.axis_ratio),
                        ),
                    )[0]
                )

            return deflection_grid

        deflection_y = calculate_deflection_component(1.0, 0)
        deflection_x = calculate_deflection_component(0.0, 1)

        return self.rotate_grid_from_reference_frame(
            np.multiply(1.0, np.vstack((deflection_y, deflection_x)).T)
        )

    @staticmethod
    def deflection_func(u, y, x, npow, axis_ratio, sigma):
        eta_u = np.sqrt(axis_ratio) * np.sqrt(
            (u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u))))
        )

        return np.exp(-0.5 * np.square(np.divide(eta_u, sigma))) / (
            (1 - (1 - axis_ratio ** 2) * u) ** (npow + 0.5)
        )

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
        return self.convergence_func(self.grid_to_eccentric_radii(grid))

    def convergence_func(self, grid_radius: float) -> float:
        return self.mass_to_light_ratio * self.image_2d_via_radii_from(grid_radius)

    def image_2d_via_radii_from(self, grid_radii: np.ndarray):
        """Calculate the intensity of the Gaussian light profile on a grid of radial coordinates.

        Parameters
        ----------
        grid_radii
            The radial distance from the centre of the profile. for each coordinate on the grid.

        Note: sigma is divided by sqrt(q) here.
        """
        return np.multiply(
            self.intensity,
            np.exp(
                -0.5
                * np.square(
                    np.divide(grid_radii, self.sigma / np.sqrt(self.axis_ratio))
                )
            ),
        )

    @property
    def axis_ratio(self):
        axis_ratio = super().axis_ratio
        return axis_ratio if axis_ratio < 0.9999 else 0.9999

    def zeta_from(self, grid: aa.type.Grid2DLike):
        q2 = self.axis_ratio ** 2.0
        ind_pos_y = grid[:, 0] >= 0
        shape_grid = np.shape(grid)
        output_grid = np.zeros((shape_grid[0]), dtype=np.complex128)
        scale_factor = self.axis_ratio / (self.sigma * np.sqrt(2.0 * (1.0 - q2)))

        xs_0 = grid[:, 1][ind_pos_y] * scale_factor
        ys_0 = grid[:, 0][ind_pos_y] * scale_factor
        xs_1 = grid[:, 1][~ind_pos_y] * scale_factor
        ys_1 = -grid[:, 0][~ind_pos_y] * scale_factor

        output_grid[ind_pos_y] = -1j * (
            wofz(xs_0 + 1j * ys_0)
            - np.exp(-(xs_0 ** 2.0) * (1.0 - q2) - ys_0 * ys_0 * (1.0 / q2 - 1.0))
            * wofz(self.axis_ratio * xs_0 + 1j * ys_0 / self.axis_ratio)
        )

        output_grid[~ind_pos_y] = np.conj(
            -1j
            * (
                wofz(xs_1 + 1j * ys_1)
                - np.exp(-(xs_1 ** 2.0) * (1.0 - q2) - ys_1 * ys_1 * (1.0 / q2 - 1.0))
                * wofz(self.axis_ratio * xs_1 + 1j * ys_1 / self.axis_ratio)
            )
        )

        return output_grid

    def with_new_normalization(self, normalization):

        mass_profile = copy.copy(self)
        mass_profile.mass_to_light_ratio = normalization
        return mass_profile


# noinspection PyAbstractClass
class AbstractEllSersic(MassProfile, MassProfileMGE, MassProfileCSE, StellarProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        sersic_index: float = 0.6,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The Sersic mass profile, the mass profiles of the light profiles that are used to fit and subtract the lens \
        model_galaxy's light.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps
            The first and second ellipticity components of the elliptical coordinate system, (see the module
            `autogalaxy -> convert.py` for the convention).
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second).
        effective_radius
            The radius containing half the light of this profile.
        sersic_index
            Controls the concentration of the profile (lower -> less concentrated, higher -> more concentrated).
        mass_to_light_ratio
            The mass-to-light ratio of the light profiles
        """
        super(AbstractEllSersic, self).__init__(
            centre=centre, elliptical_comps=elliptical_comps
        )
        super(MassProfile, self).__init__(
            centre=centre, elliptical_comps=elliptical_comps
        )
        super(MassProfileMGE, self).__init__()
        super(MassProfileCSE, self).__init__()
        self.mass_to_light_ratio = mass_to_light_ratio
        self.intensity = intensity
        self.effective_radius = effective_radius
        self.sersic_index = sersic_index

    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):
        return self.deflections_2d_via_cse_from(grid=grid)

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_2d_via_mge_from(self, grid: aa.type.Grid2DLike):
        """
        Calculate the projected 2D deflection angles from a grid of (y,x) arc second coordinates, by computing and
        summing the convergence of each individual cse used to decompose the mass profile.

        The cored steep elliptical (cse) decomposition of a the elliptical NFW mass
        profile (e.g. `decompose_convergence_via_cse`) is using equation (12) of
        Oguri 2021 (https://arxiv.org/abs/2106.11464).

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the convergence is computed on.
        """
        return self._deflections_2d_via_mge_from(
            grid=grid, sigmas_factor=np.sqrt(self.axis_ratio)
        )

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_2d_via_cse_from(self, grid: aa.type.Grid2DLike):
        """
        Calculate the projected 2D deflection angles from a grid of (y,x) arc second coordinates, by computing and
        summing the convergence of each individual cse used to decompose the mass profile.

        The cored steep elliptical (cse) decomposition of a the elliptical NFW mass
        profile (e.g. `decompose_convergence_via_cse`) is using equation (12) of
        Oguri 2021 (https://arxiv.org/abs/2106.11464).

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the convergence is computed on.
        """
        return self._deflections_2d_via_cse_from(grid=grid)

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
        return self.convergence_func(self.grid_to_eccentric_radii(grid))

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def convergence_2d_via_mge_from(self, grid: aa.type.Grid2DLike):
        """
        Calculate the projected convergence at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the convergence is computed on.

        """

        eccentric_radii = self.grid_to_eccentric_radii(grid=grid)

        return self._convergence_2d_via_mge_from(grid_radii=eccentric_radii)

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

        elliptical_radii = self.grid_to_elliptical_radii(grid=grid)

        return self._convergence_2d_via_cse_from(grid_radii=elliptical_radii)

    def convergence_func(self, grid_radius: float) -> float:
        return self.mass_to_light_ratio * self.image_2d_via_radii_from(grid_radius)

    @aa.grid_dec.grid_2d_to_structure
    def potential_2d_from(self, grid: aa.type.Grid2DLike):
        return np.zeros(shape=grid.shape[0])

    def image_2d_via_radii_from(self, radius: np.ndarray):
        """
        Returns the intensity of the profile at a given radius.

            Parameters
            ----------
            radius
                The distance from the centre of the profile.
        """
        return self.intensity * np.exp(
            -self.sersic_constant
            * (((radius / self.effective_radius) ** (1.0 / self.sersic_index)) - 1)
        )

    def decompose_convergence_via_mge(self) -> Tuple[List, List]:
        radii_min = self.effective_radius / 100.0
        radii_max = self.effective_radius * 20.0

        def sersic_2d(r):
            return (
                self.mass_to_light_ratio
                * self.intensity
                * np.exp(
                    -self.sersic_constant
                    * (((r / self.effective_radius) ** (1.0 / self.sersic_index)) - 1.0)
                )
            )

        return self._decompose_convergence_via_mge(
            func=sersic_2d, radii_min=radii_min, radii_max=radii_max
        )

    def decompose_convergence_via_cse(self,) -> Tuple[List, List]:
        """
        Decompose the convergence of the Sersic profile into cored steep elliptical (cse) profiles.

        This decomposition uses the standard 2d profile of a Sersic mass profile.

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

        upper_dex, lower_dex, total_cses, sample_points = cse_settings_from(
            effective_radius=self.effective_radius,
            sersic_index=self.sersic_index,
            sersic_constant=self.sersic_constant,
            mass_to_light_gradient=0.0,
        )

        scaled_effective_radius = self.effective_radius / np.sqrt(self.axis_ratio)
        radii_min = scaled_effective_radius / 10.0 ** lower_dex
        radii_max = scaled_effective_radius * 10.0 ** upper_dex

        def sersic_2d(r):
            return (
                self.mass_to_light_ratio
                * self.intensity
                * np.exp(
                    -self.sersic_constant
                    * (
                        ((r / scaled_effective_radius) ** (1.0 / self.sersic_index))
                        - 1.0
                    )
                )
            )

        return self._decompose_convergence_via_cse_from(
            func=sersic_2d,
            radii_min=radii_min,
            radii_max=radii_max,
            total_cses=total_cses,
            sample_points=sample_points,
        )

    @property
    def sersic_constant(self):
        """A parameter derived from Sersic index which ensures that effective radius contains 50% of the profile's
        total integrated light.
        """
        return (
            (2 * self.sersic_index)
            - (1.0 / 3.0)
            + (4.0 / (405.0 * self.sersic_index))
            + (46.0 / (25515.0 * self.sersic_index ** 2))
            + (131.0 / (1148175.0 * self.sersic_index ** 3))
            - (2194697.0 / (30690717750.0 * self.sersic_index ** 4))
        )

    @property
    def ellipticity_rescale(self):
        return 1.0 - ((1.0 - self.axis_ratio) / 2.0)

    @property
    def elliptical_effective_radius(self):
        """
        The effective_radius of a Sersic light profile is defined as the circular effective radius. This is the \
        radius within which a circular aperture contains half the profiles's total integrated light. For elliptical \
        systems, this won't robustly capture the light profile's elliptical shape.

        The elliptical effective radius instead describes the major-axis radius of the ellipse containing \
        half the light, and may be more appropriate for highly flattened systems like disk galaxies.
        """
        return self.effective_radius / np.sqrt(self.axis_ratio)

    def with_new_normalization(self, normalization):

        mass_profile = copy.copy(self)
        mass_profile.mass_to_light_ratio = normalization
        return mass_profile


class EllSersic(AbstractEllSersic, MassProfileMGE, MassProfileCSE):
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
            sersic_constant = self.sersic_constant

            deflection_grid = self.axis_ratio * grid[:, index]

            for i in range(grid.shape[0]):

                deflection_grid[i] *= (
                    self.intensity
                    * self.mass_to_light_ratio
                    * quad(
                        self.deflection_func,
                        a=0.0,
                        b=1.0,
                        args=(
                            grid[i, 0],
                            grid[i, 1],
                            npow,
                            self.axis_ratio,
                            self.sersic_index,
                            self.effective_radius,
                            sersic_constant,
                        ),
                    )[0]
                )

            return deflection_grid

        deflection_y = calculate_deflection_component(1.0, 0)
        deflection_x = calculate_deflection_component(0.0, 1)

        return self.rotate_grid_from_reference_frame(
            np.multiply(1.0, np.vstack((deflection_y, deflection_x)).T)
        )

    @staticmethod
    def deflection_func(
        u, y, x, npow, axis_ratio, sersic_index, effective_radius, sersic_constant
    ):
        eta_u = np.sqrt(axis_ratio) * np.sqrt(
            (u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u))))
        )

        return np.exp(
            -sersic_constant
            * (((eta_u / effective_radius) ** (1.0 / sersic_index)) - 1)
        ) / ((1 - (1 - axis_ratio ** 2) * u) ** (npow + 0.5))


class SphSersic(EllSersic):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        sersic_index: float = 0.6,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The Sersic mass profile, the mass profiles of the light profiles that are used to fit and subtract the lens
        model_galaxy's light.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius
            The circular radius containing half the light of this profile.
        sersic_index
            Controls the concentration of the profile (lower -> less concentrated, higher -> more concentrated).
        mass_to_light_ratio
            The mass-to-light ratio of the light profile.
        """
        super().__init__(
            centre=centre,
            elliptical_comps=(0.0, 0.0),
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class EllExponential(EllSersic):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The EllExponential mass profile, the mass profiles of the light profiles that are used to fit and
        subtract the lens model_galaxy's light.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps
            The first and second ellipticity components of the elliptical coordinate system, (see the module
            `autogalaxy -> convert.py` for the convention).
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second).
        effective_radius
            The circular radius containing half the light of this profile.
        mass_to_light_ratio
            The mass-to-light ratio of the light profiles
        """
        super().__init__(
            centre=centre,
            elliptical_comps=elliptical_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=1.0,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class SphExponential(EllExponential):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The Exponential mass profile, the mass profiles of the light profiles that are used to fit and subtract the lens
        model_galaxy's light.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second).
        effective_radius
            The circular radius containing half the light of this profile.
        mass_to_light_ratio
            The mass-to-light ratio of the light profiles.
        """
        super().__init__(
            centre=centre,
            elliptical_comps=(0.0, 0.0),
            intensity=intensity,
            effective_radius=effective_radius,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class EllDevVaucouleurs(EllSersic):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The EllDevVaucouleurs mass profile, the mass profiles of the light profiles that are used to fit and
        subtract the lens model_galaxy's light.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps
            The first and second ellipticity components of the elliptical coordinate system, (see the module
            `autogalaxy -> convert.py` for the convention).
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second).
        effective_radius
            The radius containing half the light of this profile.
        mass_to_light_ratio
            The mass-to-light ratio of the light profile.
        """
        super().__init__(
            centre=centre,
            elliptical_comps=elliptical_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=4.0,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class SphDevVaucouleurs(EllDevVaucouleurs):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The DevVaucouleurs mass profile, the mass profiles of the light profiles that are used to fit and subtract the
        lens model_galaxy's light.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second).
        effective_radius
            The circular radius containing half the light of this profile.
        mass_to_light_ratio
            The mass-to-light ratio of the light profiles.
        """
        super().__init__(
            centre=centre,
            elliptical_comps=(0.0, 0.0),
            intensity=intensity,
            effective_radius=effective_radius,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class EllSersicRadialGradient(AbstractEllSersic):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        sersic_index: float = 0.6,
        mass_to_light_ratio: float = 1.0,
        mass_to_light_gradient: float = 0.0,
    ):
        """
        Setup a Sersic mass and light profiles.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps
            The first and second ellipticity components of the elliptical coordinate system, (see the module
            `autogalaxy -> convert.py` for the convention).
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second).
        effective_radius
            The circular radius containing half the light of this profile.
        sersic_index
            Controls the concentration of the profile (lower -> less concentrated, higher -> more concentrated).
        mass_to_light_ratio
            The mass-to-light ratio of the light profile.
        mass_to_light_gradient
            The mass-to-light radial gradient.
        """
        super().__init__(
            centre=centre,
            elliptical_comps=elliptical_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            mass_to_light_ratio=mass_to_light_ratio,
        )
        self.mass_to_light_gradient = mass_to_light_gradient

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
            sersic_constant = self.sersic_constant

            deflection_grid = self.axis_ratio * grid[:, index]

            for i in range(grid.shape[0]):

                deflection_grid[i] *= (
                    self.intensity
                    * self.mass_to_light_ratio
                    * quad(
                        self.deflection_func,
                        a=0.0,
                        b=1.0,
                        args=(
                            grid[i, 0],
                            grid[i, 1],
                            npow,
                            self.axis_ratio,
                            self.sersic_index,
                            self.effective_radius,
                            self.mass_to_light_gradient,
                            sersic_constant,
                        ),
                    )[0]
                )
            return deflection_grid

        deflection_y = calculate_deflection_component(1.0, 0)
        deflection_x = calculate_deflection_component(0.0, 1)

        return self.rotate_grid_from_reference_frame(
            np.multiply(1.0, np.vstack((deflection_y, deflection_x)).T)
        )

    @staticmethod
    def deflection_func(
        u,
        y,
        x,
        npow,
        axis_ratio,
        sersic_index,
        effective_radius,
        mass_to_light_gradient,
        sersic_constant,
    ):
        eta_u = np.sqrt(axis_ratio) * np.sqrt(
            (u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u))))
        )

        return (
            (((axis_ratio * eta_u) / effective_radius) ** -mass_to_light_gradient)
            * np.exp(
                -sersic_constant
                * (((eta_u / effective_radius) ** (1.0 / sersic_index)) - 1)
            )
            / ((1 - (1 - axis_ratio ** 2) * u) ** (npow + 0.5))
        )

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
        return self.convergence_func(self.grid_to_eccentric_radii(grid))

    def convergence_func(self, grid_radius: float) -> float:
        return (
            self.mass_to_light_ratio
            * (
                ((self.axis_ratio * grid_radius) / self.effective_radius)
                ** -self.mass_to_light_gradient
            )
            * self.image_2d_via_radii_from(grid_radius)
        )

    def decompose_convergence_via_mge(self):
        radii_min = self.effective_radius / 100.0
        radii_max = self.effective_radius * 20.0

        def sersic_radial_gradient_2D(r):
            return (
                self.mass_to_light_ratio
                * self.intensity
                * (
                    ((self.axis_ratio * r) / self.effective_radius)
                    ** -self.mass_to_light_gradient
                )
                * np.exp(
                    -self.sersic_constant
                    * (((r / self.effective_radius) ** (1.0 / self.sersic_index)) - 1.0)
                )
            )

        return self._decompose_convergence_via_mge(
            func=sersic_radial_gradient_2D, radii_min=radii_min, radii_max=radii_max
        )

    def decompose_convergence_via_cse(self) -> Tuple[List, List]:
        """
        Decompose the convergence of the Sersic profile into singular isothermal elliptical (sie) profiles.

        This decomposition uses the standard 2d profile of a Sersic mass profile.

        Parameters
        ----------
        func
            The function representing the profile that is decomposed into CSEs.
        radii_min:
            The minimum radius to fit
        radii_max:
            The maximum radius to fit
        total_sies
            The number of SIEs used to approximate the input func.
        sample_points: int (should be larger than 'total_sies')
            The number of data points to fit

        Returns
        -------
        Tuple[List, List]
            A list of amplitudes and core radii of every singular isothernal ellipsoids (sie) the mass profile is decomposed
            into.
        """

        upper_dex, lower_dex, total_cses, sample_points = cse_settings_from(
            effective_radius=self.effective_radius,
            sersic_index=self.sersic_index,
            sersic_constant=self.sersic_constant,
            mass_to_light_gradient=self.mass_to_light_gradient,
        )

        scaled_effective_radius = self.effective_radius / np.sqrt(self.axis_ratio)
        radii_min = scaled_effective_radius / 10.0 ** lower_dex
        radii_max = scaled_effective_radius * 10.0 ** upper_dex

        def sersic_radial_gradient_2D(r):
            return (
                self.mass_to_light_ratio
                * self.intensity
                * (
                    ((self.axis_ratio * r) / scaled_effective_radius)
                    ** -self.mass_to_light_gradient
                )
                * np.exp(
                    -self.sersic_constant
                    * (
                        ((r / scaled_effective_radius) ** (1.0 / self.sersic_index))
                        - 1.0
                    )
                )
            )

        return self._decompose_convergence_via_cse_from(
            func=sersic_radial_gradient_2D,
            radii_min=radii_min,
            radii_max=radii_max,
            total_cses=total_cses,
            sample_points=sample_points,
        )


class SphSersicRadialGradient(EllSersicRadialGradient):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        sersic_index: float = 0.6,
        mass_to_light_ratio: float = 1.0,
        mass_to_light_gradient: float = 0.0,
    ):
        """
        Setup a Sersic mass and light profiles.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second).
        effective_radius
            The circular radius containing half the light of this profile.
        sersic_index
            Controls the concentration of the profile (lower -> less concentrated, higher -> more concentrated).
        mass_to_light_ratio
            The mass-to-light ratio of the light profile.
        mass_to_light_gradient
            The mass-to-light radial gradient.
        """
        super().__init__(
            centre=centre,
            elliptical_comps=(0.0, 0.0),
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            mass_to_light_ratio=mass_to_light_ratio,
            mass_to_light_gradient=mass_to_light_gradient,
        )


class EllSersicCore(EllSersic):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
        radius_break: float = 0.01,
        intensity_break: float = 0.05,
        gamma: float = 0.25,
        alpha: float = 3.0,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The elliptical cored-Sersic light profile.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps
            The first and second ellipticity components of the elliptical coordinate system, (see the module
            `autogalaxy -> convert.py` for the convention).
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        effective_radius
            The circular radius containing half the light of this profile.
        sersic_index
            Controls the concentration of the profile (lower -> less concentrated, higher -> more concentrated).
        radius_break
            The break radius separating the inner power-law (with logarithmic slope gamma) and outer Sersic function.
        intensity_break
            The intensity at the break radius.
        gamma
            The logarithmic power-law slope of the inner core profiles
        alpha :
            Controls the sharpness of the transition between the inner core / outer Sersic profiles.
        """

        super().__init__(
            centre=centre,
            elliptical_comps=elliptical_comps,
            intensity=intensity_break,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            mass_to_light_ratio=mass_to_light_ratio,
        )

        self.radius_break = radius_break
        self.intensity_break = intensity_break
        self.alpha = alpha
        self.gamma = gamma

    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):
        return self.deflections_2d_via_mge_from(grid=grid)

    def image_2d_via_radii_from(self, grid_radii: np.ndarray):
        """
        Calculate the intensity of the cored-Sersic light profile on a grid of radial coordinates.

        Parameters
        ----------
        grid_radii
            The radial distance from the centre of the profile. for each coordinate on the grid.
        """
        return np.multiply(
            np.multiply(
                self.intensity_prime,
                np.power(
                    np.add(
                        1,
                        np.power(np.divide(self.radius_break, grid_radii), self.alpha),
                    ),
                    (self.gamma / self.alpha),
                ),
            ),
            np.exp(
                np.multiply(
                    -self.sersic_constant,
                    (
                        np.power(
                            np.divide(
                                np.add(
                                    np.power(grid_radii, self.alpha),
                                    (self.radius_break ** self.alpha),
                                ),
                                (self.effective_radius ** self.alpha),
                            ),
                            (1.0 / (self.alpha * self.sersic_index)),
                        )
                    ),
                )
            ),
        )

    def decompose_convergence_via_mge(self):

        radii_min = self.effective_radius / 50.0
        radii_max = self.effective_radius * 20.0

        def core_sersic_2D(r):
            return (
                self.mass_to_light_ratio
                * self.intensity_prime
                * (1.0 + (self.radius_break / r) ** self.alpha)
                ** (self.gamma / self.alpha)
                * np.exp(
                    -self.sersic_constant
                    * (
                        (r ** self.alpha + self.radius_break ** self.alpha)
                        / self.effective_radius ** self.alpha
                    )
                    ** (1.0 / (self.sersic_index * self.alpha))
                )
            )

        return self._decompose_convergence_via_mge(
            func=core_sersic_2D, radii_min=radii_min, radii_max=radii_max
        )

    @property
    def intensity_prime(self):
        """Overall intensity normalisation in the rescaled Core-Sersic light profiles (electrons per second)"""
        return (
            self.intensity_break
            * (2.0 ** (-self.gamma / self.alpha))
            * np.exp(
                self.sersic_constant
                * (
                    ((2.0 ** (1.0 / self.alpha)) * self.radius_break)
                    / self.effective_radius
                )
                ** (1.0 / self.sersic_index)
            )
        )


class SphSersicCore(EllSersicCore):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
        radius_break: float = 0.01,
        intensity_break: float = 0.05,
        gamma: float = 0.25,
        alpha: float = 3.0,
    ):
        """
        The elliptical cored-Sersic light profile.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        effective_radius
            The circular radius containing half the light of this profile.
        sersic_index
            Controls the concentration of the profile (lower -> less concentrated, higher -> more concentrated).
        radius_break
            The break radius separating the inner power-law (with logarithmic slope gamma) and outer Sersic function.
        intensity_break
            The intensity at the break radius.
        gamma
            The logarithmic power-law slope of the inner core profiles
        alpha :
            Controls the sharpness of the transition between the inner core / outer Sersic profiles.
        """
        super().__init__(
            centre=centre,
            elliptical_comps=(0.0, 0.0),
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            radius_break=radius_break,
            intensity_break=intensity_break,
            gamma=gamma,
            alpha=alpha,
        )
        self.radius_break = radius_break
        self.intensity_break = intensity_break
        self.alpha = alpha
        self.gamma = gamma


class EllChameleon(MassProfile, StellarProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        core_radius_0: float = 0.01,
        core_radius_1: float = 0.02,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The elliptical Chamelon mass profile.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps
            The first and second ellipticity components of the elliptical coordinate system, (see the module
            `autogalaxy -> convert.py` for the convention).
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        core_radius_0 : the core size of the first elliptical cored Isothermal profile.
        core_radius_1 : core_radius_0 + core_radius_1 is the core size of the second elliptical cored Isothermal profile.
            We use core_radius_1 here is to avoid negative values.

        Profile form:
            mass_to_light_ratio * intensity *\
                (1.0 / Sqrt(x^2 + (y/q)^2 + core_radius_0^2) - 1.0 / Sqrt(x^2 + (y/q)^2 + (core_radius_0 + core_radius_1)**2.0))
        """

        super(EllChameleon, self).__init__(
            centre=centre, elliptical_comps=elliptical_comps
        )
        super(MassProfile, self).__init__(
            centre=centre, elliptical_comps=elliptical_comps
        )
        self.mass_to_light_ratio = mass_to_light_ratio
        self.intensity = intensity
        self.core_radius_0 = core_radius_0
        self.core_radius_1 = core_radius_1

    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):
        return self.deflections_2d_via_analytic_from(grid=grid)

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_2d_via_analytic_from(self, grid: aa.type.Grid2DLike):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.
        Following Eq. (15) and (16), but the parameters are slightly different.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """

        factor = (
            2.0
            * self.mass_to_light_ratio
            * self.intensity
            / (1 + self.axis_ratio)
            * self.axis_ratio
            / np.sqrt(1.0 - self.axis_ratio ** 2.0)
        )

        core_radius_0 = np.sqrt(
            (4.0 * self.core_radius_0 ** 2.0) / (1.0 + self.axis_ratio) ** 2
        )
        core_radius_1 = np.sqrt(
            (4.0 * self.core_radius_1 ** 2.0) / (1.0 + self.axis_ratio) ** 2
        )

        psi0 = psi_from(
            grid=grid, axis_ratio=self.axis_ratio, core_radius=core_radius_0
        )
        psi1 = psi_from(
            grid=grid, axis_ratio=self.axis_ratio, core_radius=core_radius_1
        )

        deflection_y0 = np.arctanh(
            np.divide(
                np.multiply(np.sqrt(1.0 - self.axis_ratio ** 2.0), grid[:, 0]),
                np.add(psi0, self.axis_ratio ** 2.0 * core_radius_0),
            )
        )

        deflection_x0 = np.arctan(
            np.divide(
                np.multiply(np.sqrt(1.0 - self.axis_ratio ** 2.0), grid[:, 1]),
                np.add(psi0, core_radius_0),
            )
        )

        deflection_y1 = np.arctanh(
            np.divide(
                np.multiply(np.sqrt(1.0 - self.axis_ratio ** 2.0), grid[:, 0]),
                np.add(psi1, self.axis_ratio ** 2.0 * core_radius_1),
            )
        )

        deflection_x1 = np.arctan(
            np.divide(
                np.multiply(np.sqrt(1.0 - self.axis_ratio ** 2.0), grid[:, 1]),
                np.add(psi1, core_radius_1),
            )
        )

        deflection_y = np.subtract(deflection_y0, deflection_y1)
        deflection_x = np.subtract(deflection_x0, deflection_x1)

        return self.rotate_grid_from_reference_frame(
            np.multiply(factor, np.vstack((deflection_y, deflection_x)).T)
        )

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
        return self.convergence_func(self.grid_to_elliptical_radii(grid))

    def convergence_func(self, grid_radius: float) -> float:
        return self.mass_to_light_ratio * self.image_2d_via_radii_from(grid_radius)

    @aa.grid_dec.grid_2d_to_structure
    def potential_2d_from(self, grid: aa.type.Grid2DLike):
        return np.zeros(shape=grid.shape[0])

    def image_2d_via_radii_from(self, grid_radii: np.ndarray):
        """Calculate the intensity of the Chamelon light profile on a grid of radial coordinates.

        Parameters
        ----------
        grid_radii
            The radial distance from the centre of the profile. for each coordinate on the grid.
        """

        axis_ratio_factor = (1.0 + self.axis_ratio) ** 2.0

        return np.multiply(
            self.intensity / (1 + self.axis_ratio),
            np.add(
                np.divide(
                    1.0,
                    np.sqrt(
                        np.add(
                            np.square(grid_radii),
                            (4.0 * self.core_radius_0 ** 2.0) / axis_ratio_factor,
                        )
                    ),
                ),
                -np.divide(
                    1.0,
                    np.sqrt(
                        np.add(
                            np.square(grid_radii),
                            (4.0 * self.core_radius_1 ** 2.0) / axis_ratio_factor,
                        )
                    ),
                ),
            ),
        )

    @property
    def axis_ratio(self):
        axis_ratio = super().axis_ratio
        return axis_ratio if axis_ratio < 0.99999 else 0.99999

    def with_new_normalization(self, normalization):

        mass_profile = copy.copy(self)
        mass_profile.mass_to_light_ratio = normalization
        return mass_profile


class SphChameleon(EllChameleon):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        core_radius_0: float = 0.01,
        core_radius_1: float = 0.02,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The spherica; Chameleon mass profile.

        Profile form:
            mass_to_light_ratio * intensity *\
                (1.0 / Sqrt(x^2 + (y/q)^2 + core_radius_0^2) - 1.0 / Sqrt(x^2 + (y/q)^2 + (core_radius_0 + core_radius_1)**2.0))

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps
            The first and second ellipticity components of the elliptical coordinate system, (see the module
            `autogalaxy -> convert.py` for the convention).
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        core_radius_0 : the core size of the first elliptical cored Isothermal profile.
        core_radius_1 : core_radius_0 + core_radius_1 is the core size of the second elliptical cored Isothermal profile.
            We use core_radius_1 here is to avoid negative values.
       """

        super().__init__(
            centre=centre,
            elliptical_comps=(0.0, 0.0),
            intensity=intensity,
            core_radius_0=core_radius_0,
            core_radius_1=core_radius_1,
            mass_to_light_ratio=mass_to_light_ratio,
        )


def cse_settings_from(
    effective_radius, sersic_index, sersic_constant, mass_to_light_gradient
):

    if mass_to_light_gradient > 0.5:

        if effective_radius > 0.2:

            lower_dex = 6.0
            upper_dex = np.min(
                [np.log10((18.0 / sersic_constant) ** sersic_index), 1.1]
            )

            if sersic_index <= 1.2:
                total_cses = 50
                sample_points = 80
            elif sersic_index > 3.8:
                total_cses = 40
                sample_points = 50
                lower_dex = 6.5
            else:
                total_cses = 30
                sample_points = 50

        else:
            if sersic_index <= 1.2:
                upper_dex = 1.0
                total_cses = 50
                sample_points = 80
                lower_dex = 4.5

            elif sersic_index > 3.8:
                total_cses = 40
                sample_points = 50
                lower_dex = 6.0
                upper_dex = 1.5
            else:
                upper_dex = 1.1
                lower_dex = 6.0
                total_cses = 30
                sample_points = 50
    else:

        upper_dex = np.min(
            [
                np.log10((23.0 / sersic_constant) ** sersic_index),
                0.85 - np.log10(effective_radius),
            ]
        )

        if (sersic_index <= 0.9) and (sersic_index > 0.8):
            total_cses = 50
            sample_points = 80
            upper_dex = np.log10((18.0 / sersic_constant) ** sersic_index)
            lower_dex = 4.3 + np.log10(effective_radius)
        elif sersic_index <= 0.8:
            total_cses = 50
            sample_points = 80
            upper_dex = np.log10((16.0 / sersic_constant) ** sersic_index)
            lower_dex = 4.0 + np.log10(effective_radius)
        elif sersic_index > 3.8:
            total_cses = 40
            sample_points = 50
            lower_dex = 4.5 + np.log10(effective_radius)
        else:
            lower_dex = 3.5 + np.log10(effective_radius)
            total_cses = 30
            sample_points = 50

    return upper_dex, lower_dex, total_cses, sample_points
