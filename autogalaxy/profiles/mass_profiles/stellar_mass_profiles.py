from autogalaxy.profiles.mass_profiles.mass_profiles import psi_from
import numpy as np
from autoarray.structures import grids
from autogalaxy.profiles import mass_profiles as mp

from pyquad import quad_grid
from scipy.special import wofz
import typing
import copy

from autogalaxy.profiles.mass_profiles.mass_profiles import MassProfileMGE


class StellarProfile:

    pass


class EllipticalGaussian(mp.EllipticalMassProfile, StellarProfile):
    def __init__(
        self,
        centre: typing.Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        sigma: float = 0.01,
        mass_to_light_ratio: float = 1.0,
    ):
        """The elliptical Gaussian light profile.

        Parameters
        ----------
        centre : (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps : (float, float)
            The first and second ellipticity components of the elliptical coordinate system, where
            fac = (1 - axis_ratio) / (1 + axis_ratio), ellip_y = fac * sin(2*phi) and ellip_x = fac * cos(2*phi).
        intensity : float
            Overall intensity normalisation of the light profiles (electrons per second).
        sigma : float
            The sigma value of the Gaussian.
        """

        super(EllipticalGaussian, self).__init__(
            centre=centre, elliptical_comps=elliptical_comps
        )
        super(mp.EllipticalMassProfile, self).__init__(
            centre=centre, elliptical_comps=elliptical_comps
        )
        self.mass_to_light_ratio = mass_to_light_ratio
        self.intensity = intensity
        self.sigma = sigma

        if self.axis_ratio > 0.9999:
            self.axis_ratio = 0.9999

    def zeta_from_grid(self, grid):
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

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def deflections_from_grid(self, grid):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : aa.Grid2D
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """

        deflections = (
            self.mass_to_light_ratio
            * self.intensity
            * self.sigma
            * np.sqrt((2 * np.pi) / (1.0 - self.axis_ratio ** 2.0))
            * self.zeta_from_grid(grid=grid)
        )

        return self.rotate_grid_from_profile(
            np.multiply(
                1.0, np.vstack((-1.0 * np.imag(deflections), np.real(deflections))).T
            )
        )

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def deflections_from_grid_via_integrator(self, grid):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : aa.Grid2D
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        Note: sigma is divided by sqrt(q) here.

        """

        def calculate_deflection_component(npow, index):
            deflection_grid = self.axis_ratio * grid[:, index]
            deflection_grid *= (
                self.intensity
                * self.mass_to_light_ratio
                * quad_grid(
                    self.deflection_func,
                    0.0,
                    1.0,
                    grid,
                    args=(npow, self.axis_ratio, self.sigma / np.sqrt(self.axis_ratio)),
                )[0]
            )

            return deflection_grid

        deflection_y = calculate_deflection_component(1.0, 0)
        deflection_x = calculate_deflection_component(0.0, 1)

        return self.rotate_grid_from_profile(
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

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def convergence_from_grid(self, grid):
        """Calculate the projected convergence at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : aa.Grid2D
            The grid of (y,x) arc-second coordinates the convergence is computed on.

        """
        return self.convergence_func(self.grid_to_eccentric_radii(grid))

    def convergence_func(self, grid_radius):
        return self.mass_to_light_ratio * self.image_from_grid_radii(grid_radius)

    def image_from_grid_radii(self, grid_radii):
        """Calculate the intensity of the Gaussian light profile on a grid of radial coordinates.

        Parameters
        ----------
        grid_radii : float
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

    def with_new_normalization(self, normalization):

        mass_profile = copy.copy(self)
        mass_profile.mass_to_light_ratio = normalization
        return mass_profile


# noinspection PyAbstractClass
class AbstractEllipticalSersic(
    mp.EllipticalMassProfile, MassProfileMGE, StellarProfile
):
    def __init__(
        self,
        centre: typing.Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0),
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
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps : (float, float)
            The first and second ellipticity components of the elliptical coordinate system, where
            fac = (1 - axis_ratio) / (1 + axis_ratio), ellip_y = fac * sin(2*phi) and ellip_x = fac * cos(2*phi).
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second).
        effective_radius : float
            The radius containing half the light of this profile.
        sersic_index : float
            Controls the concentration of the of the profile (lower value -> less concentrated, \
            higher value -> more concentrated).
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profiles
        """
        super(AbstractEllipticalSersic, self).__init__(
            centre=centre, elliptical_comps=elliptical_comps
        )
        super(mp.EllipticalMassProfile, self).__init__(
            centre=centre, elliptical_comps=elliptical_comps
        )
        super(MassProfileMGE, self).__init__()
        self.mass_to_light_ratio = mass_to_light_ratio
        self.intensity = intensity
        self.effective_radius = effective_radius
        self.sersic_index = sersic_index

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def convergence_from_grid(self, grid):
        """Calculate the projected convergence at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : aa.Grid2D
            The grid of (y,x) arc-second coordinates the convergence is computed on.

        """
        return self.convergence_func(self.grid_to_eccentric_radii(grid))

    def convergence_func(self, grid_radius):
        return self.mass_to_light_ratio * self.image_from_grid_radii(grid_radius)

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def convergence_from_grid_via_gaussians(self, grid):
        """Calculate the projected convergence at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : aa.Grid2D
            The grid of (y,x) arc-second coordinates the convergence is computed on.

        """

        eccentric_radii = self.grid_to_eccentric_radii(grid=grid)

        return self._convergence_from_grid_via_gaussians(grid_radii=eccentric_radii)

    @grids.grid_like_to_structure
    def potential_from_grid(self, grid):
        return np.zeros(shape=grid.shape[0])

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def deflections_from_grid(self, grid):
        return self._deflections_from_grid_via_gaussians(
            grid=grid, sigmas_factor=np.sqrt(self.axis_ratio)
        )

    @property
    def ellipticity_rescale(self):
        return 1.0 - ((1.0 - self.axis_ratio) / 2.0)

    def image_from_grid_radii(self, radius):
        """
        Returns the intensity of the profile at a given radius.

            Parameters
            ----------
            radius : float
                The distance from the centre of the profile.
        """
        return self.intensity * np.exp(
            -self.sersic_constant
            * (((radius / self.effective_radius) ** (1.0 / self.sersic_index)) - 1)
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
    def elliptical_effective_radius(self):
        """The effective_radius of a Sersic light profile is defined as the circular effective radius. This is the \
        radius within which a circular aperture contains half the profiles's total integrated light. For elliptical \
        systems, this won't robustly capture the light profile's elliptical shape.

        The elliptical effective radius instead describes the major-axis radius of the ellipse containing \
        half the light, and may be more appropriate for highly flattened systems like disk galaxies."""
        return self.effective_radius / np.sqrt(self.axis_ratio)

    def decompose_convergence_into_gaussians(self):
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

        return self._decompose_convergence_into_gaussians(
            func=sersic_2d, radii_min=radii_min, radii_max=radii_max
        )

    def with_new_normalization(self, normalization):

        mass_profile = copy.copy(self)
        mass_profile.mass_to_light_ratio = normalization
        return mass_profile


class EllipticalSersic(AbstractEllipticalSersic, MassProfileMGE):
    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def deflections_from_grid_via_integrator(self, grid):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : aa.Grid2D
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """

        def calculate_deflection_component(npow, index):
            sersic_constant = self.sersic_constant

            deflection_grid = self.axis_ratio * grid[:, index]
            deflection_grid *= (
                self.intensity
                * self.mass_to_light_ratio
                * quad_grid(
                    self.deflection_func,
                    0.0,
                    1.0,
                    grid,
                    args=(
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

        return self.rotate_grid_from_profile(
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


class SphericalSersic(EllipticalSersic):
    def __init__(
        self,
        centre: typing.Tuple[float, float] = (0.0, 0.0),
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
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius : float
            The circular radius containing half the light of this profile.
        sersic_index : float
            Controls the concentration of the of the profile (lower value -> less concentrated, \
            higher value -> more concentrated).
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profile.
        """
        super(SphericalSersic, self).__init__(
            centre=centre,
            elliptical_comps=(0.0, 0.0),
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class EllipticalExponential(EllipticalSersic):
    def __init__(
        self,
        centre: typing.Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The EllipticalExponential mass profile, the mass profiles of the light profiles that are used to fit and
        subtract the lens model_galaxy's light.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps : (float, float)
            The first and second ellipticity components of the elliptical coordinate system, where
            fac = (1 - axis_ratio) / (1 + axis_ratio), ellip_y = fac * sin(2*phi) and ellip_x = fac * cos(2*phi).
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second).
        effective_radius : float
            The circular radius containing half the light of this profile.
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profiles
        """
        super(EllipticalExponential, self).__init__(
            centre=centre,
            elliptical_comps=elliptical_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=1.0,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class SphericalExponential(EllipticalExponential):
    def __init__(
        self,
        centre: typing.Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The Exponential mass profile, the mass profiles of the light profiles that are used to fit and subtract the lens
        model_galaxy's light.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second).
        effective_radius : float
            The circular radius containing half the light of this profile.
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profiles.
        """
        super(SphericalExponential, self).__init__(
            centre=centre,
            elliptical_comps=(0.0, 0.0),
            intensity=intensity,
            effective_radius=effective_radius,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class EllipticalDevVaucouleurs(EllipticalSersic):
    def __init__(
        self,
        centre: typing.Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The EllipticalDevVaucouleurs mass profile, the mass profiles of the light profiles that are used to fit and
        subtract the lens model_galaxy's light.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps : (float, float)
            The first and second ellipticity components of the elliptical coordinate system, where
            fac = (1 - axis_ratio) / (1 + axis_ratio), ellip_y = fac * sin(2*phi) and ellip_x = fac * cos(2*phi).
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second).
        effective_radius : float
            The radius containing half the light of this profile.
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profile.
        """
        super(EllipticalDevVaucouleurs, self).__init__(
            centre=centre,
            elliptical_comps=elliptical_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=4.0,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class SphericalDevVaucouleurs(EllipticalDevVaucouleurs):
    def __init__(
        self,
        centre: typing.Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The DevVaucouleurs mass profile, the mass profiles of the light profiles that are used to fit and subtract the
        lens model_galaxy's light.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second).
        effective_radius : float
            The circular radius containing half the light of this profile.
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profiles.
        """
        super(SphericalDevVaucouleurs, self).__init__(
            centre=centre,
            elliptical_comps=(0.0, 0.0),
            intensity=intensity,
            effective_radius=effective_radius,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class EllipticalSersicRadialGradient(AbstractEllipticalSersic):
    def __init__(
        self,
        centre: typing.Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0),
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
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps : (float, float)
            The first and second ellipticity components of the elliptical coordinate system, where
            fac = (1 - axis_ratio) / (1 + axis_ratio), ellip_y = fac * sin(2*phi) and ellip_x = fac * cos(2*phi).
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second).
        effective_radius : float
            The circular radius containing half the light of this profile.
        sersic_index : float
            Controls the concentration of the of the profile (lower value -> less concentrated, \
            higher value -> more concentrated).
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profile.
        mass_to_light_gradient : float
            The mass-to-light radial gradient.
        """
        super(EllipticalSersicRadialGradient, self).__init__(
            centre=centre,
            elliptical_comps=elliptical_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            mass_to_light_ratio=mass_to_light_ratio,
        )
        self.mass_to_light_gradient = mass_to_light_gradient

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def convergence_from_grid(self, grid):
        """Calculate the projected convergence at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : aa.Grid2D
            The grid of (y,x) arc-second coordinates the convergence is computed on.

        """
        return self.convergence_func(self.grid_to_eccentric_radii(grid))

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def deflections_via_integrator_from_grid(self, grid):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : aa.Grid2D
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """

        def calculate_deflection_component(npow, index):
            sersic_constant = self.sersic_constant

            deflection_grid = self.axis_ratio * grid[:, index]
            deflection_grid *= (
                self.intensity
                * self.mass_to_light_ratio
                * quad_grid(
                    self.deflection_func,
                    0.0,
                    1.0,
                    grid,
                    args=(
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

        return self.rotate_grid_from_profile(
            np.multiply(1.0, np.vstack((deflection_y, deflection_x)).T)
        )

    def convergence_func(self, grid_radius):
        return (
            self.mass_to_light_ratio
            * (
                ((self.axis_ratio * grid_radius) / self.effective_radius)
                ** -self.mass_to_light_gradient
            )
            * self.image_from_grid_radii(grid_radius)
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

    def decompose_convergence_into_gaussians(self):
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

        return self._decompose_convergence_into_gaussians(
            func=sersic_radial_gradient_2D, radii_min=radii_min, radii_max=radii_max
        )


class SphericalSersicRadialGradient(EllipticalSersicRadialGradient):
    def __init__(
        self,
        centre: typing.Tuple[float, float] = (0.0, 0.0),
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
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second).
        effective_radius : float
            The circular radius containing half the light of this profile.
        sersic_index : float
            Controls the concentration of the of the profile (lower value -> less concentrated, \
            higher value -> more concentrated).
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profile.
        mass_to_light_gradient : float
            The mass-to-light radial gradient.
        """
        super(SphericalSersicRadialGradient, self).__init__(
            centre=centre,
            elliptical_comps=(0.0, 0.0),
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            mass_to_light_ratio=mass_to_light_ratio,
            mass_to_light_gradient=mass_to_light_gradient,
        )


class EllipticalCoreSersic(EllipticalSersic):
    def __init__(
        self,
        centre: typing.Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
        radius_break: float = 0.01,
        intensity_break: float = 0.05,
        gamma: float = 0.25,
        alpha: float = 3.0,
        mass_to_light_ratio: float = 1.0,
    ):
        """ The elliptical cored-Sersic light profile.

        Parameters
        ----------
        centre : (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps : (float, float)
            The first and second ellipticity components of the elliptical coordinate system, where
            fac = (1 - axis_ratio) / (1 + axis_ratio), ellip_y = fac * sin(2*phi) and ellip_x = fac * cos(2*phi).
        intensity : float
            Overall intensity normalisation of the light profiles (electrons per second).
        effective_radius : float
            The circular radius containing half the light of this profile.
        sersic_index : Int
            Controls the concentration of the of the profile (lower value -> less concentrated, \
            higher value -> more concentrated).
        radius_break : Float
            The break radius separating the inner power-law (with logarithmic slope gamma) and outer Sersic function.
        intensity_break : Float
            The intensity at the break radius.
        gamma : Float
            The logarithmic power-law slope of the inner core profiles
        alpha :
            Controls the sharpness of the transition between the inner core / outer Sersic profiles.
        """
        super(EllipticalCoreSersic, self).__init__(
            centre=centre,
            elliptical_comps=elliptical_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            mass_to_light_ratio=mass_to_light_ratio,
        )
        self.radius_break = radius_break
        self.intensity_break = intensity_break
        self.alpha = alpha
        self.gamma = gamma

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

    def image_from_grid_radii(self, grid_radii):
        """Calculate the intensity of the cored-Sersic light profile on a grid of radial coordinates.

        Parameters
        ----------
        grid_radii : float
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

    def decompose_convergence_into_gaussians(self):
        radii_min = self.effective_radius / 50.0
        radii_max = self.effective_radius * 20.0

        def core_sersic_2D(r):
            return (
                self.mass_to_light_ratio
                * self.intensity
                * (
                    self.intensity_prime
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
            )

        return self._decompose_convergence_into_gaussians(
            func=core_sersic_2D, radii_min=radii_min, radii_max=radii_max
        )


class SphericalCoreSersic(EllipticalCoreSersic):
    def __init__(
        self,
        centre: typing.Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
        radius_break: float = 0.01,
        intensity_break: float = 0.05,
        gamma: float = 0.25,
        alpha: float = 3.0,
    ):
        """ The elliptical cored-Sersic light profile.

        Parameters
        ----------
        centre : (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        intensity : float
            Overall intensity normalisation of the light profiles (electrons per second).
        effective_radius : float
            The circular radius containing half the light of this profile.
        sersic_index : Int
            Controls the concentration of the of the profile (lower value -> less concentrated, \
            higher value -> more concentrated).
        radius_break : Float
            The break radius separating the inner power-law (with logarithmic slope gamma) and outer Sersic function.
        intensity_break : Float
            The intensity at the break radius.
        gamma : Float
            The logarithmic power-law slope of the inner core profiles
        alpha :
            Controls the sharpness of the transition between the inner core / outer Sersic profiles.
        """
        super(SphericalCoreSersic, self).__init__(
            centre=centre,
            elliptical_comps=(0.0, 0.0),
            intensity=intensity,
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


class EllipticalChameleon(mp.EllipticalMassProfile, StellarProfile):
    def __init__(
        self,
        centre: typing.Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        core_radius_0: float = 0.01,
        core_radius_1: float = 0.02,
        mass_to_light_ratio: float = 1.0,
    ):
        """ The elliptical Chamelon mass profile.

        Parameters
        ----------
        centre : (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps : (float, float)
            The first and second ellipticity components of the elliptical coordinate system, where
            fac = (1 - axis_ratio) / (1 + axis_ratio), ellip_y = fac * sin(2*phi) and ellip_x = fac * cos(2*phi).
        intensity : float
            Overall intensity normalisation of the light profiles (electrons per second).
        core_radius_0 : the core size of the first elliptical cored Isothermal profile.
        core_radius_1 : core_radius_0 + core_radius_1 is the core size of the second elliptical cored Isothermal profile.
            We use core_radius_1 here is to avoid negative values.

        Profile form:
            mass_to_light_ratio * intensity *\
                (1.0 / Sqrt(x^2 + (y/q)^2 + core_radius_0^2) - 1.0 / Sqrt(x^2 + (y/q)^2 + (core_radius_0 + core_radius_1)**2.0))
        """

        super(EllipticalChameleon, self).__init__(
            centre=centre, elliptical_comps=elliptical_comps
        )
        super(mp.EllipticalMassProfile, self).__init__(
            centre=centre, elliptical_comps=elliptical_comps
        )
        self.mass_to_light_ratio = mass_to_light_ratio
        self.intensity = intensity
        self.core_radius_0 = core_radius_0
        self.core_radius_1 = core_radius_1
        if self.axis_ratio > 0.99999:
            self.axis_ratio = 0.99999

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def deflections_from_grid(self, grid):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.
        Following Eq. (15) and (16), but the parameters are slightly different.

        Parameters
        ----------
        grid : aa.Grid2D
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

        return self.rotate_grid_from_profile(
            np.multiply(factor, np.vstack((deflection_y, deflection_x)).T)
        )

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def convergence_from_grid(self, grid):
        """Calculate the projected convergence at a given set of arc-second gridded coordinates.
        Parameters
        ----------
        grid : aa.Grid2D
            The grid of (y,x) arc-second coordinates the convergence is computed on.
        """
        return self.convergence_func(self.grid_to_elliptical_radii(grid))

    def convergence_func(self, grid_radius):
        return self.mass_to_light_ratio * self.image_from_grid_radii(grid_radius)

    @grids.grid_like_to_structure
    def potential_from_grid(self, grid):
        return np.zeros(shape=grid.shape[0])

    def image_from_grid_radii(self, grid_radii):
        """Calculate the intensity of the Chamelon light profile on a grid of radial coordinates.

        Parameters
        ----------
        grid_radii : float
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

    def with_new_normalization(self, normalization):

        mass_profile = copy.copy(self)
        mass_profile.mass_to_light_ratio = normalization
        return mass_profile


class SphericalChameleon(EllipticalChameleon):
    def __init__(
        self,
        centre: typing.Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        core_radius_0: float = 0.01,
        core_radius_1: float = 0.02,
        mass_to_light_ratio: float = 1.0,
    ):
        """ The spherica; Chameleon mass profile.

        Profile form:
            mass_to_light_ratio * intensity *\
                (1.0 / Sqrt(x^2 + (y/q)^2 + core_radius_0^2) - 1.0 / Sqrt(x^2 + (y/q)^2 + (core_radius_0 + core_radius_1)**2.0))

        Parameters
        ----------
        centre : (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps : (float, float)
            The first and second ellipticity components of the elliptical coordinate system, where
            fac = (1 - axis_ratio) / (1 + axis_ratio), ellip_y = fac * sin(2*phi) and ellip_x = fac * cos(2*phi).
        intensity : float
            Overall intensity normalisation of the light profiles (electrons per second).
        core_radius_0 : the core size of the first elliptical cored Isothermal profile.
        core_radius_1 : core_radius_0 + core_radius_1 is the core size of the second elliptical cored Isothermal profile.
            We use core_radius_1 here is to avoid negative values.
       """

        super(SphericalChameleon, self).__init__(
            centre=centre,
            elliptical_comps=(0.0, 0.0),
            intensity=intensity,
            core_radius_0=core_radius_0,
            core_radius_1=core_radius_1,
            mass_to_light_ratio=mass_to_light_ratio,
        )
