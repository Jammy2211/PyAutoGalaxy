import warnings

import autofit as af
import numpy as np
from autoarray.structures import arrays
from autoarray.structures import grids
from autogalaxy import dimensions as dim
from autogalaxy.profiles import mass_profiles as mp
from pyquad import quad_grid
from scipy.special import wofz
import typing


class StellarProfile:

    pass


class EllipticalGaussian(mp.EllipticalMassProfile, StellarProfile):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0),
        intensity: dim.Luminosity = 0.1,
        sigma: dim.Length = 0.01,
        mass_to_light_ratio: dim.MassOverLuminosity = 1.0,
    ):
        """ The elliptical Gaussian light profile.

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

    def omega_from_grid_and_q(self, grid_complex, q):
        x = np.real(grid_complex)
        y = np.imag(grid_complex)
        faddeeva = wofz((q * x) + (1j * y / q))
        return (
            np.exp(-x ** 2.0 - 2.0 * 1j * x * y) * np.exp(y ** 2.0)
            - (
                1j
                * np.exp(
                    -x ** 2.0 * (1.0 - q ** 2.0) - y ** 2.0 * ((1.0 / q ** 2.0) - 1.0)
                )
            )
            * faddeeva
        )

    def sigma_from_grid(self, grid):
        input_grid = (self.axis_ratio * (grid[:, 1] + 1j * grid[:, 0])) / (
            self.sigma * np.sqrt(2 * (1.0 - self.axis_ratio ** 2.0))
        )
        return self.omega_from_grid_and_q(
            grid_complex=input_grid, q=1
        ) - self.omega_from_grid_and_q(grid_complex=input_grid, q=self.axis_ratio)

    def deflections_from_grid(self, grid):

        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                return self.deflections_from_grid_via_analytic(grid=grid)
            except RuntimeWarning:
                return self.deflections_from_grid_via_integrator(grid=grid)

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def deflections_from_grid_via_analytic(self, grid):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : aa.Grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """

        deflections = (
            self.mass_to_light_ratio
            * self.intensity
            * (1.0 / self.sigma * np.sqrt(2.0 * np.pi))
            * self.sigma
            * np.sqrt((2 * np.pi) / (1.0 - self.axis_ratio ** 2.0))
            * self.sigma_from_grid(grid=grid)
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
        grid : aa.Grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """

        def calculate_deflection_component(npow, index):

            deflection_grid = self.axis_ratio * grid[:, index]
            deflection_grid *= (
                (1.0 / self.sigma * np.sqrt(2.0 * np.pi))
                * self.intensity
                * self.mass_to_light_ratio
                * quad_grid(
                    self.deflection_func,
                    0.0,
                    1.0,
                    grid,
                    args=(npow, self.axis_ratio, self.sigma),
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
        """ Calculate the projected convergence at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : aa.Grid
            The grid of (y,x) arc-second coordinates the convergence is computed on.

        """
        return self.convergence_func(self.grid_to_eccentric_radii(grid))

    def convergence_func(self, grid_radius):
        return self.mass_to_light_ratio * self.intensity_at_radius(grid_radius)

    def intensity_at_radius(self, grid_radii):
        """Calculate the intensity of the Gaussian light profile on a grid of radial coordinates.

        Parameters
        ----------
        grid_radii : float
            The radial distance from the centre of the profile. for each coordinate on the grid.
        """
        return np.multiply(
            np.divide(self.intensity, self.sigma * np.sqrt(2.0 * np.pi)),
            np.exp(-0.5 * np.square(np.divide(grid_radii, self.sigma))),
        )


# noinspection PyAbstractClass
class AbstractEllipticalSersic(mp.EllipticalMassProfile, StellarProfile):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0),
        intensity: dim.Luminosity = 0.1,
        effective_radius: dim.Length = 0.6,
        sersic_index: float = 0.6,
        mass_to_light_ratio: dim.MassOverLuminosity = 1.0,
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
        self.mass_to_light_ratio = mass_to_light_ratio
        self.intensity = intensity
        self.effective_radius = effective_radius
        self.sersic_index = sersic_index

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
        return self.convergence_func(self.grid_to_eccentric_radii(grid))

    def convergence_func(self, grid_radius):
        return self.mass_to_light_ratio * self.intensity_at_radius(grid_radius)

    @grids.grid_like_to_structure
    def potential_from_grid(self, grid):
        return arrays.Array.manual_1d(
            array=np.zeros(shape=grid.shape[0]), shape_2d=grid.sub_shape_2d
        )

    @property
    def ellipticity_rescale(self):
        return 1.0 - ((1.0 - self.axis_ratio) / 2.0)

    def intensity_at_radius(self, radius):
        """ Compute the intensity of the profile at a given radius.

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
        """ A parameter derived from Sersic index which ensures that effective radius contains 50% of the profile's
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

    @property
    def unit_mass(self):
        return self.mass_to_light_ratio.unit_mass


class EllipticalSersic(AbstractEllipticalSersic):
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


class SphericalSersic(EllipticalSersic):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        intensity: dim.Luminosity = 0.1,
        effective_radius: dim.Length = 0.6,
        sersic_index: float = 0.6,
        mass_to_light_ratio: dim.MassOverLuminosity = 1.0,
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
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0),
        intensity: dim.Luminosity = 0.1,
        effective_radius: dim.Length = 0.6,
        mass_to_light_ratio: dim.MassOverLuminosity = 1.0,
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
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        intensity: dim.Luminosity = 0.1,
        effective_radius: dim.Length = 0.6,
        mass_to_light_ratio: dim.MassOverLuminosity = 1.0,
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
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0),
        intensity: dim.Luminosity = 0.1,
        effective_radius: dim.Length = 0.6,
        mass_to_light_ratio: dim.MassOverLuminosity = 1.0,
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
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        intensity: dim.Luminosity = 0.1,
        effective_radius: dim.Length = 0.6,
        mass_to_light_ratio: dim.MassOverLuminosity = 1.0,
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
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0),
        intensity: dim.Luminosity = 0.1,
        effective_radius: dim.Length = 0.6,
        sersic_index: float = 0.6,
        mass_to_light_ratio: dim.MassOverLuminosity = 1.0,
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
        """ Calculate the projected convergence at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : aa.Grid
            The grid of (y,x) arc-second coordinates the convergence is computed on.

        """
        return self.convergence_func(self.grid_to_eccentric_radii(grid))

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
            * self.intensity_at_radius(grid_radius)
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


class SphericalSersicRadialGradient(EllipticalSersicRadialGradient):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        intensity: dim.Luminosity = 0.1,
        effective_radius: dim.Length = 0.6,
        sersic_index: float = 0.6,
        mass_to_light_ratio: dim.MassOverLuminosity = 1.0,
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
