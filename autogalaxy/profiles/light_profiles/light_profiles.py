import numpy as np
from scipy.integrate import quad
from typing import Tuple, Union

import autoarray as aa

from autogalaxy.operate.image import OperateImage
from autogalaxy.profiles.geometry_profiles import EllProfile


class LightProfile(EllProfile, OperateImage):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
    ):
        """
        Abstract base class for an elliptical light-profile.

        Each light profile has an analytic equation associated with it that describes its 1D surface brightness.

        Given an input grid of 1D or 2D (y,x) coordinates the light profile can be used to evaluate its surface
        brightness in 1D or as a 2D image.

        Associated with a light profile is a spherical or elliptical geometry, which describes its `centre` of
        emission and ellipticity. Geometric transformations are performed by decorators linked to the **PyAutoArray**
        `geometry` package.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps
            The first and second ellipticity components of the elliptical coordinate system (see the module
            `autogalaxy -> convert.py` for the convention).
        """
        super().__init__(centre=centre, elliptical_comps=elliptical_comps)
        self.intensity = intensity

    def image_2d_from(self, grid: aa.type.Grid2DLike) -> aa.Array2D:
        """
        Returns the light profile's 2D image from a 2D grid of Cartesian (y,x) coordinates, which may have been
        transformed using the light profile's geometry.

        If the coordinates have not been transformed to the profile's geometry, this is performed automatically.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates in the original reference frame of the grid.

        Returns
        -------
        image
            The image of the `LightProfile` evaluated at every (y,x) coordinate on the transformed grid.
        """
        raise NotImplementedError()

    def image_2d_via_radii_from(self, grid_radii: np.ndarray) -> np.ndarray:
        """
        Returns the light profile's 2D image from a 1D grid of coordinates which are the radial distance of each
        coordinate from the light profile `centre`.

        Parameters
        ----------
        grid_radii
            The radial distances from the centre of the profile, for each coordinate on the grid.
        """
        raise NotImplementedError()

    @aa.grid_dec.grid_1d_to_structure
    def image_1d_from(self, grid: aa.type.Grid1D2DLike) -> aa.type.Grid1D2DLike:
        """
        Returns the light profile's 1D image from a grid of Cartesian coordinates, which may have been
        transformed using the light profile's geometry.

        If a 1D grid is input the image is evaluated every coordinate on the grid. If a 2D grid is input, this is
        converted to a 1D grid by aligning with the major-axis of the light profile's elliptical geometry.

        Internally, this function uses a 2D grid to compute the image, which is mapped to a 1D data structure on return
        via the `grid_1d_to_structure` decorator. This avoids code repetition by ensuring that light profiles only use
        their `image_2d_from()`  function to evaluate their image.

        Parameters
        ----------
        grid
            A 1D or 2D grid of coordinates which are used to evaluate the light profile in 1D.

        Returns
        -------
        image
            The 1D image of the light profile evaluated at every (x,) coordinate on the 1D transformed grid.
        """
        return self.image_2d_from(grid=grid)

    def luminosity_within_circle_from(self, radius: float) -> float:
        """
        Integrate the light profile to compute the total luminosity within a circle of specified radius. This is
        centred on the light profile's `centre`.

        The `intensity` of a light profile is in dimension units, which are given physical meaning when the light
        profile is compared to data with physical units. The luminosity output by this function therefore is also
        dimensionless until compared to data.

        Parameters
        ----------
        radius
            The radius of the circle to compute the dimensionless luminosity within.
        """

        return quad(func=self.luminosity_integral, a=0.0, b=radius)[0]

    def luminosity_integral(self, x: np.ndarray) -> np.ndarray:
        """
        Routine to integrate the luminosity of an elliptical light profile.

        The axis ratio is set to 1.0 for computing the luminosity within a circle

        Parameters
        ----------
        x
            The 1D (x) radial coordinates where the luminosity integral is evaluated.
        """
        return 2 * np.pi * x * self.image_2d_via_radii_from(x)

    @property
    def half_light_radius(self) -> float:

        if hasattr(self, "effective_radius"):
            return self.effective_radius


class EllGaussian(LightProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        sigma: float = 0.01,
    ):
        """
        The elliptical Gaussian light profile.

        See `autogalaxy.profiles.light_profiles.light_profiles.LightProfile` for a description of light profile objects.

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
            The sigma value of the Gaussian, corresponding to ~ 1 / sqrt(2 log(2)) the full width half maximum.
        """

        super().__init__(
            centre=centre, elliptical_comps=elliptical_comps, intensity=intensity
        )
        self.sigma = sigma

    def image_2d_via_radii_from(self, grid_radii: np.ndarray) -> np.ndarray:
        """
        Returns the 2D image of the Gaussian light profile from a grid of coordinates which are the radial distance of
        each coordinate from the its `centre`.

        Note: sigma is divided by sqrt(q) here.

        Parameters
        ----------
        grid_radii
            The radial distances from the centre of the profile, for each coordinate on the grid.
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

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def image_2d_from(self, grid: aa.type.Grid2DLike) -> np.ndarray:
        """
        Returns the Gaussian light profile's 2D image from a 2D grid of Cartesian (y,x) coordinates.

        If the coordinates have not been transformed to the profile's geometry, this is performed automatically.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates in the original reference frame of the grid.

        Returns
        -------
        image
            The image of the Gaussian evaluated at every (y,x) coordinate on the transformed grid.
        """

        return self.image_2d_via_radii_from(self.grid_to_eccentric_radii(grid))


class SphGaussian(EllGaussian):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        sigma: float = 0.01,
    ):
        """
        The spherical Gaussian light profile.

        See `autogalaxy.profiles.light_profiles.light_profiles.LightProfile` for a description of light profile objects.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        sigma
            The sigma value of the Gaussian, corresponding to ~ 1 / sqrt(2 log(2)) the full width half maximum.
        """
        super().__init__(
            centre=centre, elliptical_comps=(0.0, 0.0), intensity=intensity, sigma=sigma
        )


class AbstractEllSersic(LightProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
    ):
        """
        Abstract base class for elliptical Sersic light profiles.

        See `autogalaxy.profiles.light_profiles.light_profiles.LightProfile` for a description of light profile objects.

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
            The circular radius containing half the light of this light profile.
        sersic_index
            Controls the concentration of the profile (lower -> less concentrated, higher -> more concentrated).
        """
        super().__init__(
            centre=centre, elliptical_comps=elliptical_comps, intensity=intensity
        )
        self.effective_radius = effective_radius
        self.sersic_index = sersic_index

    @property
    def elliptical_effective_radius(self) -> float:
        """
        The `effective_radius` of a Sersic light profile is defined as the circular effective radius, which is the
        radius within which a circular aperture contains half the profile's total integrated light.

        For elliptical systems, this will not robustly capture the light profile's elliptical shape.

        The elliptical effective radius instead describes the major-axis radius of the ellipse containing
        half the light, and may be more appropriate for highly flattened systems like disk galaxies.
        """
        return self.effective_radius / np.sqrt(self.axis_ratio)

    @property
    def sersic_constant(self) -> float:
        """
        A parameter derived from Sersic index which ensures that effective radius contains 50% of the profile's
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

    def image_2d_via_radii_from(self, radius: np.ndarray) -> np.ndarray:
        """
        Returns the 2D image of the Sersic light profile from a grid of coordinates which are the radial distances of
        each coordinate from the its `centre`.

        Parameters
        ----------
        grid_radii
            The radial distances from the centre of the profile, for each coordinate on the grid.
        """
        return self.intensity * np.exp(
            -self.sersic_constant
            * (((radius / self.effective_radius) ** (1.0 / self.sersic_index)) - 1)
        )


class EllSersic(AbstractEllSersic, LightProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
    ):
        """
        The elliptical Sersic light profile.

        See `autogalaxy.profiles.light_profiles.light_profiles.LightProfile` for a description of light profile objects.

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
        """
        super().__init__(
            centre=centre,
            elliptical_comps=elliptical_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
        )

    def image_2d_via_radii_from(self, grid_radii: np.ndarray) -> np.ndarray:
        """
        Returns the 2D image of the Sersic light profile from a grid of coordinates which are the radial distances of
        each coordinate from the its `centre`.

        Parameters
        ----------
        grid_radii
            The radial distances from the centre of the profile, for each coordinate on the grid.
        """
        np.seterr(all="ignore")
        return np.multiply(
            self.intensity,
            np.exp(
                np.multiply(
                    -self.sersic_constant,
                    np.add(
                        np.power(
                            np.divide(grid_radii, self.effective_radius),
                            1.0 / self.sersic_index,
                        ),
                        -1,
                    ),
                )
            ),
        )

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def image_2d_from(self, grid: aa.type.Grid2DLike) -> aa.Array2D:
        """
        Returns the Sersic light profile's 2D image from a 2D grid of Cartesian (y,x) coordinates.

        If the coordinates have not been transformed to the profile's geometry, this is performed automatically.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates in the original reference frame of the grid.

        Returns
        -------
        image
            The image of the Sersic evaluated at every (y,x) coordinate on the transformed grid.
        """
        return self.image_2d_via_radii_from(self.grid_to_eccentric_radii(grid))


class SphSersic(EllSersic):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
    ):
        """
        The spherical Sersic light profile.

        See `autogalaxy.profiles.light_profiles.light_profiles.LightProfile` for a description of light profile objects.

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
            Controls the concentration of the of the light profile.
        """
        super().__init__(
            centre=centre,
            elliptical_comps=(0.0, 0.0),
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
        )


class EllExponential(EllSersic):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
    ):
        """
        The elliptical exponential profile.

        This is a specific case of the elliptical Sersic profile where `sersic_index=1.0`.

        See `autogalaxy.profiles.light_profiles.light_profiles.LightProfile` for a description of light profile objects.

        Parameters
        ----------
        centre
            The (y,x) arc-second centre of the light profile.
        elliptical_comps
            The first and second ellipticity components of the elliptical coordinate system, (see the module
            `autogalaxy -> convert.py` for the convention).
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        effective_radius
            The circular radius containing half the light of this profile.
        """
        super().__init__(
            centre=centre,
            elliptical_comps=elliptical_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=1.0,
        )


class SphExponential(EllExponential):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
    ):
        """
        The spherical exponential profile.

        This is a specific case of the elliptical Sersic profile where `sersic_index=1.0`.

        See `autogalaxy.profiles.light_profiles.light_profiles.LightProfile` for a description of light profile objects.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        effective_radius
            The circular radius containing half the light of this profile.
        """
        super().__init__(
            centre=centre,
            elliptical_comps=(0.0, 0.0),
            intensity=intensity,
            effective_radius=effective_radius,
        )


class EllDevVaucouleurs(EllSersic):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
    ):
        """
        The elliptical Dev Vaucouleurs light profile.

        This is a specific case of the elliptical Sersic profile where `sersic_index=4.0`.

        See `autogalaxy.profiles.light_profiles.light_profiles.LightProfile` for a description of light profile objects.

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
        """
        super().__init__(
            centre=centre,
            elliptical_comps=elliptical_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=4.0,
        )


class SphDevVaucouleurs(EllDevVaucouleurs):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
    ):
        """
        The spherical Dev Vaucouleurs light profile.

        This is a specific case of the elliptical Sersic profile where `sersic_index=4.0`.

        See `autogalaxy.profiles.light_profiles.light_profiles.LightProfile` for a description of light profile objects.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        effective_radius
            The circular radius containing half the light of this profile.
        """
        super().__init__(
            centre=centre,
            elliptical_comps=(0.0, 0.0),
            intensity=intensity,
            effective_radius=effective_radius,
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
    ):
        """
        The elliptical cored-Sersic light profile.

        See `autogalaxy.profiles.light_profiles.light_profiles.LightProfile` for a description of light profile objects.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps
            The first and second ellipticity components of the elliptical coordinate system, (see the module
            `autogalaxy -> convert.py` for the convention).
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
        alpha
            Controls the sharpness of the transition between the inner core / outer Sersic profiles.
        """

        super().__init__(
            centre=centre,
            elliptical_comps=elliptical_comps,
            intensity=intensity_break,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
        )

        self.radius_break = radius_break
        self.intensity_break = intensity_break
        self.alpha = alpha
        self.gamma = gamma

    @property
    def intensity_prime(self) -> float:
        """
        Overall intensity normalisation in the rescaled cored Sersic light profile.

        Like the `intensity` parameter, the units of `intensity_prime` are dimensionless and derived from the data
        the light profile's image is compared too, which are expected to be electrons per second.
        """
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

    def image_2d_via_radii_from(self, grid_radii: np.ndarray) -> np.ndarray:
        """
        Returns the 2D image of the Sersic light profile from a grid of coordinates which are the radial distances of
        each coordinate from the its `centre`.

        Parameters
        ----------
        grid_radii
            The radial distances from the centre of the profile, for each coordinate on the grid.
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

        See `autogalaxy.profiles.light_profiles.light_profiles.LightProfile` for a description of light profile objects.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
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


class EllExponentialCore(EllSersicCore):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        effective_radius: float = 0.6,
        radius_break: float = 0.01,
        intensity_break: float = 0.05,
        gamma: float = 0.25,
        alpha: float = 3.0,
    ):
        """
        The elliptical cored-Exponential light profile.

        See `autogalaxy.profiles.light_profiles.light_profiles.LightProfile` for a description of light profile objects.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps
            The first and second ellipticity components of the elliptical coordinate system, (see the module
            `autogalaxy -> convert.py` for the convention).
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
            intensity_break=intensity_break,
            effective_radius=effective_radius,
            sersic_index=1.0,
            radius_break=radius_break,
            gamma=gamma,
            alpha=alpha,
        )


class SphExponentialCore(EllExponentialCore):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        effective_radius: float = 0.6,
        radius_break: float = 0.01,
        intensity_break: float = 0.05,
        gamma: float = 0.25,
        alpha: float = 3.0,
    ):
        """
        The elliptical cored-Exponential light profile.

        See `autogalaxy.profiles.light_profiles.light_profiles.LightProfile` for a description of light profile objects.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        effective_radius
            The circular radius containing half the light of this profile.
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
            radius_break=radius_break,
            intensity_break=intensity_break,
            gamma=gamma,
            alpha=alpha,
        )

        self.radius_break = radius_break
        self.intensity_break = intensity_break
        self.alpha = alpha
        self.gamma = gamma


class EllChameleon(LightProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        core_radius_0: float = 0.01,
        core_radius_1: float = 0.05,
    ):
        """
        The elliptical Chameleon light profile.

        This light profile closely approximes the Elliptical Sersic light profile, by representing it as two cored
        elliptical isothermal profiles. This is convenient for lensing calculations, because the deflection angles of
        an isothermal profile can be evaluated analyticially efficiently.

        See `autogalaxy.profiles.light_profiles.light_profiles.LightProfile` for a description of light profile objects.

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
        core_radius_0
            The core size of the first elliptical cored Isothermal profile.
        core_radius_1
            The core size of the second elliptical cored Isothermal profile.
        """

        super().__init__(
            centre=centre, elliptical_comps=elliptical_comps, intensity=intensity
        )
        self.core_radius_0 = core_radius_0
        self.core_radius_1 = core_radius_1

    @property
    def axis_ratio(self) -> float:
        """
        The elliptical isothermal mass profile deflection angles break down for perfectly spherical systems where
        `axis_ratio=1.0`, thus we remove these solutions.
        """
        axis_ratio = super().axis_ratio
        return axis_ratio if axis_ratio < 0.99999 else 0.99999

    def image_2d_via_radii_from(self, grid_radii: np.ndarray) -> np.ndarray:
        """
        Returns the 2D image of the Sersic light profile from a grid of coordinates which are the radial distances of
        each coordinate from the its `centre`.

        Parameters
        ----------
        grid_radii
            The radial distances from the centre of the profile, for each coordinate on the grid.
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

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def image_2d_from(self, grid: aa.type.Grid2DLike) -> np.ndarray:
        """
        Returns the Chameleon light profile's 2D image from a 2D grid of Cartesian (y,x) coordinates.

        If the coordinates have not been transformed to the profile's geometry, this is performed automatically.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates in the original reference frame of the grid.

        Returns
        -------
        image
            The image of the Chameleon evaluated at every (y,x) coordinate on the transformed grid.
        """
        return self.image_2d_via_radii_from(self.grid_to_elliptical_radii(grid))


class SphChameleon(EllChameleon):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        core_radius_0: float = 0.01,
        core_radius_1: float = 0.05,
    ):
        """
        The spherical Chameleon light profile.

        This light profile closely approximes the Elliptical Sersic light profile, by representing it as two cored
        elliptical isothermal profiles. This is convenient for lensing calculations, because the deflection angles of
        an isothermal profile can be evaluated analyticially efficiently.

        See `autogalaxy.profiles.light_profiles.light_profiles.LightProfile` for a description of light profile objects.

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
        core_radius_0
            The core size of the first elliptical cored Isothermal profile.
        core_radius_1
            The core size of the second elliptical cored Isothermal profile.
        """

        super().__init__(
            centre=centre,
            elliptical_comps=(0.0, 0.0),
            intensity=intensity,
            core_radius_0=core_radius_0,
            core_radius_1=core_radius_1,
        )


class EllEff(LightProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        eta: float = 1.5,
    ):
        """
        The elliptical Elson, Fall and Freeman (EFF) light profile, which is commonly used to represent the clumps of
        Lyman-alpha emitter galaxies (see https://arxiv.org/abs/1708.08854).

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
        eta
            Scales the intensity gradient of the profile.
        """

        super().__init__(
            centre=centre, elliptical_comps=elliptical_comps, intensity=intensity
        )

        self.effective_radius = effective_radius
        self.eta = eta

    def image_2d_via_radii_from(self, grid_radii: np.ndarray) -> np.ndarray:
        """
        Returns the 2D image of the Sersic light profile from a grid of coordinates which are the radial distances of
        each coordinate from the its `centre`.

        Parameters
        ----------
        grid_radii
            The radial distances from the centre of the profile, for each coordinate on the grid.
        """
        np.seterr(all="ignore")
        return self.intensity * (1 + (grid_radii / self.effective_radius) ** 2) ** (
            -self.eta
        )

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def image_2d_from(self, grid: aa.type.Grid2DLike) -> np.ndarray:
        """
        Returns the Eff light profile's 2D image from a 2D grid of Cartesian (y,x) coordinates.

        If the coordinates have not been transformed to the profile's geometry, this is performed automatically.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates in the original reference frame of the grid.

        Returns
        -------
        image
            The image of the Eff evaluated at every (y,x) coordinate on the transformed grid.
        """
        return self.image_2d_via_radii_from(self.grid_to_eccentric_radii(grid))

    @property
    def half_light_radius(self) -> float:
        return self.effective_radius * np.sqrt(0.5 ** (1.0 / (1.0 - self.eta)) - 1.0)


class SphEff(EllEff):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        eta: float = 1.5,
    ):
        """
        The spherical Elson, Fall and Freeman (EFF) light profile, which is commonly used to represent the clumps of
        Lyman-alpha emitter galaxies (see https://arxiv.org/abs/1708.08854).

        See `autogalaxy.profiles.light_profiles.light_profiles.LightProfile` for a description of light profile objects.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        effective_radius
            The circular radius containing half the light of this profile.
        eta
            Scales the intensity gradient of the profile.
        """

        super().__init__(
            centre=centre,
            elliptical_comps=(0.0, 0.0),
            intensity=intensity,
            effective_radius=effective_radius,
            eta=eta,
        )
