import numpy as np
from autoarray.structures import grids
from autogalaxy.profiles import geometry_profiles
from scipy.integrate import quad
import typing


class LightProfile:
    """Mixin class that implements functions common to all light profiles"""

    def image_from_grid_radii(self, grid_radii):
        """
        Abstract method for obtaining intensity at on a grid of radii.

        Parameters
        ----------
        grid_radii : float
            The radial distance from the centre of the profile. for each coordinate on the grid.
        """
        raise NotImplementedError("image_from_grid_radii should be overridden")

    # noinspection PyMethodMayBeStatic
    def image_from_grid(self, grid, grid_radial_minimum=None):
        """
        Abstract method for obtaining intensity at a grid of Cartesian (y,x) coordinates.

        Parameters
        ----------
        grid : grid_like
            The (y, x) coordinates in the original reference frame of the grid.
        Returns
        -------
        intensity : np.ndarray
            The value of intensity at the given radius
        """
        raise NotImplementedError("image_from_grid should be overridden")

    def luminosity_within_circle(self, radius: float):
        raise NotImplementedError()


# noinspection PyAbstractClass
class EllipticalLightProfile(geometry_profiles.EllipticalProfile, LightProfile):
    """Generic class for an elliptical light profiles"""

    def __init__(
        self,
        centre: typing.Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
    ):
        """Abstract class for an elliptical light-profile.

        Parameters
        ----------
        centre : (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps : (float, float)
            The first and second ellipticity components of the elliptical coordinate system, where
            fac = (1 - axis_ratio) / (1 + axis_ratio), ellip_y = fac * sin(2*phi) and ellip_x = fac * cos(2*phi).
        """
        super(EllipticalLightProfile, self).__init__(
            centre=centre, elliptical_comps=elliptical_comps
        )
        self.intensity = intensity

    @property
    def light_profile_centres(self):
        return grids.Grid2DIrregularGrouped([self.centre])

    def blurred_image_from_grid_and_psf(self, grid, psf, blurring_grid):
        """Evaluate the light profile image on an input `Grid2D` of coordinates and then convolve it with a PSF.

        The `Grid2D` may be masked, in which case values outside but near the edge of the mask will convolve light into
        the mask. A blurring grid is therefore required, which evaluates the image on pixels on the mask edge such that
        their light is blurred into it by the PSF.

        The grid and blurring_grid must be a `Grid2D` objects so the evaluated image can be mapped to a uniform 2D array
        and binned up for convolution. They therefore cannot be `Grid2DIrregularGrouped` objects.

        Parameters
        ----------
        grid : Grid2D
            The (y, x) coordinates in the original reference frame of the grid.
        psf : aa.Kernel2D
            The PSF the evaluated light profile image is convolved with.
        blurring_grid : Grid2D
            The (y,x) coordinates neighboring the (masked) grid whose light is blurred into the image.

        """
        image = self.image_from_grid(grid=grid)

        blurring_image = self.image_from_grid(grid=blurring_grid)

        return psf.convolved_array_from_array_2d_and_mask(
            array_2d=image.native_binned + blurring_image.native_binned, mask=grid.mask
        )

    def blurred_image_from_grid_and_convolver(self, grid, convolver, blurring_grid):
        """Evaluate the light profile image on an input `Grid2D` of coordinates and then convolve it with a PSF using a
        *Convolver* object.

        The `Grid2D` may be masked, in which case values outside but near the edge of the mask will convolve light into
        the mask. A blurring grid is therefore required, which evaluates the image on pixels on the mask edge such that
        their light is blurred into it by the Convolver.

        The grid and blurring_grid must be a `Grid2D` objects so the evaluated image can be mapped to a uniform 2D array
        and binned up for convolution. They therefore cannot be `Grid2DIrregularGrouped` objects.

        Parameters
        ----------
        grid : Grid2D
            The (y, x) coordinates in the original reference frame of the grid.
        Convolver : aa.Convolver
            The Convolver object used to blur the PSF.
        blurring_grid : Grid2D
            The (y,x) coordinates neighboring the (masked) grid whose light is blurred into the image.

        """
        image = self.image_from_grid(grid=grid)

        blurring_image = self.image_from_grid(grid=blurring_grid)

        return convolver.convolved_image_from_image_and_blurring_image(
            image=image.slim_binned, blurring_image=blurring_image.slim_binned
        )

    def profile_visibilities_from_grid_and_transformer(self, grid, transformer):

        image = self.image_from_grid(grid=grid)

        return transformer.visibilities_from_image(image=image.slim_binned)

    def luminosity_within_circle(self, radius: float) -> float:
        """Integrate the light profile to compute the total luminosity within a circle of specified radius. This is \
        centred on the light profile's centre.

        The following unit_label for mass can be specified and output:

        - Electrons per second (default) - 'eps'.
        - Counts - 'counts' (multiplies the luminosity in electrons per second by the exposure time).

        Parameters
        ----------
        radius : float
            The radius of the circle to compute the dimensionless mass within.
        unit_luminosity : str
            The unit_label the luminosity is returned in {esp, counts}.
        exposure_time : float or None
            The exposure time of the observation, which converts luminosity from electrons per second unit_label to counts.
        """

        return quad(func=self.luminosity_integral, a=0.0, b=radius)[0]

    def luminosity_integral(self, x):
        """Routine to integrate the luminosity of an elliptical light profile.

        The axis ratio is set to 1.0 for computing the luminosity within a circle"""
        return 2 * np.pi * x * self.image_from_grid_radii(x)


class PointSource(EllipticalLightProfile):
    def __init__(self, centre: typing.Tuple[float, float] = (0.0, 0.0)):

        super().__init__(centre=centre, elliptical_comps=(0.0, 0.0))

    @grids.grid_like_to_structure
    def image_from_grid(self, grid):
        return np.zeros(shape=grid.shape[0])


class PointSourceFlux(PointSource):
    def __init__(
        self, centre: typing.Tuple[float, float] = (0.0, 0.0), flux: float = 0.1
    ):

        super().__init__(centre=centre)

        self.flux = flux


class EllipticalGaussian(EllipticalLightProfile):
    def __init__(
        self,
        centre: typing.Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        sigma: float = 0.01,
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
            The sigma value of the Gaussian, correspodning to ~ 1 / sqrt(2 log(2)) the full width half maximum.
        """

        super(EllipticalGaussian, self).__init__(
            centre=centre, elliptical_comps=elliptical_comps, intensity=intensity
        )
        self.sigma = sigma

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

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def image_from_grid(self, grid, grid_radial_minimum=None):
        """
        Calculate the intensity of the light profile on a grid of Cartesian (y,x) coordinates.

        If the coordinates have not been transformed to the profile's geometry, this is performed automatically.

        Parameters
        ----------
        grid : grid_like
            The (y, x) coordinates in the original reference frame of the grid.
        """

        return self.image_from_grid_radii(self.grid_to_eccentric_radii(grid))


class SphericalGaussian(EllipticalGaussian):
    def __init__(
        self,
        centre: typing.Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        sigma: float = 0.01,
    ):
        """The spherical Gaussian light profile.

        Parameters
        ----------
        centre : (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        intensity : float
            Overall intensity normalisation of the light profiles (electrons per second).
        sigma : float
            The sigma value of the Gaussian, correspodning to ~ 1 / sqrt(2 log(2)) the full width half maximum.
        """
        super(SphericalGaussian, self).__init__(
            centre=centre, elliptical_comps=(0.0, 0.0), intensity=intensity, sigma=sigma
        )


class AbstractEllipticalSersic(EllipticalLightProfile):
    def __init__(
        self,
        centre: typing.Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
    ):
        """ Abstract base class for an elliptical Sersic light profile, used for computing its effective radius and
        Sersic instance.

        Parameters
        ----------
        centre : (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps : (float, float)
            The first and second ellipticity components of the elliptical coordinate system, where
            fac = (1 - axis_ratio) / (1 + axis_ratio), ellip_y = fac * sin(2*phi) and ellip_x = fac * cos(2*phi).
        intensity : float
            Overall intensity normalisation in the light profiles (electrons per second)
        effective_radius : float
            The circular radius containing half the light of this model_mapper
        sersic_index : Int
            Controls the concentration of the of the profile (lower value -> less concentrated, \
            higher value -> more concentrated).
        """
        super(AbstractEllipticalSersic, self).__init__(
            centre=centre, elliptical_comps=elliptical_comps, intensity=intensity
        )
        self.effective_radius = effective_radius
        self.sersic_index = sersic_index

    @property
    def elliptical_effective_radius(self):
        """
        The effective_radius of a Sersic light profile is defined as the circular effective radius. This is the
        radius within which a circular aperture contains half the profiles's total integrated light. For elliptical
        systems, this won't robustly capture the light profile's elliptical shape.

        The elliptical effective radius instead describes the major-axis radius of the ellipse containing
        half the light, and may be more appropriate for highly flattened systems like disk galaxies.
        """
        return self.effective_radius / np.sqrt(self.axis_ratio)

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


class EllipticalSersic(AbstractEllipticalSersic, EllipticalLightProfile):
    def __init__(
        self,
        centre: typing.Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
    ):
        """ The elliptical Sersic light profile.

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
        """
        super(EllipticalSersic, self).__init__(
            centre=centre,
            elliptical_comps=elliptical_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
        )

    def image_from_grid_radii(self, grid_radii):
        """
        Calculate the intensity of the Sersic light profile on a grid of radial coordinates.

        Parameters
        ----------
        grid_radii : float
            The radial distance from the centre of the profile. for each coordinate on the grid.
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

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def image_from_grid(self, grid, grid_radial_minimum=None):
        """Calculate the intensity of the light profile on a grid of Cartesian (y,x) coordinates.

        If the coordinates have not been transformed to the profile's geometry, this is performed automatically.

        Parameters
        ----------
        grid : grid_like
            The (y, x) coordinates in the original reference frame of the grid.
        """
        return self.image_from_grid_radii(self.grid_to_eccentric_radii(grid))


class SphericalSersic(EllipticalSersic):
    def __init__(
        self,
        centre: typing.Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
    ):
        """The spherical Sersic light profile.

        Parameters
        ----------
        centre : (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        intensity : float
            Overall intensity normalisation of the light profiles (electrons per second).
        effective_radius : float
            The circular radius containing half the light of this profile.
        sersic_index : Int
            Controls the concentration of the of the light profile.
        """
        super(SphericalSersic, self).__init__(
            centre=centre,
            elliptical_comps=(0.0, 0.0),
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
        )


class EllipticalExponential(EllipticalSersic):
    def __init__(
        self,
        centre: typing.Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
    ):
        """The elliptical exponential profile.

        This is a subset of the elliptical Sersic profile, specific to the case that sersic_index = 1.0.

        Parameters
        ----------
        centre : (float, float)
            The (y,x) arc-second centre of the light profile.
        elliptical_comps : (float, float)
            The first and second ellipticity components of the elliptical coordinate system, where
            fac = (1 - axis_ratio) / (1 + axis_ratio), ellip_y = fac * sin(2*phi) and ellip_x = fac * cos(2*phi).
        intensity : float
            Overall intensity normalisation of the light profiles (electrons per second).
        effective_radius : float
            The circular radius containing half the light of this profile.
        """
        super(EllipticalExponential, self).__init__(
            centre=centre,
            elliptical_comps=elliptical_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=1.0,
        )


class SphericalExponential(EllipticalExponential):
    def __init__(
        self,
        centre: typing.Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
    ):
        """The spherical exponential profile.

        This is a subset of the elliptical Sersic profile, specific to the case that sersic_index = 1.0.

        Parameters
        ----------
        centre : (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        intensity : float
            Overall intensity normalisation of the light profiles (electrons per second).
        effective_radius : float
            The circular radius containing half the light of this profile.
        """
        super(SphericalExponential, self).__init__(
            centre=centre,
            elliptical_comps=(0.0, 0.0),
            intensity=intensity,
            effective_radius=effective_radius,
        )


class EllipticalDevVaucouleurs(EllipticalSersic):
    def __init__(
        self,
        centre: typing.Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
    ):
        """The elliptical Dev Vaucouleurs light profile.

        This is a subset of the elliptical Sersic profile, specific to the case that sersic_index = 4.0.

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
        """
        super(EllipticalDevVaucouleurs, self).__init__(
            centre=centre,
            elliptical_comps=elliptical_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=4.0,
        )


class SphericalDevVaucouleurs(EllipticalDevVaucouleurs):
    def __init__(
        self,
        centre: typing.Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
    ):
        """The spherical Dev Vaucouleurs light profile.

        This is a subset of the elliptical Sersic profile, specific to the case that sersic_index = 1.0.

        Parameters
        ----------
        centre : (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        intensity : float
            Overall intensity normalisation of the light profiles (electrons per second).
        effective_radius : float
            The circular radius containing half the light of this profile.
        """
        super(SphericalDevVaucouleurs, self).__init__(
            centre=centre,
            elliptical_comps=(0.0, 0.0),
            intensity=intensity,
            effective_radius=effective_radius,
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


class EllipticalChameleon(EllipticalLightProfile):
    def __init__(
        self,
        centre: typing.Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        core_radius_0: float = 0.01,
        core_radius_1: float = 0.05,
    ):
        """ The elliptical Chameleon light profile.

        Profile form:
            mass_to_light_ratio * intensity *\
                (1.0 / Sqrt(x^2 + (y/q)^2 + rc^2) - 1.0 / Sqrt(x^2 + (y/q)^2 + (rc + dr)**2.0))

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
        core_radius_1 : rc + dr is the core size of the second elliptical cored Isothermal profile.
             We use dr here is to avoid negative values.
        """

        super(EllipticalChameleon, self).__init__(
            centre=centre, elliptical_comps=elliptical_comps, intensity=intensity
        )
        self.core_radius_0 = core_radius_0
        self.core_radius_1 = core_radius_1
        if self.axis_ratio > 0.99999:
            self.axis_ratio = 0.99999

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

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def image_from_grid(self, grid, grid_radial_minimum=None):
        """
        Calculate the intensity of the light profile on a grid of Cartesian (y,x) coordinates.
        If the coordinates have not been transformed to the profile's geometry, this is performed automatically.
        Parameters
        ----------
        grid : grid_like
            The (y, x) coordinates in the original reference frame of the grid.
        """
        return self.image_from_grid_radii(self.grid_to_elliptical_radii(grid))


class SphericalChameleon(EllipticalChameleon):
    def __init__(
        self,
        centre: typing.Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        core_radius_0: float = 0.01,
        core_radius_1: float = 0.05,
    ):
        """ The spherical Chameleon light profile.

        Profile form:
            mass_to_light_ratio * intensity *\
                (1.0 / Sqrt(x^2 + (y/q)^2 + rc^2) - 1.0 / Sqrt(x^2 + (y/q)^2 + (rc + dr)**2.0))

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
        core_radius_1 : rc + dr is the core size of the second elliptical cored Isothermal profile.
             We use dr here is to avoid negative values.
        """

        super(SphericalChameleon, self).__init__(
            centre=centre,
            elliptical_comps=(0.0, 0.0),
            intensity=intensity,
            core_radius_0=core_radius_0,
            core_radius_1=core_radius_1,
        )
