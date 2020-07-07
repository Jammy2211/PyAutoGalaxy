import autofit as af
import numpy as np
from astropy import cosmology as cosmo
from autoarray.structures import grids
from autofit.text import formatter
from autogalaxy import dimensions as dim
from autogalaxy.profiles import geometry_profiles
from autogalaxy.util import cosmology_util
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
        raise NotImplementedError("intensity_at_radius should be overridden")

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
        intensity : ndarray
            The value of intensity at the given radius
        """
        raise NotImplementedError("intensity_from_grid should be overridden")

    def luminosity_within_circle_in_units(
        self,
        radius: dim.Length,
        unit_luminosity="eps",
        exposure_time=None,
        redshift_profile=None,
        cosmology=cosmo.Planck15,
        **kwargs
    ):
        raise NotImplementedError()

    def summarize_in_units(
        self,
        radii,
        unit_length="arcsec",
        unit_luminosity="eps",
        exposure_time=None,
        redshift_profile=None,
        cosmology=cosmo.Planck15,
        **kwargs
    ):
        return ["Light Profile = {}\n".format(self.__class__.__name__)]


# noinspection PyAbstractClass
class EllipticalLightProfile(geometry_profiles.EllipticalProfile, LightProfile):
    """Generic class for an elliptical light profiles"""

    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0),
        intensity: dim.Luminosity = 0.1,
    ):
        """  Abstract class for an elliptical light-profile.

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
        return grids.GridCoordinates([self.centre])

    def blurred_image_from_grid_and_psf(self, grid, psf, blurring_grid):
        """Evaluate the light profile image on an input *Grid* of coordinates and then convolve it with a PSF.

        The *Grid* may be masked, in which case values outside but near the edge of the mask will convolve light into
        the mask. A blurring grid is therefore required, which evaluates the image on pixels on the mask edge such that
        their light is blurred into it by the PSF.

        The grid and blurring_grid must be a *Grid* objects so the evaluated image can be mapped to a uniform 2D array
        and binned up for convolution. They therefore cannot be *GridCoordinates* objects.

        Parameters
        ----------
        grid : Grid
            The (y, x) coordinates in the original reference frame of the grid.
        psf : aa.Kernel
            The PSF the evaluated light profile image is convolved with.
        blurring_grid : Grid
            The (y,x) coordinates neighboring the (masked) grid whose light is blurred into the image.

        """
        image = self.image_from_grid(grid=grid)

        blurring_image = self.image_from_grid(grid=blurring_grid)

        return psf.convolved_array_from_array_2d_and_mask(
            array_2d=image.in_2d_binned + blurring_image.in_2d_binned, mask=grid.mask
        )

    def blurred_image_from_grid_and_convolver(self, grid, convolver, blurring_grid):
        """Evaluate the light profile image on an input *Grid* of coordinates and then convolve it with a PSF using a
        *Convolver* object.

        The *Grid* may be masked, in which case values outside but near the edge of the mask will convolve light into
        the mask. A blurring grid is therefore required, which evaluates the image on pixels on the mask edge such that
        their light is blurred into it by the Convolver.

        The grid and blurring_grid must be a *Grid* objects so the evaluated image can be mapped to a uniform 2D array
        and binned up for convolution. They therefore cannot be *GridCoordinates* objects.

        Parameters
        ----------
        grid : Grid
            The (y, x) coordinates in the original reference frame of the grid.
        Convolver : aa.Convolver
            The Convolver object used to blur the PSF.
        blurring_grid : Grid
            The (y,x) coordinates neighboring the (masked) grid whose light is blurred into the image.

        """
        image = self.image_from_grid(grid=grid)

        blurring_image = self.image_from_grid(grid=blurring_grid)

        return convolver.convolved_image_from_image_and_blurring_image(
            image=image.in_1d_binned, blurring_image=blurring_image.in_1d_binned
        )

    def profile_visibilities_from_grid_and_transformer(self, grid, transformer):

        image = self.image_from_grid(grid=grid)

        return transformer.visibilities_from_image(image=image.in_1d_binned)

    def luminosity_within_circle_in_units(
        self,
        radius: dim.Length,
        unit_luminosity="eps",
        exposure_time=None,
        redshift_object=None,
        cosmology=cosmo.Planck15,
        **kwargs
    ):
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

        if not hasattr(radius, "unit_length"):
            radius = dim.Length(value=radius, unit_length="arcsec")

        if self.unit_length is not radius.unit_length:

            kpc_per_arcsec = cosmology_util.kpc_per_arcsec_from(
                redshift=redshift_object, cosmology=cosmology
            )

            radius = radius.convert(
                unit_length=self.unit_length, kpc_per_arcsec=kpc_per_arcsec
            )

        luminosity = dim.Luminosity(
            value=quad(self.luminosity_integral, a=0.0, b=radius)[0],
            unit_luminosity=self.unit_luminosity,
        )
        return luminosity.convert(
            unit_luminosity=unit_luminosity, exposure_time=exposure_time
        )

    def luminosity_integral(self, x):
        """Routine to integrate the luminosity of an elliptical light profile.

        The axis ratio is set to 1.0 for computing the luminosity within a circle"""
        return 2 * np.pi * x * self.image_from_grid_radii(x)

    def summarize_in_units(
        self,
        radii,
        prefix="",
        unit_length="arcsec",
        unit_luminosity="eps",
        exposure_time=None,
        redshift_profile=None,
        cosmology=cosmo.Planck15,
        whitespace=80,
        **kwargs
    ):
        summary = super().summarize_in_units(
            radii=radii,
            unit_length=unit_length,
            unit_luminosity=unit_luminosity,
            exposure_time=exposure_time,
            redshift_profile=redshift_profile,
            cosmology=cosmology,
            kwargs=kwargs,
        )

        for radius in radii:
            luminosity = self.luminosity_within_circle_in_units(
                unit_luminosity=unit_luminosity,
                radius=radius,
                redshift_object=redshift_profile,
                exposure_time=exposure_time,
                cosmology=cosmology,
                kwargs=kwargs,
            )

            summary += [
                formatter.within_radius_label_value_and_unit_string(
                    prefix=prefix + "luminosity",
                    radius=radius,
                    unit_length=unit_length,
                    value=luminosity,
                    unit_value=unit_luminosity,
                    whitespace=whitespace,
                )
            ]

        return summary


class EllipticalGaussian(EllipticalLightProfile):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0),
        intensity: dim.Luminosity = 0.1,
        sigma: dim.Length = 0.01,
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
        """
        return np.multiply(
            np.divide(self.intensity, self.sigma * np.sqrt(2.0 * np.pi)),
            np.exp(-0.5 * np.square(np.divide(grid_radii, self.sigma))),
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


class SphericalGaussian(EllipticalGaussian):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        intensity: dim.Luminosity = 0.1,
        sigma: dim.Length = 0.01,
    ):
        """ The spherical Gaussian light profile.

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
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0),
        intensity: dim.Luminosity = 0.1,
        effective_radius: dim.Length = 0.6,
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

    def new_profile_with_units_distance_converted(
        self, units_distance, kpc_per_arcsec=None
    ):
        self.units_distance = units_distance
        self.centre = self.centre.convert(
            unit_distance=units_distance, kpc_per_arcsec=kpc_per_arcsec
        )
        self.effective_radius = self.effective_radius.convert(
            unit_distance=units_distance, kpc_per_arcsec=kpc_per_arcsec
        )
        return self

    @property
    def elliptical_effective_radius(self):
        """The effective_radius of a Sersic light profile is defined as the circular effective radius. This is the \
        radius within which a circular aperture contains half the profiles's total integrated light. For elliptical \
        systems, this won't robustly capture the light profile's elliptical shape.

        The elliptical effective radius instead describes the major-axis radius of the ellipse containing \
        half the light, and may be more appropriate for highly flattened systems like disk galaxies."""
        return self.effective_radius / np.sqrt(self.axis_ratio)

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


class EllipticalSersic(AbstractEllipticalSersic, EllipticalLightProfile):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0),
        intensity: dim.Luminosity = 0.1,
        effective_radius: dim.Length = 0.6,
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
        """ Calculate the intensity of the light profile on a grid of Cartesian (y,x) coordinates.

        If the coordinates have not been transformed to the profile's geometry, this is performed automatically.

        Parameters
        ----------
        grid : grid_like
            The (y, x) coordinates in the original reference frame of the grid.
        """
        return self.image_from_grid_radii(self.grid_to_eccentric_radii(grid))


class SphericalSersic(EllipticalSersic):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        intensity: dim.Luminosity = 0.1,
        effective_radius: dim.Length = 0.6,
        sersic_index: float = 4.0,
    ):
        """ The spherical Sersic light profile.

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
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0),
        intensity: dim.Luminosity = 0.1,
        effective_radius: dim.Length = 0.6,
    ):
        """ The elliptical exponential profile.

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
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        intensity: dim.Luminosity = 0.1,
        effective_radius: dim.Length = 0.6,
    ):
        """ The spherical exponential profile.

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
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0),
        intensity: dim.Luminosity = 0.1,
        effective_radius: dim.Length = 0.6,
    ):
        """ The elliptical Dev Vaucouleurs light profile.

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
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        intensity: dim.Luminosity = 0.1,
        effective_radius: dim.Length = 0.6,
    ):
        """ The spherical Dev Vaucouleurs light profile.

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
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0),
        intensity: dim.Luminosity = 0.1,
        effective_radius: dim.Length = 0.6,
        sersic_index: float = 4.0,
        radius_break: dim.Length = 0.01,
        intensity_break: dim.Luminosity = 0.05,
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

    def new_profile_with_units_distance_converted(
        self, units_distance, kpc_per_arcsec=None
    ):
        self.units_distance = units_distance
        self.centre = self.centre.convert(
            unit_distance=units_distance, kpc_per_arcsec=kpc_per_arcsec
        )
        self.effective_radius = self.effective_radius.convert(
            unit_distance=units_distance, kpc_per_arcsec=kpc_per_arcsec
        )
        self.radius_break = self.radius_break.convert(
            unit_distance=units_distance, kpc_per_arcsec=kpc_per_arcsec
        )
        return self

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
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        intensity: dim.Luminosity = 0.1,
        effective_radius: dim.Length = 0.6,
        sersic_index: float = 4.0,
        radius_break: dim.Length = 0.01,
        intensity_break: dim.Luminosity = 0.05,
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
