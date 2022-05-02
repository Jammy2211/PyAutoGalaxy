import numpy as np
from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.light_profiles import light_profiles as lp
from scipy.optimize import root_scalar


class LightProfileSNR:
    def __init__(self, signal_to_noise_ratio: float = 10.0):
        """
        This light profile class sets the `intensity` of the light profile using input noise properties of a simulation
        (e.g. using the `exposure_time`, `background_sky_level`).

        This means that the intensities of the light profiles can be automatically adjusted when an `SimulatorImaging`
        object is used to simulate imaging data, whereby the intensity of each light profile is set to produce an
        image with the input `signal_to_noise_ratio` of this class.

        The brightest pixel of the image of the light profile is used to do this, thus the S/N in all other pixels
        away from the brightest pixel will be below the input `signal_to_noise_ratio`.

        The intensity is set using an input grid, meaning that for strong lensing calculations the ray-traced grid
        can be used such that the S/N accounts for the magnification of a source galaxy.

        Parameters
        ----------
        signal_to_noise_ratio
            The signal-to-noises ratio that the simulated light profile will produce.
        """
        self.signal_to_noise_ratio = signal_to_noise_ratio

    def image_2d_from(self, grid: aa.type.Grid2DLike) -> aa.Array2D:
        """
        Abstract method for obtaining intensity at a grid of Cartesian (y,x) coordinates.

        Parameters
        ----------
        grid
            The (y, x) coordinates in the original reference frame of the grid.

        Returns
        -------
        image
            The image of the `LightProfile` evaluated at every (y,x) coordinate on the grid.
        """
        raise NotImplementedError()

    def set_intensity_from(
        self,
        grid: aa.type.Grid2DLike,
        exposure_time: float,
        background_sky_level: float = 0.0,
    ):
        """
        Set the `intensity` of the light profile as follows:

        - Evaluate the image of the light profile on an input grid.
        - Take the value of the brightest pixel.
        - Use an input `exposure_time` and `background_sky` (e.g. from the `SimulatorImaging` object) to determine
        what value of `intensity` gives the desired signal to noise ratio for the image.

        The intensity is set using an input grid, meaning that for strong lensing calculations the ray-traced grid
        can be used such that the S/N accounts for the magnification of a source galaxy.


        Parameters
        ----------
        grid
            The (y, x) coordinates in the original reference frame of the grid.
        exposure_time
            The exposure time of the simulated imaging.
        background_sky_level
            The level of the background sky of the simulated imaging.
        """
        self.intensity = 1.0
        if hasattr(self, "intensity_break"):
            self.intensity_break = 1.0

        background_sky_level_counts = background_sky_level * exposure_time

        brightest_value = np.max(self.image_2d_from(grid=grid))

        def func(intensity_factor):

            signal = intensity_factor * brightest_value * exposure_time
            noise = np.sqrt(signal + background_sky_level_counts)

            return signal / noise - self.signal_to_noise_ratio

        intensity_factor = root_scalar(func, bracket=[1.0e-8, 1.0e8]).root

        self.intensity *= intensity_factor
        if hasattr(self, "intensity_break"):
            self.intensity_break *= intensity_factor


class EllGaussian(lp.EllGaussian, LightProfileSNR):
    def __init__(
        self,
        signal_to_noise_ratio: float = 10.0,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        sigma: float = 0.01,
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
            The sigma value of the Gaussian, corresponding to ~ 1 / sqrt(2 log(2)) the full width half maximum.
        """

        super().__init__(
            centre=centre, elliptical_comps=elliptical_comps, intensity=0.0, sigma=sigma
        )
        LightProfileSNR.__init__(self, signal_to_noise_ratio=signal_to_noise_ratio)


class SphGaussian(lp.SphGaussian, LightProfileSNR):
    def __init__(
        self,
        signal_to_noise_ratio: float = 10.0,
        centre: Tuple[float, float] = (0.0, 0.0),
        sigma: float = 0.01,
    ):
        """
        The spherical Gaussian light profile.

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
        super().__init__(centre=centre, intensity=0.0, sigma=sigma)
        LightProfileSNR.__init__(self, signal_to_noise_ratio=signal_to_noise_ratio)


class EllSersic(lp.EllSersic, LightProfileSNR):
    def __init__(
        self,
        signal_to_noise_ratio: float = 10.0,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
    ):
        """
        An elliptical Sersic light profile.

        Instead of an `intensity` a `signal_to_noise_ratio` is input which sets the signal to noise of the brightest
        pixel of the profile's image when used to simulate imaging data.

        Parameters
        ----------
        signal_to_noise_ratio
            The signal to noise of the light profile when it is used to simulate strong lens imaging.
        centre
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps
            The first and second ellipticity components of the elliptical coordinate system, (see the module
            `autogalaxy -> convert.py` for the convention).
        effective_radius
            The circular radius containing half the light of this profile.
        sersic_index
            Controls the concentration of the profile (lower -> less concentrated, higher -> more concentrated).
        """
        super().__init__(
            centre=centre,
            elliptical_comps=elliptical_comps,
            intensity=0.0,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
        )

        LightProfileSNR.__init__(self, signal_to_noise_ratio=signal_to_noise_ratio)


class SphSersic(lp.SphSersic, LightProfileSNR):
    def __init__(
        self,
        signal_to_noise_ratio: float = 10.0,
        centre: Tuple[float, float] = (0.0, 0.0),
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
    ):
        """
        The spherical Sersic light profile.

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
            intensity=0.0,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
        )
        LightProfileSNR.__init__(self, signal_to_noise_ratio=signal_to_noise_ratio)


class EllExponential(lp.EllExponential, LightProfileSNR):
    def __init__(
        self,
        signal_to_noise_ratio: float = 10.0,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        effective_radius: float = 0.6,
    ):
        """
        The elliptical exponential profile.

        This is a subset of the elliptical Sersic profile, specific to the case that sersic_index = 1.0.

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
            intensity=0.0,
            effective_radius=effective_radius,
        )
        LightProfileSNR.__init__(self, signal_to_noise_ratio=signal_to_noise_ratio)


class SphExponential(lp.SphExponential, LightProfileSNR):
    def __init__(
        self,
        signal_to_noise_ratio: float = 10.0,
        centre: Tuple[float, float] = (0.0, 0.0),
        effective_radius: float = 0.6,
    ):
        """
        The spherical exponential profile.

        This is a subset of the elliptical Sersic profile, specific to the case that sersic_index = 1.0.

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
        super().__init__(centre=centre, effective_radius=effective_radius)
        LightProfileSNR.__init__(self, signal_to_noise_ratio=signal_to_noise_ratio)


class EllDevVaucouleurs(lp.EllDevVaucouleurs, LightProfileSNR):
    def __init__(
        self,
        signal_to_noise_ratio: float = 10.0,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        effective_radius: float = 0.6,
    ):
        """
        The elliptical Dev Vaucouleurs light profile.

        This is a subset of the elliptical Sersic profile, specific to the case that sersic_index = 4.0.

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
            intensity=0.0,
            effective_radius=effective_radius,
        )
        LightProfileSNR.__init__(self, signal_to_noise_ratio=signal_to_noise_ratio)


class SphDevVaucouleurs(lp.SphDevVaucouleurs, LightProfileSNR):
    def __init__(
        self,
        signal_to_noise_ratio: float = 10.0,
        centre: Tuple[float, float] = (0.0, 0.0),
        effective_radius: float = 0.6,
    ):
        """
        The spherical Dev Vaucouleurs light profile.

        This is a subset of the elliptical Sersic profile, specific to the case that sersic_index = 1.0.

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
            centre=centre, intensity=0.0, effective_radius=effective_radius
        )
        LightProfileSNR.__init__(self, signal_to_noise_ratio=signal_to_noise_ratio)


class EllSersicCore(lp.EllSersicCore, LightProfileSNR):
    def __init__(
        self,
        signal_to_noise_ratio: float = 10.0,
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

        Instead of an `intensity` a `signal_to_noise_ratio` is input which sets the signal to noise of the brightest
        pixel of the profile's image when used to simulate imaging data.

        Parameters
        ----------
        signal_to_noise_ratio
            The signal to noise of the light profile when it is used to simulate strong lens imaging.
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
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            radius_break=radius_break,
            intensity_break=0.0,
            alpha=alpha,
            gamma=gamma,
        )

        self.intensity = self.intensity_break

        LightProfileSNR.__init__(self, signal_to_noise_ratio=signal_to_noise_ratio)


class EllChameleon(lp.EllChameleon, LightProfileSNR):
    def __init__(
        self,
        signal_to_noise_ratio: float = 10.0,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        core_radius_0: float = 0.01,
        core_radius_1: float = 0.05,
    ):
        """
        The elliptical Chameleon light profile.

        Profile form:
            mass_to_light_ratio * intensity *\
                (1.0 / Sqrt(x^2 + (y/q)^2 + rc^2) - 1.0 / Sqrt(x^2 + (y/q)^2 + (rc + dr)**2.0))

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
        core_radius_1 : rc + dr is the core size of the second elliptical cored Isothermal profile.
             We use dr here is to avoid negative values.
        """

        super().__init__(
            centre=centre,
            elliptical_comps=elliptical_comps,
            intensity=0.0,
            core_radius_0=core_radius_0,
            core_radius_1=core_radius_1,
        )
        LightProfileSNR.__init__(self, signal_to_noise_ratio=signal_to_noise_ratio)


class SphChameleon(lp.SphChameleon, LightProfileSNR):
    def __init__(
        self,
        signal_to_noise_ratio: float = 10.0,
        centre: Tuple[float, float] = (0.0, 0.0),
        core_radius_0: float = 0.01,
        core_radius_1: float = 0.05,
    ):
        """
        The spherical Chameleon light profile.

        Profile form:
            mass_to_light_ratio * intensity *\
                (1.0 / Sqrt(x^2 + (y/q)^2 + rc^2) - 1.0 / Sqrt(x^2 + (y/q)^2 + (rc + dr)**2.0))

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
        core_radius_1 : rc + dr is the core size of the second elliptical cored Isothermal profile.
             We use dr here is to avoid negative values.
        """

        super().__init__(
            centre=centre,
            intensity=0.0,
            core_radius_0=core_radius_0,
            core_radius_1=core_radius_1,
        )
        LightProfileSNR.__init__(self, signal_to_noise_ratio=signal_to_noise_ratio)


class EllEff(lp.EllEff, LightProfileSNR):
    def __init__(
        self,
        signal_to_noise_ratio: float = 10.0,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        effective_radius: float = 0.6,
        eta: float = 1.5,
    ):
        """
        The elliptical eff light profile, which is commonly used to represent the clumps of Lyman-alpha emitter
        galaxies.

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
            centre=centre,
            elliptical_comps=elliptical_comps,
            intensity=0.0,
            effective_radius=effective_radius,
            eta=eta,
        )
        LightProfileSNR.__init__(self, signal_to_noise_ratio=signal_to_noise_ratio)


class SphEff(lp.SphEff, LightProfileSNR):
    def __init__(
        self,
        signal_to_noise_ratio: float = 10.0,
        centre: Tuple[float, float] = (0.0, 0.0),
        effective_radius: float = 0.6,
        eta: float = 1.5,
    ):
        """
        The spherical eff light profile, which is commonly used to represent the clumps of Lyman-alpha emitter
        galaxies.

        This profile is introduced in the following paper:

        https://arxiv.org/abs/1708.08854

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
            centre=centre, intensity=0.0, effective_radius=effective_radius, eta=eta
        )
        LightProfileSNR.__init__(self, signal_to_noise_ratio=signal_to_noise_ratio)
