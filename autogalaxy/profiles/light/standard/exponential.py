from typing import Tuple

from autogalaxy.profiles.light.base.sersic import EllSersic


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

        See `autogalaxy.profiles.light.light_profiles.LightProfile` for a description of light profile objects.

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

        See `autogalaxy.profiles.light.light_profiles.LightProfile` for a description of light profile objects.

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
