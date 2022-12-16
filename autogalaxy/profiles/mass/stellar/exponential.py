from typing import Tuple

from autogalaxy.profiles.mass.stellar.sersic import Sersic


class Exponential(Sersic):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The Exponential mass profile, the mass profiles of the light profiles that are used to fit and
        subtract the lens model_galaxy's light.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second).
        effective_radius
            The circular radius containing half the light of this profile.
        mass_to_light_ratio
            The mass-to-light ratio of the light profiles
        """
        super().__init__(
            centre=centre,
            ell_comps=ell_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=1.0,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class ExponentialSph(Exponential):
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
            ell_comps=(0.0, 0.0),
            intensity=intensity,
            effective_radius=effective_radius,
            mass_to_light_ratio=mass_to_light_ratio,
        )
