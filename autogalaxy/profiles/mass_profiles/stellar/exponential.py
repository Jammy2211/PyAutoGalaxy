import copy
import numpy as np
from scipy.special import wofz
from scipy.integrate import quad
from typing import List, Tuple

import autoarray as aa

from autogalaxy.profiles.mass_profiles.abstract.abstract import MassProfile
from autogalaxy.profiles.mass_profiles.abstract.mge import (
    MassProfileMGE,
)
from autogalaxy.profiles.mass_profiles.abstract.cse import (
    MassProfileCSE,
)
from autogalaxy.profiles.mass_profiles.stellar.abstract import StellarProfile

from autogalaxy.profiles.mass_profiles.mass_profiles import psi_from


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
