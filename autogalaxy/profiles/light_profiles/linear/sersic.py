import inspect
import numpy as np
from typing import Dict, List, Optional, Tuple

from autoconf import cached_property
import autoarray as aa
import autofit as af

from autogalaxy.profiles.light_profiles.light_profiles_operated import (
    LightProfileOperated,
)

from autogalaxy.profiles.light_profiles import light_profiles as lp
from autogalaxy.profiles import light_and_mass_profiles as lmp

from autogalaxy import exc


class EllSersic(lp.EllSersic, LightProfileLinear):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
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
        effective_radius
            The circular radius containing half the light of this profile.
        sersic_index
            Controls the concentration of the profile (lower -> less concentrated, higher -> more concentrated).
        """

        super().__init__(
            centre=centre,
            elliptical_comps=elliptical_comps,
            intensity=1.0,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
        )

    @property
    def lp_cls(self):
        return lp.EllSersic

    @property
    def lmp_cls(self):
        return lmp.EllSersic