from typing import Tuple

from autogalaxy.profiles.light.linear.abstract import LightProfileLinear

from autogalaxy.profiles.light import standard as lp
from autogalaxy.profiles import light_and_mass_profiles as lmp


class DevVaucouleurs(lp.DevVaucouleurs, LightProfileLinear):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        effective_radius: float = 0.6,
    ):
        """
        The elliptical DevVaucouleurs light profile.

        See `autogalaxy.profiles.light.light_profiles.LightProfile` for a description of light profile objects.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system, (see the module
            `autogalaxy -> convert.py` for the convention).
        effective_radius
            The circular radius containing half the light of this profile.
        """
        super().__init__(
            centre=centre,
            ell_comps=ell_comps,
            intensity=1.0,
            effective_radius=effective_radius,
        )

    @property
    def lp_cls(self):
        return lp.DevVaucouleurs

    @property
    def lmp_cls(self):
        return lmp.DevVaucouleurs
