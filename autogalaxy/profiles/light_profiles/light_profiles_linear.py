import numpy as np
from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.light_profiles import light_profiles as lp


class LightProfileLinear(lp.LightProfile, aa.LinearObj):
    def mapping_matrix_from(self, grid: aa.type.Grid2DLike) -> np.ndarray:
        return self.image_2d_from(grid=grid).slim


class EllSersic(lp.AbstractEllSersic, LightProfileLinear):
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
