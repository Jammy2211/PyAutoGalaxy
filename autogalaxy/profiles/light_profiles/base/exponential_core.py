import numpy as np
from scipy.integrate import quad
from typing import Optional, Tuple, Union

import autoarray as aa

from autogalaxy.profiles.light_profiles.light_profile_decorators import (
    check_operated_only,
)
from autogalaxy.operate.image import OperateImage
from autogalaxy.profiles.geometry_profiles import EllProfile


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
